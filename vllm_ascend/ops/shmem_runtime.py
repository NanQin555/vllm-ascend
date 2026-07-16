import atexit
import importlib
import os
import threading
from typing import Any, Optional

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_BLOCK_DIMS = 20
_DEFAULT_LOCAL_MEM_SIZE = 1024 * 1024 * 1024
_DEFAULT_IP_PORT = "tcp://127.0.0.1:8667"
_OUTPUT_BUFFER_ALIGNMENT = 512
_CACHED_BLOCK_DIMS: Optional[int] = None
_KERNEL_NAME_BY_DTYPE = {
    torch.bfloat16: "shmem_matmul_allreduce_overlap_bf16",
}


def _strip_tcp_prefix(ip_port: str) -> str:
    if ip_port.startswith("tcp://"):
        return ip_port[len("tcp://") :]
    return ip_port


def _get_block_dims() -> int:
    global _CACHED_BLOCK_DIMS
    if _CACHED_BLOCK_DIMS is None:
        _CACHED_BLOCK_DIMS = int(
            os.getenv("VLLM_ASCEND_SHMEM_BLOCK_DIMS", str(_DEFAULT_BLOCK_DIMS))
        )
    return _CACHED_BLOCK_DIMS


def _current_stream_handle() -> int:
    current_stream = torch.npu.current_stream()
    stream_handle = getattr(current_stream, "npu_stream", None)
    if stream_handle is None:
        raise RuntimeError("current stream does not expose npu_stream")
    return int(stream_handle)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * torch.empty((), dtype=dtype).element_size()


def _get_configured_output_buffer_bytes() -> int:
    value = os.getenv("VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES")
    if value is None or value == "":
        return 0
    buffer_bytes = int(value)
    if buffer_bytes <= 0:
        raise RuntimeError(
            "VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES must be positive"
        )
    return _align_up(buffer_bytes, _OUTPUT_BUFFER_ALIGNMENT)


def _get_prealloc_output_tokens() -> int:
    value = os.getenv("VLLM_ASCEND_SHMEM_OUTPUT_MAX_TOKENS")
    if value is not None and value != "":
        max_tokens = int(value)
        if max_tokens <= 0:
            raise RuntimeError(
                "VLLM_ASCEND_SHMEM_OUTPUT_MAX_TOKENS must be positive"
            )
        return max_tokens

    return int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)


def _is_graph_capturing() -> bool:
    if not is_forward_context_available():
        return False
    return bool(getattr(get_forward_context(), "capturing", False))


def _build_weight_for_shmem(layer: torch.nn.Module) -> torch.Tensor:
    cached = getattr(layer, "_shmem_matmul_allreduce_weight_t", None)
    if cached is not None:
        return cached

    weight_t = layer.weight.transpose(0, 1).contiguous()
    setattr(layer, "_shmem_matmul_allreduce_weight_t", weight_t)
    return weight_t


def prepare_shmem_matmul_allreduce(layer: torch.nn.Module) -> None:
    weight = getattr(layer, "weight", None)
    reason = None
    kernel_name = _KERNEL_NAME_BY_DTYPE.get(getattr(weight, "dtype", None))

    if weight is None:
        reason = "missing_weight"
    elif int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1")) == 2:
        reason = "unsupported_nz_layout"
    elif weight.ndim != 2:
        reason = "weight_rank_ne_2"
    elif kernel_name is None:
        reason = f"unsupported_weight_dtype:{getattr(weight, 'dtype', None)}"

    setattr(layer, "_shmem_static_reason", reason)
    setattr(layer, "_shmem_kernel_name", kernel_name)
    setattr(layer, "_shmem_block_dims", _get_block_dims())
    setattr(layer, "_shmem_kernel_entry", None)
    setattr(layer, "_shmem_matmul_allreduce_weight_t", None)


class _SymmetricOutputBuffer:
    def __init__(
        self,
        ash: Any,
        tensor_from_ptr: Any,
        dtype: torch.dtype,
        device: torch.device,
        requested_bytes: int,
    ) -> None:
        configured_bytes = _get_configured_output_buffer_bytes()
        self.buffer_bytes = max(
            _align_up(requested_bytes, _OUTPUT_BUFFER_ALIGNMENT),
            configured_bytes,
        )
        self.dtype = dtype
        self.device = device
        self._ash = ash
        self._tensor_from_ptr = tensor_from_ptr
        self._ptr = int(ash.aclshmem_malloc(self.buffer_bytes) or 0)
        self._tensors: dict[tuple[int, ...], torch.Tensor] = {}

        if not self._ptr:
            raise RuntimeError(
                "aclshmem_malloc failed for shmem output buffer: "
                f"buffer_bytes={self.buffer_bytes}"
            )

    def _ensure_tensor_for_shape(self, shape: tuple[int, ...]) -> None:
        if shape not in self._tensors:
            self._tensors[shape] = self._tensor_from_ptr(
                self._ptr,
                shape,
                self.dtype,
                self.device,
            )

    def make_tensor(self, shape: tuple[int, ...], requested_bytes: int) -> torch.Tensor:
        if requested_bytes > self.buffer_bytes:
            raise RuntimeError(
                "shmem output shape exceeds fixed buffer capacity: "
                f"requested_bytes={requested_bytes} "
                f"buffer_bytes={self.buffer_bytes}. "
                "Set VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES to the maximum "
                "captured output size."
            )
        if shape not in self._tensors:
            if _is_graph_capturing():
                raise RuntimeError(
                    "shmem output tensor wrapper was not prepared before "
                    "graph capture: "
                    f"shape={shape}. Run a non-graph warmup for this "
                    "capture shape before ACL graph capture."
                )
            self._ensure_tensor_for_shape(shape)
        return self._tensors[shape]

    def free(self) -> None:
        self._tensors.clear()
        if self._ptr:
            self._ash.aclshmem_free(self._ptr)
            self._ptr = 0


class _ShmemRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._ash = None
        self._tensor_from_ptr = None
        self._shmem_operators = None
        self._operators: dict[int, object] = {}
        self._kernel_entries: dict[tuple[int, str], Any] = {}
        self._output_buffers: dict[tuple[torch.dtype, str], _SymmetricOutputBuffer] = {}
        self._moe_workspaces: dict[tuple[str, int, int, int, int, int],
                                   torch.Tensor] = {}

    def ensure_initialized(self) -> Optional[str]:
        with self._lock:
            if self._initialized:
                return None

            try:
                ash = importlib.import_module("shmem")
            except ImportError as exc:
                return f"missing_shmem_runtime:{exc}"

            try:
                shmem_operators = importlib.import_module("shmem_operators")
            except ImportError as exc:
                return f"missing_shmem_operators:{exc}"

            if not dist.is_initialized():
                return "torch_distributed_not_initialized"

            ip_port = os.getenv("VLLM_ASCEND_SHMEM_IP_PORT", _DEFAULT_IP_PORT)
            os.environ.setdefault("SHMEM_UID_SESSION_ID", _strip_tcp_prefix(ip_port))

            attr = ash.InitAttr()
            attr.my_rank = dist.get_rank()
            attr.n_ranks = dist.get_world_size()
            attr.local_mem_size = int(
                os.getenv(
                    "VLLM_ASCEND_SHMEM_LOCAL_MEM_SIZE",
                    str(_DEFAULT_LOCAL_MEM_SIZE),
                )
            )
            attr.ip_port = ip_port

            ret = ash.aclshmem_init(attr)
            if ret != 0:
                return f"aclshmem_init_failed:{ret}"

            tensor_from_ptr = getattr(ash, "construct_tensor_from_ptr", None)
            if tensor_from_ptr is None:
                tensor_module = importlib.import_module("shmem.construct_tensor")
                tensor_from_ptr = tensor_module.construct_tensor_from_ptr

            self._ash = ash
            self._tensor_from_ptr = tensor_from_ptr
            self._shmem_operators = shmem_operators
            self._initialized = True
            logger.info(
                "Initialized shmem runtime for matmul-allreduce: rank=%s world_size=%s ip_port=%s",
                attr.my_rank,
                attr.n_ranks,
                ip_port,
            )
            return None

    def get_kernel_entry(self, block_dims: int, kernel_name: str):
        with self._lock:
            key = (block_dims, kernel_name)
            kernel_entry = self._kernel_entries.get(key)
            if kernel_entry is None:
                operator = self._operators.get(block_dims)
                if operator is None:
                    assert self._shmem_operators is not None
                    operator = self._shmem_operators.ShmemOperators(block_dims)
                    self._operators[block_dims] = operator
                kernel_entry = getattr(operator, kernel_name, None)
                self._kernel_entries[key] = kernel_entry
            return kernel_entry

    def get_symmetric_output(
        self,
        layer: torch.nn.Module,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        del layer
        return self._get_symmetric_output(shape, dtype, device)

    def get_moe_workspace(
        self,
        device: torch.device,
        m: int,
        hidden_size: int,
        gate_up_size: int,
        top_k: int,
        expert_per_rank: int,
        required_bytes: int,
    ) -> torch.Tensor:
        with self._lock:
            device_id = device.index
            if device_id is None:
                device_id = torch.npu.current_device()
            normalized_device = torch.device(f"npu:{device_id}")
            key = (
                str(normalized_device),
                m,
                hidden_size,
                gate_up_size,
                top_k,
                expert_per_rank,
            )
            workspace = self._moe_workspaces.get(key)
            if workspace is None:
                if _is_graph_capturing():
                    raise RuntimeError(
                        "SHMEM MoE workspace was not allocated before graph "
                        "capture; run an eager warmup for every captured token "
                        "shape")
                workspace = torch.empty(
                    required_bytes,
                    dtype=torch.uint8,
                    device=normalized_device,
                )
                self._moe_workspaces[key] = workspace
                logger.info(
                    "Allocated SHMEM MoE workspace: device=%s shape=(%s,%s,%s) "
                    "top_k=%s local_experts=%s bytes=%s",
                    normalized_device,
                    m,
                    hidden_size,
                    gate_up_size,
                    top_k,
                    expert_per_rank,
                    required_bytes,
                )
            elif workspace.numel() < required_bytes:
                raise RuntimeError(
                    "cached SHMEM MoE workspace is smaller than the kernel "
                    f"request: cached={workspace.numel()} required={required_bytes}")
            return workspace

    def prepare_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._prepare_symmetric_output(shape, dtype, device)

    def _prepare_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._get_symmetric_output(shape, dtype, device)

    def _get_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        with self._lock:
            assert self._ash is not None
            assert self._tensor_from_ptr is not None
            device_id = device.index
            if device_id is None:
                device_id = torch.npu.current_device()
            normalized_device = torch.device(f"npu:{device_id}")
            key = (dtype, str(normalized_device))
            requested_bytes = _tensor_nbytes(shape, dtype)
            buffer = self._output_buffers.get(key)
            if buffer is None:
                if _is_graph_capturing():
                    raise RuntimeError(
                        "shmem output buffer was not allocated before graph "
                        "capture. Run a non-graph warmup before ACL graph "
                        "capture, or set VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES "
                        "and initialize the runtime before capture."
                    )
                buffer = _SymmetricOutputBuffer(
                    self._ash,
                    self._tensor_from_ptr,
                    dtype,
                    normalized_device,
                    requested_bytes,
                )
                self._output_buffers[key] = buffer
                logger.info(
                    "Allocated shmem output buffer: dtype=%s device=%s "
                    "buffer_bytes=%s",
                    dtype,
                    normalized_device,
                    buffer.buffer_bytes,
                )
            return buffer.make_tensor(shape, requested_bytes)

    def destroy(self) -> None:
        with self._lock:
            if not self._initialized or self._ash is None:
                return
            try:
                for buffer in self._output_buffers.values():
                    try:
                        buffer.free()
                    except Exception:
                        logger.exception("Failed to free shmem output buffer")
                self._ash.aclshmem_global_exit(0)
            except Exception:
                logger.exception("Failed to shutdown shmem runtime cleanly")
            finally:
                self._initialized = False
                self._operators.clear()
                self._kernel_entries.clear()
                self._output_buffers.clear()
                self._moe_workspaces.clear()


_RUNTIME = _ShmemRuntime()
atexit.register(_RUNTIME.destroy)


def finalize_shmem_matmul_allreduce(layer: torch.nn.Module) -> None:
    if not getattr(layer, "_can_try_shmem_matmul_allreduce", False):
        return

    if getattr(layer, "_shmem_static_reason", None) is not None:
        return

    weight_t = _build_weight_for_shmem(layer)
    if getattr(layer, "_shmem_kernel_entry", None) is not None:
        return
    if _RUNTIME.ensure_initialized() is not None:
        return
    layer._shmem_kernel_entry = _RUNTIME.get_kernel_entry(
        layer._shmem_block_dims, layer._shmem_kernel_name
    )
    _RUNTIME.prepare_symmetric_output(
        (_get_prealloc_output_tokens(), int(weight_t.shape[1])),
        weight_t.dtype,
        weight_t.device,
    )


def maybe_shmem_matmul_allreduce(
    layer: torch.nn.Module,
    input_parallel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    static_reason = getattr(layer, "_shmem_static_reason", None)
    if static_reason is not None:
        raise RuntimeError(f"shmem matmul-allreduce disabled: {static_reason}")

    weight_t = getattr(layer, "_shmem_matmul_allreduce_weight_t", None)
    if weight_t is None:
        raise RuntimeError("shmem matmul-allreduce weight is not finalized")
    if input_parallel.dtype != weight_t.dtype:
        raise RuntimeError(
            "shmem matmul-allreduce requires input and weight to use the "
            "same dtype: "
            f"input_dtype={input_parallel.dtype} weight_dtype={weight_t.dtype}"
        )
    if input_parallel.shape[-1] != weight_t.shape[0]:
        raise RuntimeError(
            "shmem matmul-allreduce input/weight shape mismatch: "
            f"input_k={input_parallel.shape[-1]} weight_k={weight_t.shape[0]}"
        )

    kernel_entry = getattr(layer, "_shmem_kernel_entry", None)
    if kernel_entry is None:
        raise RuntimeError("shmem matmul-allreduce kernel entry is not initialized")

    if input_parallel.is_contiguous():
        input_2d = input_parallel.reshape(-1, input_parallel.shape[-1])
    else:
        input_2d = input_parallel.contiguous().reshape(-1, input_parallel.shape[-1])

    stream_handle = _current_stream_handle()
    output_2d = _RUNTIME.get_symmetric_output(
        layer,
        (input_2d.shape[0], weight_t.shape[1]),
        input_2d.dtype,
        input_2d.device,
    )
    kernel_entry(
        input_2d.data_ptr(),
        weight_t.data_ptr(),
        output_2d.data_ptr(),
        input_2d.shape[0],
        weight_t.shape[1],
        input_2d.shape[1],
        stream_handle,
    )
    if bias is not None:
        raise RuntimeError(
            "shmem matmul-allreduce overlap returns symmetric output directly "
            "and does not support fused bias"
        )
    return output_2d.reshape(*input_parallel.shape[:-1], weight_t.shape[1])


def shmem_dispatch_ffn_combine(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the experimental BF16 catccos/SHMEM EP MoE kernel.

    The device kernel consumes row-major weights laid out as
    ``[local_experts, K, 2 * intermediate]`` and
    ``[local_experts, intermediate, K]``. Communication is performed over the
    already initialized SHMEM world, which must match the vLLM EP group.
    """
    tensors = {
        "hidden_states": hidden_states,
        "w13": w13,
        "w2": w2,
    }
    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise RuntimeError(
                f"SHMEM MoE only supports BF16, but {name} has {tensor.dtype}")
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"SHMEM MoE requires contiguous row-major {name}; got "
                f"shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}")

    if hidden_states.ndim < 2:
        raise RuntimeError("SHMEM MoE hidden_states must have rank >= 2")
    hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    if w13.ndim != 3 or w2.ndim != 3:
        raise RuntimeError(
            "SHMEM MoE weights must be rank-3 [local_experts, K, N]")

    m, hidden_size = hidden_2d.shape
    expert_per_rank, w13_k, gate_up_size = w13.shape
    w2_experts, intermediate_size, w2_n = w2.shape
    if w13_k != hidden_size or w2_experts != expert_per_rank or \
            intermediate_size * 2 != gate_up_size or w2_n != hidden_size:
        raise RuntimeError(
            "SHMEM MoE weight shape mismatch: "
            f"hidden={tuple(hidden_2d.shape)} w13={tuple(w13.shape)} "
            f"w2={tuple(w2.shape)}")

    if topk_ids.ndim != 2 or topk_weights.ndim != 2 or \
            topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != m:
        raise RuntimeError(
            "SHMEM MoE expects matching [M, top_k] ids and weights: "
            f"ids={tuple(topk_ids.shape)} weights={tuple(topk_weights.shape)} "
            f"M={m}")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise RuntimeError("SHMEM MoE expert ids must be contiguous int32")
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        raise RuntimeError("SHMEM MoE top-k weights must be contiguous float32")

    init_reason = _RUNTIME.ensure_initialized()
    if init_reason is not None:
        raise RuntimeError(f"SHMEM MoE runtime initialization failed: {init_reason}")

    block_dims = _get_block_dims()
    workspace_size_entry = _RUNTIME.get_kernel_entry(
        block_dims, "shmem_moe_workspace_size_bf16")
    kernel_entry = _RUNTIME.get_kernel_entry(
        block_dims, "shmem_dispatch_ffn_combine_bf16")
    if workspace_size_entry is None or kernel_entry is None:
        raise RuntimeError(
            "installed shmem_operators module does not contain the MoE "
            "DispatchFFNCombine entry points")

    top_k = int(topk_ids.shape[1])
    # Reuse one workspace for a power-of-two token bucket. Exact-M caching
    # would retain a very large buffer for every prefill length seen by the
    # engine and quickly exhaust NPU memory.
    workspace_m = 1 << (int(m) - 1).bit_length()
    required_bytes = int(
        workspace_size_entry(
            workspace_m,
            hidden_size,
            gate_up_size,
            top_k,
            expert_per_rank,
        ))
    workspace = _RUNTIME.get_moe_workspace(
        hidden_2d.device,
        workspace_m,
        hidden_size,
        gate_up_size,
        top_k,
        expert_per_rank,
        required_bytes,
    )
    output = torch.empty_like(hidden_2d)
    kernel_entry(
        hidden_2d.data_ptr(),
        w13.data_ptr(),
        w2.data_ptr(),
        topk_ids.data_ptr(),
        topk_weights.data_ptr(),
        output.data_ptr(),
        workspace.data_ptr(),
        required_bytes,
        m,
        hidden_size,
        gate_up_size,
        top_k,
        expert_per_rank,
        _current_stream_handle(),
    )
    return output.reshape(*hidden_states.shape[:-1], hidden_size)
