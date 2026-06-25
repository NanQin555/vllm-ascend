import atexit
import importlib
import os
import re
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
_OWNER_BASE_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_KERNEL_NAME_BY_DTYPE = {
    torch.bfloat16: "shmem_matmul_allreduce_overlap_bf16",
}
_AG_KERNEL_NAME_BY_DTYPE = {
    torch.float16: "shmem_allgather_matmul",
    torch.bfloat16: "shmem_allgather_matmul_bf16",
}
_RS_KERNEL_NAME_BY_DTYPE = {
    torch.float16: "shmem_matmul_reduce_scatter",
    torch.bfloat16: "shmem_matmul_reduce_scatter_bf16",
}


def shmem_sequence_parallel_enabled() -> bool:
    return os.getenv(
        "VLLM_ASCEND_ENABLE_SHMEM_SEQUENCE_PARALLEL", "0"
    ).lower() in {"1", "true", "yes", "on"}


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


def _stable_owner_base(prefix: str) -> int:
    configured = os.getenv("VLLM_ASCEND_SHMEM_OWNER_BASE")
    if configured is not None and configured != "":
        return int(configured)

    match = _OWNER_BASE_LAYER_RE.search(prefix)
    if match is not None:
        op_offset = 1 if "down_proj" in prefix else 0
        return int(match.group(1)) * 2 + op_offset

    value = 2166136261
    for byte in prefix.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xffffffff
    return value


def _is_graph_capturing() -> bool:
    if not is_forward_context_available():
        return False
    return bool(getattr(get_forward_context(), "capturing", False))


def _build_transposed_weight(layer: torch.nn.Module, attr_name: str) -> torch.Tensor:
    cached = getattr(layer, attr_name, None)
    if cached is not None:
        return cached

    weight_t = layer.weight.transpose(0, 1).contiguous()
    setattr(layer, attr_name, weight_t)
    return weight_t


def _build_weight_for_shmem(layer: torch.nn.Module) -> torch.Tensor:
    return _build_transposed_weight(layer, "_shmem_matmul_allreduce_weight_t")


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
    prefix = str(getattr(layer, "unique_prefix", None) or getattr(layer, "prefix", ""))
    setattr(layer, "_shmem_owner_base", _stable_owner_base(prefix))
    setattr(layer, "_shmem_kernel_entry", None)
    setattr(layer, "_shmem_matmul_allreduce_weight_t", None)


def prepare_shmem_sequence_parallel(layer: torch.nn.Module, kind: str) -> None:
    weight = getattr(layer, "weight", None)
    reason = None
    if kind == "allgather_matmul":
        kernel_name = _AG_KERNEL_NAME_BY_DTYPE.get(getattr(weight, "dtype", None))
    elif kind == "matmul_reduce_scatter":
        kernel_name = _RS_KERNEL_NAME_BY_DTYPE.get(getattr(weight, "dtype", None))
    else:
        raise RuntimeError(f"unknown shmem sequence parallel kind: {kind}")

    if weight is None:
        reason = "missing_weight"
    elif int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1")) == 2:
        reason = "unsupported_nz_layout"
    elif weight.ndim != 2:
        reason = "weight_rank_ne_2"
    elif kernel_name is None:
        reason = f"unsupported_weight_dtype:{getattr(weight, 'dtype', None)}"

    setattr(layer, "_shmem_sequence_static_reason", reason)
    setattr(layer, "_shmem_sequence_kind", kind)
    setattr(layer, "_shmem_sequence_kernel_name", kernel_name)
    setattr(layer, "_shmem_sequence_block_dims", _get_block_dims())
    setattr(layer, "_shmem_sequence_kernel_entry", None)
    setattr(layer, "_shmem_sequence_weight_t", None)


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
        self._output_buffers: dict[
            tuple[torch.dtype, str, str], _SymmetricOutputBuffer
        ] = {}

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
        purpose: str = "default",
    ) -> torch.Tensor:
        del layer
        return self._get_symmetric_output(shape, dtype, device, purpose)

    def prepare_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        purpose: str = "default",
    ) -> None:
        self._prepare_symmetric_output(shape, dtype, device, purpose)

    def _prepare_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        purpose: str,
    ) -> None:
        self._get_symmetric_output(shape, dtype, device, purpose)

    def _get_symmetric_output(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        purpose: str,
    ) -> torch.Tensor:
        with self._lock:
            assert self._ash is not None
            assert self._tensor_from_ptr is not None
            device_id = device.index
            if device_id is None:
                device_id = torch.npu.current_device()
            normalized_device = torch.device(f"npu:{device_id}")
            key = (dtype, str(normalized_device), purpose)
            requested_bytes = _tensor_nbytes(shape, dtype)
            buffer = self._output_buffers.get(key)
            if buffer is not None and requested_bytes > buffer.buffer_bytes:
                if _is_graph_capturing():
                    raise RuntimeError(
                        "shmem output shape exceeds fixed buffer capacity "
                        "during graph capture: "
                        f"requested_bytes={requested_bytes} "
                        f"buffer_bytes={buffer.buffer_bytes}. "
                        "Run a non-graph warmup for the maximum captured "
                        "shape, or set VLLM_ASCEND_SHMEM_OUTPUT_BUFFER_BYTES."
                    )
                logger.info(
                    "Growing shmem output buffer: dtype=%s device=%s "
                    "purpose=%s old_buffer_bytes=%s requested_bytes=%s",
                    dtype,
                    normalized_device,
                    purpose,
                    buffer.buffer_bytes,
                    requested_bytes,
                )
                buffer.free()
                buffer = None
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


def finalize_shmem_sequence_parallel(layer: torch.nn.Module) -> None:
    if not getattr(layer, "_can_try_shmem_sequence_parallel", False):
        return

    if getattr(layer, "_shmem_sequence_static_reason", None) is not None:
        return

    weight_t = _build_transposed_weight(layer, "_shmem_sequence_weight_t")
    if getattr(layer, "_shmem_sequence_kernel_entry", None) is not None:
        return
    if _RUNTIME.ensure_initialized() is not None:
        return

    layer._shmem_sequence_kernel_entry = _RUNTIME.get_kernel_entry(
        layer._shmem_sequence_block_dims,
        layer._shmem_sequence_kernel_name,
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    max_tokens = _get_prealloc_output_tokens()
    local_tokens = (max_tokens + world_size - 1) // world_size
    padded_tokens = local_tokens * world_size
    if layer._shmem_sequence_kind == "allgather_matmul":
        output_shape = (padded_tokens, int(weight_t.shape[1]))
        purpose = "sp_allgather_matmul"
    else:
        output_shape = (local_tokens, int(weight_t.shape[1]))
        purpose = "sp_matmul_reduce_scatter"

    _RUNTIME.prepare_symmetric_output(
        output_shape,
        weight_t.dtype,
        weight_t.device,
        purpose,
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
        int(getattr(layer, "_shmem_owner_base", 0xffffffff)),
    )
    if bias is not None:
        raise RuntimeError(
            "shmem matmul-allreduce overlap returns symmetric output directly "
            "and does not support fused bias"
        )
    return output_2d.reshape(*input_parallel.shape[:-1], weight_t.shape[1])


def _get_sequence_weight(layer: torch.nn.Module) -> torch.Tensor:
    weight_t = getattr(layer, "_shmem_sequence_weight_t", None)
    if weight_t is None:
        raise RuntimeError("shmem sequence-parallel weight is not finalized")
    return weight_t


def _get_sequence_kernel(layer: torch.nn.Module):
    static_reason = getattr(layer, "_shmem_sequence_static_reason", None)
    if static_reason is not None:
        raise RuntimeError(f"shmem sequence parallel disabled: {static_reason}")

    kernel_entry = getattr(layer, "_shmem_sequence_kernel_entry", None)
    if kernel_entry is None:
        raise RuntimeError("shmem sequence-parallel kernel entry is not initialized")
    return kernel_entry


def maybe_shmem_allgather_matmul(
    layer: torch.nn.Module,
    input_local: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weight_t = _get_sequence_weight(layer)
    if input_local.dtype != weight_t.dtype:
        raise RuntimeError(
            "shmem allgather-matmul requires input and weight to use the "
            f"same dtype: input_dtype={input_local.dtype} weight_dtype={weight_t.dtype}"
        )
    if input_local.shape[-1] != weight_t.shape[0]:
        raise RuntimeError(
            "shmem allgather-matmul input/weight shape mismatch: "
            f"input_k={input_local.shape[-1]} weight_k={weight_t.shape[0]}"
        )
    if bias is not None:
        raise RuntimeError("shmem allgather-matmul does not support fused bias")

    kernel_entry = _get_sequence_kernel(layer)
    input_2d = input_local.reshape(-1, input_local.shape[-1])
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()

    world_size = dist.get_world_size()
    output_rows = input_2d.shape[0] * world_size
    output_2d = _RUNTIME.get_symmetric_output(
        layer,
        (output_rows, weight_t.shape[1]),
        input_2d.dtype,
        input_2d.device,
        "sp_allgather_matmul",
    )
    kernel_entry(
        input_2d.data_ptr(),
        weight_t.data_ptr(),
        output_2d.data_ptr(),
        input_2d.shape[0],
        weight_t.shape[1],
        input_2d.shape[1],
        _current_stream_handle(),
    )

    try:
        forward_context = get_forward_context()
        pad_size = int(getattr(forward_context, "pad_size", 0))
    except AssertionError:
        pad_size = 0
    if pad_size > 0:
        output_2d = output_2d[:-pad_size]
        output_rows -= pad_size
    return output_2d.reshape(output_rows, weight_t.shape[1])


def maybe_shmem_matmul_reduce_scatter(
    layer: torch.nn.Module,
    input_parallel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    weight_t = _get_sequence_weight(layer)
    if input_parallel.dtype != weight_t.dtype:
        raise RuntimeError(
            "shmem matmul-reduce-scatter requires input and weight to use the "
            f"same dtype: input_dtype={input_parallel.dtype} weight_dtype={weight_t.dtype}"
        )
    if input_parallel.shape[-1] != weight_t.shape[0]:
        raise RuntimeError(
            "shmem matmul-reduce-scatter input/weight shape mismatch: "
            f"input_k={input_parallel.shape[-1]} weight_k={weight_t.shape[0]}"
        )

    kernel_entry = _get_sequence_kernel(layer)
    input_2d = input_parallel.reshape(-1, input_parallel.shape[-1])
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()

    world_size = dist.get_world_size()
    if input_2d.shape[0] % world_size != 0:
        raise RuntimeError(
            "shmem matmul-reduce-scatter requires token count divisible by "
            f"world_size: tokens={input_2d.shape[0]} world_size={world_size}"
        )
    output_rows = input_2d.shape[0] // world_size
    output_2d = _RUNTIME.get_symmetric_output(
        layer,
        (output_rows, weight_t.shape[1]),
        input_2d.dtype,
        input_2d.device,
        "sp_matmul_reduce_scatter",
    )
    kernel_entry(
        input_2d.data_ptr(),
        weight_t.data_ptr(),
        output_2d.data_ptr(),
        input_2d.shape[0],
        weight_t.shape[1],
        input_2d.shape[1],
        _current_stream_handle(),
    )
    if bias is not None:
        output_2d = output_2d + bias
    return output_2d.reshape(output_rows, weight_t.shape[1])
