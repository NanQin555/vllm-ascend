import atexit
import importlib
import os
import threading
from typing import Any, Optional

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_BLOCK_DIMS = 20
_DEFAULT_LOCAL_MEM_SIZE = 1024 * 1024 * 1024
_DEFAULT_IP_PORT = "tcp://127.0.0.1:8667"
_SHMEM_DEBUG_ENV = "VLLM_ASCEND_SHMEM_DEBUG"
_CACHED_BLOCK_DIMS: Optional[int] = None
_KERNEL_NAME_BY_DTYPE = {
    torch.float16: "shmem_matmul_allreduce",
    torch.bfloat16: "shmem_matmul_allreduce_bf16",
}


def _strip_tcp_prefix(ip_port: str) -> str:
    if ip_port.startswith("tcp://"):
        return ip_port[len("tcp://") :]
    return ip_port


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _debug_logging_enabled() -> bool:
    return _env_flag(_SHMEM_DEBUG_ENV)


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
    expected_dtype = getattr(weight, "dtype", None)
    kernel_name = _KERNEL_NAME_BY_DTYPE.get(expected_dtype)

    if weight is None:
        reason = "missing_weight"
    elif int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1")) == 2:
        reason = "unsupported_nz_layout"
    elif weight.ndim != 2:
        reason = "weight_rank_ne_2"
    elif kernel_name is None:
        reason = f"unsupported_weight_dtype:{expected_dtype}"

    setattr(layer, "_shmem_static_reason", reason)
    setattr(layer, "_shmem_expected_dtype", expected_dtype)
    setattr(layer, "_shmem_kernel_name", kernel_name)
    setattr(layer, "_shmem_block_dims", _get_block_dims())


def _ensure_shmem_matmul_allreduce_prepared(layer: torch.nn.Module) -> None:
    if hasattr(layer, "_shmem_static_reason"):
        return
    prepare_shmem_matmul_allreduce(layer)


def _build_debug_details(layer: torch.nn.Module,
                         input_parallel: torch.Tensor,
                         expected_dtype: Optional[torch.dtype] = None) -> str:
    weight = getattr(layer, "weight", None)
    quant_method_name = getattr(layer, "_shmem_quant_method_name", "unknown")
    num_rows = 0 if input_parallel.ndim == 0 else input_parallel.numel() // max(input_parallel.shape[-1], 1)
    return (
        f"prefix={getattr(layer, 'prefix', '')}, "
        f"quant_method={quant_method_name}, "
        f"num_rows={num_rows}, "
        f"input_shape={tuple(input_parallel.shape)}, "
        f"weight_shape={tuple(weight.shape) if weight is not None else ()}, "
        f"expected_dtype={expected_dtype}, "
        f"input_dtype={input_parallel.dtype}, "
        f"weight_dtype={weight.dtype if weight is not None else 'None'}"
    )


def _log_skip_if_debug(layer: torch.nn.Module,
                       input_parallel: torch.Tensor,
                       reason: str,
                       expected_dtype: Optional[torch.dtype] = None) -> None:
    if not _debug_logging_enabled():
        return
    logger.info(
        "Skipping shmem matmul-allreduce: reason=%s, %s",
        reason,
        _build_debug_details(layer, input_parallel, expected_dtype),
    )


class _ShmemRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._ash = None
        self._shmem_operators = None
        self._operators: dict[tuple[int, int], object] = {}
        self._kernel_entries: dict[tuple[int, int, str], Any] = {}

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

            self._ash = ash
            self._shmem_operators = shmem_operators
            self._initialized = True
            logger.info(
                "Initialized shmem runtime for matmul-allreduce: rank=%s world_size=%s ip_port=%s",
                attr.my_rank,
                attr.n_ranks,
                ip_port,
            )
            return None

    def get_kernel_entry(self, stream_handle: int, block_dims: int, kernel_name: str):
        with self._lock:
            key = (stream_handle, block_dims, kernel_name)
            kernel_entry = self._kernel_entries.get(key)
            if kernel_entry is None:
                operator_key = (stream_handle, block_dims)
                operator = self._operators.get(operator_key)
                if operator is None:
                    assert self._shmem_operators is not None
                    operator = self._shmem_operators.ShmemOperators(block_dims, stream_handle)
                    self._operators[operator_key] = operator
                kernel_entry = getattr(operator, kernel_name, None)
                self._kernel_entries[key] = kernel_entry
            return kernel_entry

    def destroy(self) -> None:
        with self._lock:
            if not self._initialized or self._ash is None:
                return
            try:
                self._ash.aclshmem_global_exit(0)
            except Exception:
                logger.exception("Failed to shutdown shmem runtime cleanly")
            finally:
                self._initialized = False
                self._operators.clear()
                self._kernel_entries.clear()


_RUNTIME = _ShmemRuntime()
atexit.register(_RUNTIME.destroy)


def maybe_shmem_matmul_allreduce(
    layer: torch.nn.Module,
    input_parallel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    _ensure_shmem_matmul_allreduce_prepared(layer)

    expected_dtype = getattr(layer, "_shmem_expected_dtype", None)
    static_reason = getattr(layer, "_shmem_static_reason", None)
    if static_reason is not None:
        _log_skip_if_debug(layer, input_parallel, static_reason, expected_dtype)
        return None, static_reason

    if input_parallel.device.type != "npu":
        reason = f"unsupported_input_device:{input_parallel.device.type}"
        _log_skip_if_debug(layer, input_parallel, reason, expected_dtype)
        return None, reason
    if input_parallel.dtype != expected_dtype:
        reason = f"unexpected_input_dtype:{input_parallel.dtype}"
        _log_skip_if_debug(layer, input_parallel, reason, expected_dtype)
        return None, reason
    if input_parallel.ndim < 2:
        reason = "input_rank_lt_2"
        _log_skip_if_debug(layer, input_parallel, reason, expected_dtype)
        return None, reason

    weight = layer.weight
    if input_parallel.shape[-1] != weight.shape[-1]:
        reason = "input_weight_shape_mismatch"
        _log_skip_if_debug(layer, input_parallel, reason, expected_dtype)
        return None, reason

    init_error = _RUNTIME.ensure_initialized()
    if init_error is not None:
        _log_skip_if_debug(layer, input_parallel, init_error, expected_dtype)
        return None, init_error

    try:
        stream_handle = _current_stream_handle()
    except Exception as exc:
        reason = f"current_stream_failed:{exc}"
        _log_skip_if_debug(layer, input_parallel, reason, expected_dtype)
        return None, reason

    input_2d = input_parallel.contiguous().reshape(-1, input_parallel.shape[-1])

    try:
        weight_t = _build_weight_for_shmem(layer)
        block_dims = layer._shmem_block_dims
        kernel_name = layer._shmem_kernel_name
        kernel_entry = _RUNTIME.get_kernel_entry(
            stream_handle, block_dims, kernel_name
        )
        if kernel_entry is None:
            reason = f"missing_shmem_kernel_entry:{kernel_name}"
            _log_skip_if_debug(layer, input_parallel, reason, expected_dtype)
            return None, reason
        if _debug_logging_enabled():
            logger.warning(
                "Attempting shmem matmul-allreduce: %s",
                _build_debug_details(layer, input_parallel, expected_dtype),
            )
        output_2d = torch.empty(
            (input_2d.shape[0], weight_t.shape[1]),
            dtype=input_2d.dtype,
            device=input_2d.device,
        )
        kernel_entry(
            input_2d.data_ptr(),
            weight_t.data_ptr(),
            output_2d.data_ptr(),
            input_2d.shape[0],
            weight_t.shape[1],
            input_2d.shape[1],
        )
        output = output_2d.reshape(*input_parallel.shape[:-1], weight_t.shape[1])
        if bias is not None:
            output = output + bias
        if _debug_logging_enabled():
            logger.info(
                "Using shmem matmul-allreduce successfully: %s",
                _build_debug_details(layer, input_parallel, expected_dtype),
            )
        return output, None
    except Exception as exc:
        reason = f"shmem_kernel_failed:{exc}"
        logger.warning(
            "shmem matmul-allreduce failed: reason=%s, %s",
            reason,
            _build_debug_details(layer, input_parallel, expected_dtype),
        )
        return None, reason
