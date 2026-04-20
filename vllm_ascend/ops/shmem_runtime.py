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
_PHASE_HELPERS_IMPORT_ATTEMPTED = False
_GET_FORWARD_CONTEXT_FN = None
_ASCEND_ATTENTION_STATE = None


def _strip_tcp_prefix(ip_port: str) -> str:
    if ip_port.startswith("tcp://"):
        return ip_port[len("tcp://") :]
    return ip_port


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _debug_logging_enabled() -> bool:
    return _env_flag(_SHMEM_DEBUG_ENV)


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


def _build_diagnostics(layer: torch.nn.Module,
                       input_parallel: torch.Tensor) -> tuple[Optional[str], str]:
    weight = getattr(layer, "weight", None)
    quant_method_name = getattr(layer, "_shmem_quant_method_name", "unknown")
    num_rows = 0 if input_parallel.ndim == 0 else input_parallel.numel() // max(input_parallel.shape[-1], 1)

    reason: Optional[str] = None
    if input_parallel.device.type != "npu":
        reason = f"unsupported_input_device:{input_parallel.device.type}"
    elif weight is None or weight.device.type != "npu":
        reason = "unsupported_weight_device"
    elif int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1")) == 2:
        reason = "unsupported_nz_layout"
    elif weight.dtype != input_parallel.dtype:
        reason = "input_weight_dtype_mismatch"
    elif input_parallel.ndim < 2:
        reason = "input_rank_lt_2"
    elif weight.ndim != 2:
        reason = "weight_rank_ne_2"
    elif input_parallel.shape[-1] != weight.shape[-1]:
        reason = "input_weight_shape_mismatch"

    details = (
        f"prefix={getattr(layer, 'prefix', '')}, "
        f"quant_method={quant_method_name}, "
        f"num_rows={num_rows}, "
        f"input_shape={tuple(input_parallel.shape)}, "
        f"weight_shape={tuple(weight.shape) if weight is not None else ()}, "
        f"input_dtype={input_parallel.dtype}, "
        f"weight_dtype={weight.dtype if weight is not None else 'None'}"
    )
    return reason, details


class _ShmemRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._ash = None
        self._shmem_operators = None
        self._operators: dict[tuple[int, int], object] = {}

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

    def get_operator(self, stream_handle: int, block_dims: int):
        with self._lock:
            key = (stream_handle, block_dims)
            operator = self._operators.get(key)
            if operator is None:
                assert self._shmem_operators is not None
                operator = self._shmem_operators.ShmemOperators(block_dims, stream_handle)
                self._operators[key] = operator
            return operator

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


_RUNTIME = _ShmemRuntime()
atexit.register(_RUNTIME.destroy)


def maybe_shmem_matmul_allreduce(
    layer: torch.nn.Module,
    input_parallel: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    reason, details = _build_diagnostics(layer, input_parallel)
    if reason is not None:
        if _debug_logging_enabled():
            logger.info(
                "Skipping shmem matmul-allreduce: reason=%s, %s",
                reason,
                details,
            )
        return None, reason

    init_error = _RUNTIME.ensure_initialized()
    if init_error is not None:
        if _debug_logging_enabled():
            logger.info(
                "Skipping shmem matmul-allreduce: reason=%s, %s",
                init_error,
                details,
            )
        return None, init_error

    try:
        stream_handle = _current_stream_handle()
    except Exception as exc:
        reason = f"current_stream_failed:{exc}"
        if _debug_logging_enabled():
            logger.info(
                "Skipping shmem matmul-allreduce: reason=%s, %s",
                reason,
                details,
            )
        return None, reason

    block_dims = int(
        os.getenv("VLLM_ASCEND_SHMEM_BLOCK_DIMS", str(_DEFAULT_BLOCK_DIMS))
    )
    input_2d = input_parallel.contiguous().reshape(-1, input_parallel.shape[-1])

    try:
        weight_t = _build_weight_for_shmem(layer)
        operator = _RUNTIME.get_operator(stream_handle, block_dims)
        if _debug_logging_enabled():
            logger.warning(
                "Attempting shmem matmul-allreduce: %s",
                details,
            )
        output_2d = torch.empty(
            (input_2d.shape[0], weight_t.shape[1]),
            dtype=input_2d.dtype,
            device=input_2d.device,
        )
        operator.shmem_matmul_allreduce(
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
                details,
            )
        return output, None
    except Exception as exc:
        reason = f"shmem_kernel_failed:{exc}"
        logger.warning(
            "shmem matmul-allreduce failed: reason=%s, %s",
            reason,
            details,
        )
        return None, reason
