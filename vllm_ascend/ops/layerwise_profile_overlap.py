import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from vllm.forward_context import get_forward_context


_ENABLED = (
    os.getenv("VLLM_ASCEND_OVERLAP_LAYER_PROFILE", "0").lower()
    in {"1", "true", "yes", "on"}
)
_DISABLE_IN_GRAPH = (
    os.getenv("VLLM_ASCEND_OVERLAP_LAYER_PROFILE_DISABLE_IN_GRAPH", "1").lower()
    in {"1", "true", "yes", "on"}
)
_OUTPUT_DIR = os.getenv("VLLM_ASCEND_OVERLAP_LAYER_PROFILE_DIR", "")
_WRITE_LOCK = threading.Lock()
_LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)")


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    rank = os.getenv("RANK")
    return int(rank) if rank is not None else -1


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling):
            try:
                if bool(is_compiling()):
                    return True
            except Exception:
                pass

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling):
            try:
                if bool(is_compiling()):
                    return True
            except Exception:
                pass

    return False


def _is_graph_runtime_active() -> bool:
    if _DISABLE_IN_GRAPH and _is_compiling():
        return True

    try:
        forward_context = get_forward_context()
    except AssertionError:
        return False

    if _DISABLE_IN_GRAPH and bool(getattr(forward_context, "capturing", False)):
        return True

    return False


def _get_phase_and_num_tokens(
    input_tensor: Optional[torch.Tensor],
) -> tuple[str, int]:
    phase = "unknown"
    num_tokens = (
        int(input_tensor.shape[0])
        if input_tensor is not None and input_tensor.ndim > 0
        else -1
    )

    try:
        forward_context = get_forward_context()
    except AssertionError:
        return phase, num_tokens

    context_num_tokens = getattr(forward_context, "num_tokens", None)
    if context_num_tokens is not None:
        num_tokens = int(context_num_tokens)

    attn_metadata = getattr(forward_context, "attn_metadata", None)
    attn_state = getattr(attn_metadata, "attn_state", None)
    if attn_state is None and isinstance(attn_metadata, dict) and attn_metadata:
        first_meta = next(iter(attn_metadata.values()))
        attn_state = getattr(first_meta, "attn_state", None)

    attn_state_name = (
        getattr(attn_state, "name", str(attn_state))
        if attn_state is not None
        else ""
    )
    if attn_state_name in {"DecodeOnly", "SpecDecoding"}:
        phase = "decode"
    elif attn_state_name in {
        "PrefillNoCache",
        "PrefillCacheHit",
        "ChunkedPrefill",
    }:
        phase = "prefill"

    return phase, num_tokens


def _normalize_layer_name(layer_name: str) -> tuple[str, int, str]:
    match = _LAYER_RE.search(layer_name)
    layer_idx = int(match.group(1)) if match else -1

    suffix = layer_name
    marker = f"layers.{layer_idx}."
    if layer_idx >= 0 and marker in layer_name:
        suffix = layer_name.split(marker, 1)[1]

    if "." in suffix:
        parts = suffix.split(".")
        op_name = ".".join(parts[-2:])
    else:
        op_name = suffix

    return layer_name, layer_idx, op_name


def _record_entry(entry: dict[str, Any]) -> None:
    output_dir = Path(_OUTPUT_DIR or os.getcwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir / f"overlap_layer_profile_rank{_get_rank()}_pid{os.getpid()}.jsonl"
    )
    with _WRITE_LOCK:
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _measure_elapsed_us(run: Callable[[], Any]) -> tuple[Any, float]:
    try:
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = run()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end) * 1000.0)
    except Exception:
        torch.npu.synchronize()
        begin = time.perf_counter_ns()
        result = run()
        torch.npu.synchronize()
        elapsed_us = (time.perf_counter_ns() - begin) / 1000.0
        return result, float(elapsed_us)


def maybe_profile_linear(
    layer_name: str,
    module_name: str,
    input_tensor: Optional[torch.Tensor],
    run: Callable[[], Any],
    ignore_graph_guard: bool = False,
) -> Any:
    if not _ENABLED:
        return run()

    if not ignore_graph_guard and _is_graph_runtime_active():
        return run()

    result, elapsed_us = _measure_elapsed_us(run)
    normalized_name, layer_idx, op_name = _normalize_layer_name(layer_name)
    phase, num_tokens = _get_phase_and_num_tokens(input_tensor)
    entry = {
        "path": "overlap",
        "layer_name": normalized_name,
        "layer_idx": layer_idx,
        "op_name": op_name,
        "module_name": module_name,
        "phase": phase,
        "num_tokens": num_tokens,
        "elapsed_us": elapsed_us,
        "rank": _get_rank(),
        "pid": os.getpid(),
    }
    _record_entry(entry)
    return result
