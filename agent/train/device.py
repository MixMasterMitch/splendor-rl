"""Device resolution helper for the maintained CPU-first trainer."""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch


def configure_cpu_threads(num_threads: Optional[int] = None) -> Dict[str, int]:
    """Configure PyTorch CPU thread pools for this process.

    Policy:
    - If `num_threads` is explicitly provided, use it.
    - Else if env `SPLENDOR_NUM_THREADS` is set, use that.
    - Else if env `OMP_NUM_THREADS` is set, leave it alone (respect outer config).
    - Else set to `os.cpu_count()` so we use every physical core.

    Intraop threads (used by MKL/OpenMP for matmul and elementwise ops) are the
    ones that matter for net forward/backward. Interop threads (scheduling of
    parallel ops) are left at their default.

    Safe to call multiple times; PyTorch enforces a single set on the intraop
    pool so later calls may be ignored once any op has run.
    """
    total = os.cpu_count() or 1
    if num_threads is None:
        env = os.environ.get("SPLENDOR_NUM_THREADS")
        if env:
            try:
                num_threads = int(env)
            except ValueError:
                num_threads = None
    if num_threads is None and not os.environ.get("OMP_NUM_THREADS"):
        num_threads = total
    if num_threads is not None:
        num_threads = max(1, min(num_threads, total))
        try:
            torch.set_num_threads(num_threads)
        except RuntimeError:
            # Already set/used; PyTorch disallows changing after first op.
            pass
    return {
        "cpu_count": total,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def resolve_device(requested: str) -> str:
    """Return a concrete torch device string given a user request.

    The maintained trainer is CPU-only. "auto" and "cpu" both resolve to "cpu";
    any other device request is rejected so stale GPU commands fail fast.
    """
    req = (requested or "auto").lower()
    if req in ("", "auto", "cpu"):
        return "cpu"
    raise ValueError(
        f"unsupported device {requested!r}; the maintained Splendor trainer is CPU-only"
    )


def device_info(device: str) -> Dict[str, object]:
    return {
        "device": device,
        "torch": torch.__version__,
    }
