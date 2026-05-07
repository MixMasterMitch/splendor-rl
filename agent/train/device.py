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

    Accepted values:
    - ``"cpu"``      → ``"cpu"``
    - ``"cuda"``     → ``"cuda:0"`` (requires CUDA)
    - ``"cuda:N"``   → ``"cuda:N"`` after validating *N* < device_count
    - ``"auto"``/``""`` → ``"cuda:0"`` when CUDA is available, else ``"cpu"``

    Any other string raises :class:`ValueError`.
    """
    req = (requested or "auto").strip().lower()

    if req in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    if req == "cpu":
        return "cpu"

    if req == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Check your PyTorch installation and GPU drivers."
            )
        return "cuda:0"

    # Handle "cuda:N"
    if req.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA requested ({req}) but torch.cuda.is_available() is False"
            )
        try:
            idx = int(req.split(":")[1])
        except ValueError:
            raise ValueError(
                f"unsupported device {requested!r}; use cpu, cuda, cuda:N, or auto"
            )
        count = torch.cuda.device_count()
        if idx < 0 or idx >= count:
            raise ValueError(
                f"CUDA device {idx} not found; {count} devices available"
            )
        return req

    raise ValueError(
        f"unsupported device {requested!r}; use cpu, cuda, cuda:N, or auto"
    )


def device_info(device: str) -> Dict[str, object]:
    """Return metadata about the active device.

    For CUDA devices the dict additionally contains GPU name, VRAM, and
    CUDA / cuDNN version information.
    """
    info: Dict[str, object] = {
        "device": device,
        "torch": torch.__version__,
    }
    if device.startswith("cuda"):
        idx = 0
        if ":" in device:
            idx = int(device.split(":")[1])
        props = torch.cuda.get_device_properties(idx)
        mem_total = props.total_memory / (1024 ** 3)
        mem_free = (props.total_memory - torch.cuda.memory_allocated(idx)) / (1024 ** 3)
        info["gpu_name"] = props.name
        info["gpu_vram_total_gb"] = round(mem_total, 2)
        info["gpu_vram_free_gb"] = round(mem_free, 2)
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    return info


def configure_device(device: str) -> Dict[str, object]:
    """One-time device setup.

    For CUDA devices: enables ``cudnn.benchmark`` and clears the CUDA cache.
    For CPU: configures thread pools via :func:`configure_cpu_threads`.

    Returns the result of :func:`device_info` merged with any setup metadata.
    """
    info = device_info(device)

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        info["cudnn_benchmark"] = True
    else:
        thread_info = configure_cpu_threads()
        info.update(thread_info)

    return info
