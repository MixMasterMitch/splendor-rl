"""Lightweight per-phase timing instrumentation for the training loop."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


class PhaseTimer:
    """Tracks wall-clock time spent in named phases per iteration.

    Usage:
        timer = PhaseTimer(device="cuda:0", sync_cuda=True)
        with timer.phase("selfplay"):
            ...
        with timer.phase("learner"):
            ...
        summary = timer.iter_summary()  # {"selfplay": 12.3, "learner": 8.1}
    """

    def __init__(self, device: str = "cpu", sync_cuda: bool = False):
        self._device = device
        self._sync_cuda = sync_cuda and device.startswith("cuda")
        self._accum: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        if self._sync_cuda:
            import torch
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._sync_cuda:
                import torch
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            self._accum[name] = self._accum.get(name, 0.0) + elapsed

    def iter_summary(self) -> dict[str, float]:
        """Return accumulated phase times and reset for the next iteration."""
        result = dict(self._accum)
        self._accum.clear()
        return result
