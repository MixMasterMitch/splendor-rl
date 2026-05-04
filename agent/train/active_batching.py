from __future__ import annotations

import torch


def _max_power_of_two(n: int) -> int:
    out = 1
    while out * 2 <= n:
        out *= 2
    return out


def bucket_sizes(count: int, max_bucket: int) -> list[int]:
    if count <= 0:
        return []
    if max_bucket <= 0:
        raise ValueError(f"max_bucket must be positive, got {max_bucket}")
    max_size = _max_power_of_two(max_bucket)
    remaining = count
    out: list[int] = []
    while remaining > 0:
        cur = max_size
        while cur > remaining:
            cur //= 2
        out.append(cur)
        remaining -= cur
    return out


def bucket_indices(idx: torch.Tensor, max_bucket: int) -> list[torch.Tensor]:
    if idx.ndim != 1:
        raise ValueError(f"idx must be 1D, got shape {tuple(idx.shape)}")
    sizes = bucket_sizes(int(idx.numel()), max_bucket=max_bucket)
    out: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        out.append(idx.narrow(0, start, size))
        start += size
    return out


def should_bucket_compact(active_count: int, full_count: int) -> bool:
    return 0 < active_count < full_count and active_count * 2 <= full_count
