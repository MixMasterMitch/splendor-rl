"""CPU vs GPU benchmark for the Splendor RL training pipeline.

Measures throughput of selfplay, learner, and eval stages on each available
device and prints a human-readable comparison table with speedup ratios.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from ..eval.ladder import evaluate
from ..net.model import SplendorNet
from ..train.device import resolve_device
from ..train.learner import make_optimizer, step as learner_step
from ..train.replay_buffer import ReplayBuffer
from ..train.selfplay import run_selfplay


def run_benchmark(
    num_games: int = 128,
    num_sims: int = 4,
    learner_steps: int = 50,
    eval_games: int = 64,
    hidden: int = 192,
    arch: str = "flat",
    devices: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Run timed workloads on each device and return per-device metrics.

    Returns a dict mapping resolved device string to a metrics dict with keys:
        selfplay_games_per_s, selfplay_wall_s,
        learner_steps_per_s, learner_wall_s,
        eval_games_per_s, eval_wall_s
    """
    if devices is None:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

    results: dict[str, dict[str, float]] = {}

    for dev in devices:
        resolved = resolve_device(dev)
        is_cuda = resolved.startswith("cuda")

        torch.manual_seed(42)
        net = SplendorNet(hidden=hidden, arch=arch).to(resolved)
        buffer = ReplayBuffer(capacity=num_games * 200, device=resolved)

        # --- Warmup pass on GPU (excluded from timing) ---
        if is_cuda:
            run_selfplay(
                net,
                buffer,
                num_games=min(16, num_games),
                device=resolved,
                num_sims=num_sims,
                seed=0,
            )
            torch.cuda.synchronize()
            # Reset buffer so warmup data doesn't affect learner benchmark
            buffer = ReplayBuffer(capacity=num_games * 200, device=resolved)

        # --- Benchmark selfplay ---
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        run_selfplay(
            net,
            buffer,
            num_games=num_games,
            device=resolved,
            num_sims=num_sims,
            seed=42,
        )
        if is_cuda:
            torch.cuda.synchronize()
        sp_wall = time.monotonic() - t0

        # --- Benchmark learner ---
        optim = make_optimizer(net)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(learner_steps):
            learner_step(net, buffer, optim, device=resolved)
        if is_cuda:
            torch.cuda.synchronize()
        lr_wall = time.monotonic() - t0

        # --- Benchmark eval ---
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        evaluate(net, num_games=eval_games, device=resolved, num_sims=num_sims)
        if is_cuda:
            torch.cuda.synchronize()
        ev_wall = time.monotonic() - t0

        results[resolved] = {
            "selfplay_games_per_s": num_games / sp_wall if sp_wall > 0 else 0.0,
            "selfplay_wall_s": round(sp_wall, 3),
            "learner_steps_per_s": learner_steps / lr_wall if lr_wall > 0 else 0.0,
            "learner_wall_s": round(lr_wall, 3),
            "eval_games_per_s": eval_games / ev_wall if ev_wall > 0 else 0.0,
            "eval_wall_s": round(ev_wall, 3),
        }

    return results


def _print_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table with optional speedup ratios."""
    devices = list(results.keys())
    metrics = [
        ("Selfplay games/s", "selfplay_games_per_s"),
        ("Selfplay wall (s)", "selfplay_wall_s"),
        ("Learner steps/s", "learner_steps_per_s"),
        ("Learner wall (s)", "learner_wall_s"),
        ("Eval games/s", "eval_games_per_s"),
        ("Eval wall (s)", "eval_wall_s"),
    ]

    # Column widths
    label_w = max(len(m[0]) for m in metrics) + 2
    col_w = 12
    has_speedup = "cpu" in results and len(devices) > 1

    # Header
    header = f"{'Metric':<{label_w}}"
    for d in devices:
        header += f"{d:>{col_w}}"
    if has_speedup:
        header += f"{'Speedup':>{col_w}}"
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for label, key in metrics:
        row = f"{label:<{label_w}}"
        for d in devices:
            val = results[d].get(key, 0.0)
            row += f"{val:>{col_w}.2f}"
        if has_speedup and len(devices) >= 2:
            gpu_dev = [d for d in devices if d != "cpu"]
            if gpu_dev and key.endswith("_per_s"):
                cpu_val = results["cpu"].get(key, 0.0)
                gpu_val = results[gpu_dev[0]].get(key, 0.0)
                if cpu_val > 0:
                    speedup = gpu_val / cpu_val
                    row += f"{speedup:>{col_w - 1}.1f}x"
                else:
                    row += f"{'N/A':>{col_w}}"
            else:
                row += f"{'':>{col_w}}"
        print(row)

    print(sep)
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Benchmark CPU vs GPU training throughput",
    )
    p.add_argument("--num-games", type=int, default=128, help="Selfplay games (default: 128)")
    p.add_argument("--num-sims", type=int, default=4, help="MCTS simulations per move (default: 4)")
    p.add_argument("--learner-steps", type=int, default=50, help="Learner gradient steps (default: 50)")
    p.add_argument("--eval-games", type=int, default=64, help="Eval games (default: 64)")
    p.add_argument("--hidden", type=int, default=192, help="Network hidden dim (default: 192)")
    p.add_argument("--arch", type=str, default="flat", choices=["flat", "attn"], help="Network arch (default: flat)")
    args = p.parse_args(argv)

    # Auto-detect devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device detected — benchmarking CPU only.")

    print(f"Config: {args.num_games} games, {args.num_sims} sims, "
          f"{args.learner_steps} learner steps, {args.eval_games} eval games, "
          f"hidden={args.hidden}, arch={args.arch}")

    results = run_benchmark(
        num_games=args.num_games,
        num_sims=args.num_sims,
        learner_steps=args.learner_steps,
        eval_games=args.eval_games,
        hidden=args.hidden,
        arch=args.arch,
        devices=devices,
    )

    _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
