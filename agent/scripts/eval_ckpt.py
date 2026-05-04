"""Standalone checkpoint evaluator.

Loads a trained SplendorNet checkpoint and runs the ladder evaluator against
reference bots (random + heuristic). Useful for:

- Verifying the true strength of a saved checkpoint at high sim budget.
- Comparing different eval_sims settings without restarting training.
- Producing a portable strength snapshot for any .pt file.

Usage (via Bazel):

    bazel run --config=mlinfra_v7 \\
        //experimental/mloeppky/splendor/agent/scripts:eval_ckpt -- \\
        --ckpt experimental/mloeppky/splendor/agent/runs/real30_v6/checkpoints/iter_000030.pt \\
        --num-games 256 --num-sims 16

The script prints a single JSON line with the metrics and exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _resolve_ckpt(path: str) -> str:
    """Resolve a checkpoint path, falling back to the CWD."""
    if os.path.isabs(path):
        return path
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a SplendorNet checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--num-games", type=int, default=256)
    p.add_argument("--num-sims", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        choices=["cpu"],
        default="cpu",
        help="Eval device (CPU-only; default: cpu)",
    )
    args = p.parse_args()

    from agent.eval import ladder
    from agent.train import checkpointing as CK
    from agent.train import device as D

    D.configure_cpu_threads()

    ckpt_path = _resolve_ckpt(args.ckpt)
    if not os.path.exists(ckpt_path):
        print(
            json.dumps({"error": f"checkpoint not found: {ckpt_path}"}),
            file=sys.stderr,
        )
        return 1

    net, payload = CK.load_net_from_checkpoint(ckpt_path, map_location=args.device)
    spec = CK.checkpoint_net_spec(payload)
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}
    hidden = spec.hidden
    arch = spec.arch
    num_players = int(cfg.get("num_players", args.num_players))

    net.to(args.device)
    net.eval()

    t0 = time.monotonic()
    results = ladder.evaluate(
        net,
        num_players=num_players,
        num_games=args.num_games,
        device=args.device,
        num_sims=args.num_sims,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    wall_s = time.monotonic() - t0

    out = {
        "ckpt": ckpt_path,
        "iteration": payload.get("iteration"),
        "hidden": hidden,
        "arch": arch,
        "num_players": num_players,
        "num_games": args.num_games,
        "num_sims": args.num_sims,
        "wall_s": round(wall_s, 2),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items()},
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
