"""Quick smoke training run to validate the end-to-end loop.

Runs a tiny 2p training burst (few self-play games, few sims, few iters) under
`runs/smoke/`. Finishes in a couple of minutes on CPU; great as a first check
after any code change.
"""

from __future__ import annotations

import argparse
import sys

from agent.obs.run import Run
from agent.train.loop import LoopConfig, run_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick smoke training run")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Training device: cpu, cuda, or auto (prefers GPU if available; default: cpu).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Run("smoke")
    # Smoke leaves compile_net off to keep cold-start fast.
    cfg = LoopConfig(
        num_players=2,
        device=args.device,
        compile_net=False,
        hidden=64,
        arch="flat",
        selfplay_games=32,
        selfplay_sims=2,
        replay_capacity=4000,
        learner_batch=128,
        learner_steps_per_iter=4,
        checkpoint_every=1,
        max_iters=2,
        max_wall_minutes=5.0,
        eval_games=16,
        eval_sims=2,
        eval_max_turns=100,
        eval_league_opponents=0,
    )
    run_loop(run, cfg)
    run.event("smoke_done", {})
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
