"""Bounded training entrypoint.

Usage:
    python -m agent.scripts.train --run-id my_run --num-players 2 --max-iters 10 --max-wall-minutes 5

Safe to re-invoke: resumes from latest checkpoint under runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from agent.obs.run import Run
from agent.train.loop import LoopConfig, run_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Splendor RL bounded training burst")
    p.add_argument("--run-id", required=True)
    p.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cuda",
        help="Training device: cpu, cuda, or auto (prefers GPU if available; default: cuda).",
    )
    p.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="Enable mixed-precision (AMP) on CUDA. Ignored on CPU.",
    )
    p.add_argument(
        "--no-compile-net",
        dest="compile_net",
        action="store_false",
        help="Disable torch.compile on the net forward (default: disabled)",
    )
    p.add_argument(
        "--compile-net",
        dest="compile_net",
        action="store_true",
        help="Enable torch.compile on the net forward (experimental on this machine)",
    )
    p.set_defaults(compile_net=False)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument(
        "--arch",
        choices=["attn", "flat"],
        default="attn",
        help="Network architecture for policy/value inference",
    )
    p.add_argument("--selfplay-games", type=int, default=1024)
    p.add_argument(
        "--selfplay-sims",
        type=int,
        default=32,
        help="MCTS sims per selfplay action. Higher = better targets, linearly more wall time.",
    )
    p.add_argument(
        "--selfplay-max-turns",
        type=int,
        default=160,
        help="Maximum turns per self-play game before assigning stall penalties.",
    )
    p.add_argument("--replay-capacity", type=int, default=820_000)
    p.add_argument("--learner-batch", type=int, default=4096)
    p.add_argument("--learner-steps-per-iter", type=int, default=64)
    p.add_argument(
        "--entropy-bonus",
        type=float,
        default=0.015,
        help="Small policy-entropy bonus added during learning to slow premature collapse.",
    )
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--max-wall-minutes", type=float, default=720.0)
    p.add_argument(
        "--init-from",
        default="",
        help=(
            "Warm-start net weights from this checkpoint .pt (only when the run "
            "has no existing checkpoints). Optimizer state and replay buffer "
            "start fresh. Path may be absolute or relative to the workspace."
        ),
    )
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.15,
        help="Root Dirichlet exploration noise alpha (0 disables)",
    )
    p.add_argument(
        "--dirichlet-mix",
        type=float,
        default=0.40,
        help="Fraction of prior replaced by Dirichlet noise at the root",
    )
    p.add_argument(
        "--time-discount",
        type=float,
        default=0.995,
        help="Per-turn discount on value targets to pressure short games",
    )
    p.add_argument(
        "--q-scale",
        type=float,
        default=22.0,
        help="MCTS root Q-value coefficient in improved policy target",
    )
    p.add_argument(
        "--mixed-players",
        type=int,
        nargs="*",
        default=[2, 3, 4],
        help="Rotate selfplay through these player counts (e.g. --mixed-players 2 3 4). Overrides --num-players for selfplay.",
    )
    p.add_argument(
        "--league-selfplay-every",
        type=int,
        default=3,
        help="Every N iters, replace some self-play games with league-opponent games; 0 disables.",
    )
    p.add_argument(
        "--league-max-entries",
        type=int,
        default=24,
        help="Maximum number of archived league checkpoints to keep.",
    )
    p.add_argument(
        "--league-keep-recent",
        type=int,
        default=8,
        help="Number of newest league checkpoints always preserved when culling.",
    )
    p.add_argument("--rating-random-anchor", type=float, default=1000.0)
    # --- Unified eval args ---
    p.add_argument(
        "--eval-games",
        type=int,
        default=512,
        help="Total games in unified eval (mixed 2p/3p/4p).",
    )
    p.add_argument(
        "--eval-sims",
        type=int,
        default=64,
        help="MCTS sims per move for all agents during eval.",
    )
    p.add_argument(
        "--eval-max-turns",
        type=int,
        default=200,
        help="Maximum turns per eval game.",
    )
    p.add_argument(
        "--eval-league-opponents",
        type=int,
        default=4,
        help="Number of random league checkpoints to include in eval.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Run(args.run_id)
    loop_kwargs: dict = dict(
        num_players=args.num_players,
        device=args.device,
        compile_net=args.compile_net,
        hidden=args.hidden,
        arch=args.arch,
        selfplay_games=args.selfplay_games,
        selfplay_sims=args.selfplay_sims,
        selfplay_max_turns=args.selfplay_max_turns,
        replay_capacity=args.replay_capacity,
        learner_batch=args.learner_batch,
        learner_steps_per_iter=args.learner_steps_per_iter,
        entropy_bonus=args.entropy_bonus,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        max_wall_minutes=args.max_wall_minutes,
        league_selfplay_every=args.league_selfplay_every,
        league_max_entries=args.league_max_entries,
        league_keep_recent=args.league_keep_recent,
        rating_random_anchor=args.rating_random_anchor,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_mix=args.dirichlet_mix,
        time_discount=args.time_discount,
        q_scale=args.q_scale,
        init_from=args.init_from,
        use_amp=args.use_amp,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        eval_max_turns=args.eval_max_turns,
        eval_league_opponents=args.eval_league_opponents,
    )
    if args.mixed_players:
        loop_kwargs["mixed_players"] = args.mixed_players
    cfg = LoopConfig(**loop_kwargs)
    try:
        result = run_loop(run, cfg)
    except Exception as exc:
        run.event("loop_crashed", {"err": str(exc), "trace": traceback.format_exc()}, level="ERROR")
        run.close()
        raise
    run.event("train_done", {"result": result})
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
