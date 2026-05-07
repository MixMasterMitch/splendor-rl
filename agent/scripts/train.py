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
        default="cpu",
        help="Training device: cpu, cuda, or auto (prefers GPU if available; default: cpu).",
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
    async_eval_group = p.add_mutually_exclusive_group()
    async_eval_group.add_argument(
        "--async-eval",
        dest="async_eval",
        action="store_true",
        default=None,
        help="Enable async CPU eval subprocess (default: auto based on device).",
    )
    async_eval_group.add_argument(
        "--no-async-eval",
        dest="async_eval",
        action="store_false",
        default=None,
        help="Disable async CPU eval subprocess.",
    )
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument(
        "--arch",
        choices=["attn", "flat"],
        default="flat",
        help="Network architecture for policy/value inference",
    )
    p.add_argument("--selfplay-games", type=int, default=512)
    p.add_argument(
        "--selfplay-sims",
        type=int,
        default=16,
        help="MCTS sims per selfplay action. Higher = better targets, linearly more wall time.",
    )
    p.add_argument(
        "--selfplay-max-turns",
        type=int,
        default=160,
        help="Maximum turns per self-play game before assigning stall penalties.",
    )
    p.add_argument("--replay-capacity", type=int, default=600_000)
    p.add_argument("--learner-batch", type=int, default=256)
    p.add_argument("--learner-steps-per-iter", type=int, default=192)
    p.add_argument(
        "--entropy-bonus",
        type=float,
        default=0.015,
        help="Small policy-entropy bonus added during learning to slow premature collapse.",
    )
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument(
        "--eval-games",
        type=int,
        default=128,
        help="Total eval games per opponent. Split evenly across seats; keep per_seat >= 128.",
    )
    p.add_argument("--eval-sims", type=int, default=4)
    p.add_argument(
        "--eval-max-turns",
        type=int,
        default=200,
        help="Maximum turns per eval game before treating it as unfinished.",
    )
    p.add_argument(
        "--rank-eval-games",
        type=int,
        default=512,
        help="Larger evaluation batch used for checkpoint ranking candidates.",
    )
    p.add_argument(
        "--rank-eval-sims",
        type=int,
        default=16,
        help="Search budget for the larger checkpoint-ranking eval tier.",
    )
    p.add_argument(
        "--rank-eval-max-turns",
        type=int,
        default=200,
        help="Turn cap for checkpoint-ranking eval games.",
    )
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--max-wall-minutes", type=float, default=120.0)
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
        default=10.0,
        help="MCTS root Q-value coefficient in improved policy target",
    )
    p.add_argument(
        "--mixed-players",
        type=int,
        nargs="*",
        default=None,
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
    p.add_argument(
        "--league-rating-games",
        type=int,
        default=64,
        help="Head-to-head games used to fit new league checkpoint ratings.",
    )
    p.add_argument(
        "--league-rating-sims",
        type=int,
        default=8,
        help="Search sims for checkpoint rating matches against league checkpoints.",
    )
    p.add_argument(
        "--league-rating-matches",
        type=int,
        default=4,
        help="How many recent/strong league checkpoints to rate each new checkpoint against.",
    )
    p.add_argument("--rating-random-anchor", type=float, default=1000.0)
    p.add_argument("--rating-heuristic-anchor", type=float, default=2500.0)
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
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        eval_max_turns=args.eval_max_turns,
        rank_eval_games=args.rank_eval_games,
        rank_eval_sims=args.rank_eval_sims,
        rank_eval_max_turns=args.rank_eval_max_turns,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        max_wall_minutes=args.max_wall_minutes,
        league_selfplay_every=args.league_selfplay_every,
        league_max_entries=args.league_max_entries,
        league_keep_recent=args.league_keep_recent,
        league_rating_games=args.league_rating_games,
        league_rating_sims=args.league_rating_sims,
        league_rating_matches=args.league_rating_matches,
        rating_random_anchor=args.rating_random_anchor,
        rating_heuristic_anchor=args.rating_heuristic_anchor,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_mix=args.dirichlet_mix,
        time_discount=args.time_discount,
        q_scale=args.q_scale,
        init_from=args.init_from,
        use_amp=args.use_amp,
    )
    if args.async_eval is not None:
        loop_kwargs["async_eval"] = args.async_eval
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
