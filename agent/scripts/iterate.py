"""Outer iterate loop that calls `train` in bounded bursts and decides whether
to keep going based on metrics.

Usage:
    bazel run //experimental/mloeppky/splendor/agent/scripts:iterate -- \\
        --run-id my_run --max-bursts 20 --burst-max-iters 10 \\
        --burst-max-wall-minutes 5 --num-players 2

Between bursts, it reads `metrics.jsonl` and calls `health.decide_next_action`,
appends a journal entry, and either continues (possibly with reduced lr) or
stops.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from agent.obs import journal as J
from agent.obs.run import Run
from agent.train.health import decide_next_action
from agent.train.loop import LoopConfig, run_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Splendor iterate runner")
    p.add_argument("--run-id", required=True)
    p.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    p.add_argument(
        "--device",
        choices=["cpu"],
        default="cpu",
        help="Execution device (CPU-only; default: cpu).",
    )
    p.add_argument(
        "--no-compile-net",
        dest="compile_net",
        action="store_false",
        help="Disable torch.compile on the net (default: disabled)",
    )
    p.add_argument(
        "--compile-net",
        dest="compile_net",
        action="store_true",
        help="Enable torch.compile on the net (experimental on this machine)",
    )
    p.set_defaults(compile_net=False)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument(
        "--arch",
        choices=["attn", "flat"],
        default="flat",
        help="Network architecture for policy/value inference",
    )
    p.add_argument("--max-bursts", type=int, default=20)
    p.add_argument("--burst-max-iters", type=int, default=10)
    p.add_argument(
        "--burst-max-wall-minutes",
        type=float,
        default=30.0,
        help=(
            "Wall-clock budget per burst in minutes. Default 30min gives ~8 full "
            "iters at (selfplay_games=512, sims=16) on 16-core CPU."
        ),
    )
    p.add_argument("--selfplay-games", type=int, default=512)
    p.add_argument("--selfplay-sims", type=int, default=16)
    p.add_argument("--selfplay-max-turns", type=int, default=160)
    p.add_argument("--learner-batch", type=int, default=256)
    p.add_argument("--learner-steps-per-iter", type=int, default=192)
    p.add_argument("--entropy-bonus", type=float, default=0.0)
    p.add_argument("--eval-games", type=int, default=2048)
    p.add_argument("--eval-sims", type=int, default=64)
    p.add_argument("--eval-max-turns", type=int, default=200)
    p.add_argument("--league-selfplay-every", type=int, default=3)
    p.add_argument("--league-max-entries", type=int, default=24)
    p.add_argument("--league-keep-recent", type=int, default=8)
    p.add_argument("--rating-random-anchor", type=float, default=1000.0)
    p.add_argument("--rating-heuristic-anchor", type=float, default=2500.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--init-from",
        default="",
        help=(
            "Warm-start net weights from this checkpoint .pt on the FIRST burst "
            "only (no effect once the run has its own checkpoints). Useful to "
            "build on top of a strong baseline run without wasting iterations "
            "climbing from scratch."
        ),
    )
    return p


def _read_metrics(run: Run) -> list:
    out = []
    if not run.metrics_path.exists():
        return out
    with open(run.metrics_path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Run(args.run_id)
    run.event("iterate_start", vars(args))
    lr = args.lr

    for burst in range(1, args.max_bursts + 1):
        run.event("iterate_burst_begin", {"burst": burst, "lr": lr})
        cfg = LoopConfig(
            num_players=args.num_players,
            device=args.device,
            compile_net=args.compile_net,
            hidden=args.hidden,
            arch=args.arch,
            selfplay_games=args.selfplay_games,
            selfplay_sims=args.selfplay_sims,
            selfplay_max_turns=args.selfplay_max_turns,
            learner_batch=args.learner_batch,
            learner_steps_per_iter=args.learner_steps_per_iter,
            entropy_bonus=args.entropy_bonus,
            eval_games=args.eval_games,
            eval_sims=args.eval_sims,
            eval_max_turns=args.eval_max_turns,
            lr=lr,
            max_iters=args.burst_max_iters,
            max_wall_minutes=args.burst_max_wall_minutes,
            init_from=args.init_from,
            league_selfplay_every=args.league_selfplay_every,
            league_max_entries=args.league_max_entries,
            league_keep_recent=args.league_keep_recent,
            rating_random_anchor=args.rating_random_anchor,
            rating_heuristic_anchor=args.rating_heuristic_anchor,
        )
        result = run_loop(run, cfg)
        run.event("iterate_burst_end", {"burst": burst, "result": result})

        metrics = _read_metrics(run)
        decision = decide_next_action(metrics)
        J.append_entry(
            run.journal_path,
            f"burst {burst} decision: {decision}",
            body=f"result: {result}\nmetrics_tail: {metrics[-3:]}",
        )
        run.event("iterate_decision", {"burst": burst, "decision": decision})

        if decision == "stop_converged":
            run.event("iterate_stop_converged", {"burst": burst})
            break
        if decision == "stop_regressing":
            run.event("iterate_stop_regressing", {"burst": burst})
            break
        if decision == "reduce_lr":
            lr = lr * 0.1
            run.event("iterate_reduce_lr", {"new_lr": lr})

    run.event("iterate_done", {})
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
