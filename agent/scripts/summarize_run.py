"""Compact summary of a training run's metrics.jsonl.

Loads the metrics file from a run directory and prints a table with:
- iteration, win-rate vs random, win-rate vs heuristic, turn / finish timing,
  loss, LR, self-play throughput.

Usage:
    bazel run //experimental/mloeppky/splendor/agent/scripts:summarize_run -- \\
        --run-id real30_v4

Or:
    bazel run //experimental/mloeppky/splendor/agent/scripts:summarize_run -- \\
        --path /abs/path/to/run/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable, List

from agent.obs.run import Run


def _rows(path: pathlib.Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _fmt(row: dict, keys: Iterable[str], width: int = 10) -> str:
    parts = []
    for k in keys:
        v = row.get(k, "")
        if isinstance(v, float):
            parts.append(f"{v:>{width}.3f}")
        elif isinstance(v, int):
            parts.append(f"{v:>{width}d}")
        else:
            parts.append(f"{str(v):>{width}}")
    return " ".join(parts)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", help="Run id under the default runs root")
    p.add_argument("--path", help="Direct path to metrics.jsonl")
    args = p.parse_args(argv)

    if args.path:
        path = pathlib.Path(args.path)
    elif args.run_id:
        run = Run(args.run_id)
        path = run.metrics_path
    else:
        print("Either --run-id or --path is required", file=sys.stderr)
        return 2

    rows = _rows(path)
    if not rows:
        print(f"No metrics rows at {path}")
        return 1

    keys = [
        "iter",
        "loss",
        "policy_loss",
        "value_loss",
        "lr",
        "winrate_vs_random",
        "winrate_vs_heuristic",
        "rank_winrate_vs_heuristic",
        "avg_turns_vs_random",
        "avg_turns_vs_heuristic",
        "avg_finished_step_vs_heuristic",
        "max_finished_step_vs_heuristic",
        "capped_vs_heuristic",
        "buffer_size",
        "elapsed_min",
    ]
    header = " ".join(f"{k:>10}" for k in keys)
    print(f"== {path} ({len(rows)} rows) ==")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(_fmt(r, keys))

    last = rows[-1]
    print("-" * len(header))
    print(
        "latest: wr_rand={wr_r:.2f}  wr_heur={wr_h:.2f}  rank_wr_heur={rank_wr_h:.2f}  "
        "loss={loss:.3f}  policy_loss={pl:.3f}  value_loss={vl:.3f}  lr={lr:.2e}".format(
            wr_r=float(last.get("winrate_vs_random", 0.0)),
            wr_h=float(last.get("winrate_vs_heuristic", 0.0)),
            rank_wr_h=float(last.get("rank_winrate_vs_heuristic", 0.0)),
            loss=float(last.get("loss", 0.0)),
            pl=float(last.get("policy_loss", 0.0)),
            vl=float(last.get("value_loss", 0.0)),
            lr=float(last.get("lr", 0.0)),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
