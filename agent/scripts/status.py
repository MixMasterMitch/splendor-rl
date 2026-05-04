"""Human-readable status of a training run.

Usage:
    bazel run //experimental/mloeppky/splendor/agent/scripts:status -- --run-id my_run
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.obs.run import Run
from agent.obs import journal as J


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize a training run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--events", type=int, default=10)
    p.add_argument("--metrics", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Run(args.run_id, create_ok=False)
    print(f"== Run: {args.run_id} ({run.root}) ==")
    hb = run.read_heartbeat()
    print(f"heartbeat: {hb}")
    st = run.read_state()
    print(f"state: {json.dumps(st, indent=2)}")
    print()
    print(f"-- recent events (last {args.events}) --")
    for line in run.read_recent_events(args.events):
        print(line.rstrip())
    print()
    print(f"-- recent metrics (last {args.metrics}) --")
    for row in run.read_recent_metrics(args.metrics):
        print(json.dumps(row))
    print()
    last = J.read_last_entry(run.journal_path)
    if last:
        print("-- last journal entry --")
        print(last)
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
