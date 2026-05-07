from __future__ import annotations

import argparse
import json
import pathlib
import sys

from agent.obs.run import Run


def _load_manifest(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Show league checkpoints sorted by fitted rating")
    p.add_argument("--run-id", help="Run id under the default runs root")
    p.add_argument("--path", help="Direct path to league.json")
    args = p.parse_args(argv)

    if args.path:
        path = pathlib.Path(args.path)
    elif args.run_id:
        run = Run(args.run_id)
        path = run.ckpt_dir / "league" / "league.json"
    else:
        print("Either --run-id or --path is required", file=sys.stderr)
        return 2

    if not path.exists():
        print(f"No league manifest at {path}", file=sys.stderr)
        return 1

    manifest = _load_manifest(path)
    entries = list(manifest.get("entries", []))
    entries.sort(
        key=lambda e: (
            float(e.get("rating", 0.0)),
            float(e.get("score_hint", 0.0)),
            int(e.get("idx", -1)),
        ),
        reverse=True,
    )
    anchors = manifest.get("anchors", {})
    print(f"== {path} ({len(entries)} entries) ==")
    if anchors:
        print(
            "rating_system={}  random={}  heuristic={}".format(
                manifest.get("rating_system", "unknown"),
                anchors.get("random", "?"),
                anchors.get("heuristic", "?"),
            )
        )
    print(f"{'idx':>5} {'rating':>9} {'games':>7} {'score':>7} {'tag':>10} path")
    print("-" * 72)
    for entry in entries:
        ckpt = pathlib.Path(entry["path"]).name
        print(
            f"{int(entry['idx']):>5} "
            f"{float(entry.get('rating', 0.0)):>9.2f} "
            f"{int(entry.get('games', 0)):>7d} "
            f"{float(entry.get('score_hint', 0.0)):>7.3f} "
            f"{str(entry.get('tag', '')):>10} "
            f"{ckpt}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
