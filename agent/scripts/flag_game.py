"""Flag a game from DynamoDB for inclusion in training replays.

Usage:
    python3 -m agent.scripts.flag_game --game-id d798515a566a

Pulls the game record from DynamoDB and saves it to agent/training_replays/.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Flag a game for training replay injection")
    p.add_argument("--game-id", required=True, help="Game ID to pull from DynamoDB")
    p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-west-2"),
        help="AWS region (default: us-west-2)",
    )
    p.add_argument(
        "--table",
        default=os.environ.get(
            "GAMES_TABLE", "SplendorStack-GamesTableB32AB610-1C9JML169FTJT"
        ),
        help="DynamoDB games table name",
    )
    p.add_argument(
        "--output-dir",
        default="agent/training_replays",
        help="Directory to save replay files (default: agent/training_replays)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import subprocess

    # Use AWS CLI (more reliable than boto3 which needs CRT dependency)
    result = subprocess.run(
        [
            "aws", "dynamodb", "get-item",
            "--table-name", args.table,
            "--key", json.dumps({"game_id": {"S": args.game_id}}),
            "--region", args.region,
            "--output", "json",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"AWS CLI error: {result.stderr}")
        return 1

    raw = json.loads(result.stdout)
    item = raw.get("Item")
    if not item:
        print(f"Game {args.game_id} not found in {args.table}")
        return 1

    # The game data is stored as a JSON string in the 'data' field
    data_str = item.get("data", {}).get("S", "{}")
    game_data = json.loads(data_str)

    # Save to output directory
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.game_id}.json"

    with open(out_path, "w") as f:
        json.dump(game_data, f, indent=2, default=str)

    steps = game_data.get("steps", [])
    print(f"Saved game {args.game_id} ({len(steps)} steps) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
