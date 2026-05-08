"""Sync all local game records to DynamoDB and refit ratings for all users.

Unlike play.sync (which refits only the first user found), this module:
1. Uploads ALL local game records with conditional-write deduplication
2. Refits ratings for EVERY user who has completed games in DynamoDB

Usage:
    python -m play.sync_all --store-root play/play_data \
        --games-table TABLE --users-table TABLE --region us-west-2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from play.dynamo_store import DynamoPlayStore
from play.human_rating import (
    DEFAULT_INITIAL_RATING,
    RANDOM_ANCHOR_RATING,
    fit_human_rating,
)
from play.sync import _build_results_table, _read_local_games


def _build_history_from_games(completed_games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct history entries from completed game records.

    This is needed because HumanRatingStore._count_wins_from_history()
    recomputes wins from the history array on every load. Without proper
    history entries, the wins count reverts to whatever is in history.
    """
    history: list[dict[str, Any]] = []
    # Sort by created_at/updated_at for chronological order
    sorted_games = sorted(
        completed_games,
        key=lambda g: g.get("updated_at") or g.get("created_at") or "",
    )

    for game in sorted_games:
        eu = game.get("rating_update") or game.get("elo_update")
        if not eu:
            continue

        per_opponent = eu.get("per_opponent", [])
        if not per_opponent:
            continue

        # Determine human_rank: 0 = won (all opponents have score=1.0),
        # else rank = number of opponents who beat the human + 1... but
        # for simplicity in 2-player: score=1.0 means human won (rank 0),
        # score=0.0 means human lost (rank 1).
        all_won = all(float(opp.get("score", 0)) == 1.0 for opp in per_opponent)
        human_rank = 0 if all_won else 1

        entry: dict[str, Any] = {
            "timestamp": game.get("updated_at") or game.get("created_at") or "",
            "seed": int(game.get("seed", 0)),
            "human_seat": int(game.get("human_seat", 0)),
            "human_rank": human_rank,
            "opponents": per_opponent,
            "meta": {"game_id": str(game.get("game_id", ""))},
        }

        # Include rating deltas if available
        if "old_rating" in eu:
            entry["old_rating"] = eu["old_rating"]
            entry["new_rating"] = eu["new_rating"]
            entry["delta"] = eu["delta"]

        history.append(entry)

    return history


def sync_all(
    local_store_root: pathlib.Path,
    games_table_name: str,
    users_table_name: str,
    region: str,
    dry_run: bool = False,
) -> None:
    """Upload all local games and refit ratings for every user."""
    local_games = _read_local_games(local_store_root)
    print(f"Found {len(local_games)} local game records.")

    dynamo = DynamoPlayStore(
        games_table_name=games_table_name,
        users_table_name=users_table_name,
        region=region,
    )

    # --- Upload all games ---
    uploaded = 0
    skipped = 0
    errors = 0

    for game in local_games:
        if dry_run:
            uploaded += 1
            continue
        try:
            written = dynamo.put_game_if_not_exists(game)
            if written:
                uploaded += 1
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"  ERROR uploading {game.get('game_id', '?')}: {e}", file=sys.stderr)

    print(f"  Uploaded: {uploaded}, Skipped (dup): {skipped}, Errors: {errors}")

    if dry_run:
        print("  (dry-run mode — no writes performed)")
        return

    # --- Refit ratings for all users ---
    print("\nRefitting ratings for all users...")

    # Discover all users from the games table
    import boto3
    ddb = boto3.resource("dynamodb", region_name=region)
    games_table = ddb.Table(games_table_name)

    # Scan for unique user_sub values
    users: set[str] = set()
    scan_kwargs: dict[str, Any] = {"ProjectionExpression": "user_sub"}
    while True:
        resp = games_table.scan(**scan_kwargs)
        for item in resp.get("Items", []):
            u = item.get("user_sub", "")
            if u:
                users.add(str(u))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    print(f"  Found {len(users)} users with games: {sorted(users)}")

    for user in sorted(users):
        games = dynamo.list_games_for_user(user)
        completed = [g for g in games if g.get("status") == "completed"]

        if not completed:
            continue

        human_entity = f"human:{user}"
        results, anchors = _build_results_table(games, human_entity)

        existing_blob = dynamo.load_user_rating_blob(user)
        rating_before = float(existing_blob.get("rating", 0.0)) if existing_blob else 0.0

        rating_after = fit_human_rating(
            results=results,
            anchors=anchors,
            initial=rating_before if rating_before > 0 else DEFAULT_INITIAL_RATING,
            human_entity=human_entity,
        )

        total_games = len(completed)
        total_wins = sum(
            1
            for g in completed
            if g.get("rating_update") or g.get("elo_update")
            and any(
                float(opp.get("score", 0)) == 1.0
                for opp in (g.get("rating_update") or g.get("elo_update") or {}).get("per_opponent", [])
            )
        )

        # Build history entries from game records so HumanRatingStore's
        # _count_wins_from_history() returns the correct count.
        history = _build_history_from_games(completed)

        rating_blob: dict[str, Any] = {
            "rating_system": "anchored_bt",
            "anchors": anchors,
            "results": results,
            "history": history,
            "rating": rating_after,
            "games": total_games,
            "wins": total_wins,
            "username": user,
        }

        dynamo.save_user_rating_blob(user, rating_blob)
        print(f"  {user}: {rating_before:.0f} -> {rating_after:.0f} ({total_wins}W/{total_games}G)")

    print("\nAll ratings refitted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sync_all",
        description="Sync all local games to DynamoDB and refit all user ratings.",
    )
    parser.add_argument(
        "--store-root",
        type=pathlib.Path,
        default=pathlib.Path("play/play_data"),
        help="Path to local JsonPlayStore root (default: play/play_data)",
    )
    parser.add_argument("--games-table", type=str, required=True)
    parser.add_argument("--users-table", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    sync_all(
        local_store_root=args.store_root,
        games_table_name=args.games_table,
        users_table_name=args.users_table,
        region=args.region,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
