"""Sync CLI: upload local game records to DynamoDB with deduplication and rating refit.

Usage:
    splendor-sync --store-root play/play_data --games-table SplendorGames --users-table SplendorUsers --region us-east-1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any

from play.dynamo_store import DynamoPlayStore
from play.human_rating import (
    HEURISTIC_ANCHOR_RATING,
    RANDOM_ANCHOR_RATING,
    fit_human_rating,
    _add_match,
    _canonical,
    HUMAN_ENTITY,
)


@dataclasses.dataclass
class SyncReport:
    """Summary of a sync operation."""

    uploaded: int
    skipped: int  # already existed (deduplicated)
    errors: int
    rating_before: float
    rating_after: float


def _read_local_games(store_root: pathlib.Path) -> list[dict[str, Any]]:
    """Read all game JSON files from the local store root."""
    games_dir = store_root / "games"
    if not games_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for p in sorted(games_dir.glob("*.json")):
        try:
            with open(p) as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _build_results_table(
    games: list[dict[str, Any]],
    human_entity: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Rebuild pairwise results table and anchors from deduplicated game records.

    Returns (results, anchors) where results is the pairwise match table
    and anchors maps opponent entity_ids to their ratings.
    """
    results: list[dict[str, Any]] = []
    anchors: dict[str, float] = {
        "random": RANDOM_ANCHOR_RATING,
        "heuristic": HEURISTIC_ANCHOR_RATING,
    }

    # Deduplicate by game_id
    seen_ids: set[str] = set()
    unique_games: list[dict[str, Any]] = []
    for game in games:
        gid = str(game.get("game_id", ""))
        if gid in seen_ids:
            continue
        seen_ids.add(gid)
        unique_games.append(game)

    for game in unique_games:
        status = game.get("status", "")
        if status != "completed":
            continue

        rating_update = game.get("rating_update") or game.get("elo_update")
        if rating_update is None:
            continue

        human_seat = int(game.get("human_seat", 0))
        num_players = int(game.get("num_players", 2))
        per_opponent = rating_update.get("per_opponent", [])

        num_opponents = len(per_opponent)
        weight = 1.0 / max(num_opponents, 1)

        for opp in per_opponent:
            entity_id = str(opp["entity_id"])
            opp_rating = float(opp.get("opp_rating", HEURISTIC_ANCHOR_RATING))
            score = float(opp.get("score", 0.5))

            # Set anchor for this opponent
            if entity_id not in ("random", "heuristic"):
                anchors[entity_id] = opp_rating

            if score == 1.0:
                wins_h, wins_o = weight, 0.0
            else:
                wins_h, wins_o = 0.0, weight

            _add_match(results, human_entity, entity_id, wins_h, wins_o, 0.0)

    return results, anchors


def sync_local_to_cloud(
    local_store_root: pathlib.Path,
    games_table_name: str,
    users_table_name: str,
    region: str,
    dry_run: bool = False,
) -> SyncReport:
    """Upload local games to DynamoDB with deduplication, then refit ratings.

    Args:
        local_store_root: Path to the local JsonPlayStore root (e.g. play/play_data).
        games_table_name: Name of the DynamoDB games table.
        users_table_name: Name of the DynamoDB users table.
        region: AWS region for DynamoDB.
        dry_run: If True, read local games and report counts without writing.

    Returns:
        SyncReport with upload/skip/error counts and rating before/after.
    """
    local_games = _read_local_games(local_store_root)

    if not local_games:
        return SyncReport(uploaded=0, skipped=0, errors=0, rating_before=0.0, rating_after=0.0)

    # Determine the user from the first game record
    user_sub = ""
    for game in local_games:
        u = game.get("user_sub", "")
        if u:
            user_sub = str(u)
            break

    if not user_sub:
        return SyncReport(uploaded=0, skipped=0, errors=0, rating_before=0.0, rating_after=0.0)

    dynamo = DynamoPlayStore(
        games_table_name=games_table_name,
        users_table_name=users_table_name,
        region=region,
    )

    # Get current rating before sync
    human_entity = f"human:{user_sub}"
    existing_blob = dynamo.load_user_rating_blob(user_sub)
    rating_before = float(existing_blob.get("rating", 0.0)) if existing_blob else 0.0

    # Upload games with deduplication
    uploaded = 0
    skipped = 0
    errors = 0

    for game in local_games:
        if dry_run:
            # In dry-run mode, just count everything as "would upload"
            uploaded += 1
            continue
        try:
            written = dynamo.put_game_if_not_exists(game)
            if written:
                uploaded += 1
            else:
                skipped += 1
        except Exception:
            errors += 1

    if dry_run:
        return SyncReport(
            uploaded=uploaded,
            skipped=skipped,
            errors=errors,
            rating_before=rating_before,
            rating_after=rating_before,
        )

    # --- Rating refit after sync (Task 6.2) ---
    # Query all games for this user from DynamoDB (includes newly uploaded + existing)
    all_cloud_games = dynamo.list_games_for_user(user_sub)

    # Rebuild pairwise results table from deduplicated game records
    results, anchors = _build_results_table(all_cloud_games, human_entity)

    # Fit the new rating
    rating_after = fit_human_rating(
        results=results,
        anchors=anchors,
        initial=rating_before if rating_before > 0 else 1500.0,
        human_entity=human_entity,
    )

    # Count total games for the blob
    total_games = sum(
        1 for g in all_cloud_games if g.get("status") == "completed"
    )
    total_wins = sum(
        1
        for g in all_cloud_games
        if g.get("status") == "completed"
        and (g.get("rating_update") or g.get("elo_update"))
        and any(
            float(opp.get("score", 0)) == 1.0
            for opp in (g.get("rating_update") or g.get("elo_update") or {}).get("per_opponent", [])
        )
    )

    # Write updated rating blob to DynamoDB users table
    rating_blob: dict[str, Any] = {
        "rating_system": "anchored_bt",
        "anchors": anchors,
        "results": results,
        "history": [],  # History is not reconstructed from game records
        "rating": rating_after,
        "games": total_games,
        "wins": total_wins,
        "username": user_sub,
    }

    # Preserve history from existing blob if available
    if existing_blob and "history" in existing_blob:
        rating_blob["history"] = existing_blob["history"]

    dynamo.save_user_rating_blob(user_sub, rating_blob)

    return SyncReport(
        uploaded=uploaded,
        skipped=skipped,
        errors=errors,
        rating_before=rating_before,
        rating_after=rating_after,
    )


def main() -> None:
    """CLI entry point for splendor-sync."""
    parser = argparse.ArgumentParser(
        prog="splendor-sync",
        description="Sync local game records to DynamoDB with deduplication and rating refit.",
    )
    parser.add_argument(
        "--store-root",
        type=pathlib.Path,
        default=pathlib.Path("play/play_data"),
        help="Path to local JsonPlayStore root directory (default: play/play_data)",
    )
    parser.add_argument(
        "--games-table",
        type=str,
        required=True,
        help="Name of the DynamoDB games table",
    )
    parser.add_argument(
        "--users-table",
        type=str,
        required=True,
        help="Name of the DynamoDB users table",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read local games and report counts without writing to DynamoDB",
    )

    args = parser.parse_args()

    report = sync_local_to_cloud(
        local_store_root=args.store_root,
        games_table_name=args.games_table,
        users_table_name=args.users_table,
        region=args.region,
        dry_run=args.dry_run,
    )

    # Print sync report summary
    print("=" * 50)
    print("Sync Report")
    print("=" * 50)
    print(f"  Uploaded:       {report.uploaded}")
    print(f"  Skipped (dup):  {report.skipped}")
    print(f"  Errors:         {report.errors}")
    print(f"  Rating before:  {report.rating_before:.1f}")
    print(f"  Rating after:   {report.rating_after:.1f}")
    if report.rating_after != report.rating_before:
        delta = report.rating_after - report.rating_before
        print(f"  Rating delta:   {delta:+.1f}")
    print("=" * 50)

    if report.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
