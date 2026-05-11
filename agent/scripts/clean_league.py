"""Remove inactive ML agents from the league and recompute ratings.

Opens league.json, removes all entries with "active": false, prunes their
pairwise records from the results table, deletes their checkpoint files from
disk, and recomputes ratings from the remaining data.

A backup is saved to league.json.bak before any modifications.

Usage:
    python -m agent.scripts.clean_league [--league-root agent/runs/league] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove inactive ML agents from the league and recompute ratings."
    )
    parser.add_argument(
        "--league-root",
        type=str,
        default="agent/runs/league",
        help="Path to the league directory (default: agent/runs/league)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without modifying anything.",
    )
    args = parser.parse_args()

    league_root = pathlib.Path(args.league_root)
    manifest_path = league_root / "league.json"

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    results = manifest.get("results", [])

    # Identify active vs inactive entries
    active_entries = [e for e in entries if e.get("active", False)]
    inactive_entries = [e for e in entries if not e.get("active", False)]

    if not inactive_entries:
        print("No inactive entries to remove. League is already clean.")
        return

    # Build set of entity IDs to remove
    inactive_ids = {f"ckpt:{e['idx']}" for e in inactive_entries}
    # Entity IDs that survive (active entries + anchors + floating entities)
    keep_ids = {f"ckpt:{e['idx']}" for e in active_entries}
    keep_ids.update(manifest.get("anchors", {}).keys())
    # Floating entities (heuristic_opus, bedrock_claude_sonnet, etc.) participate
    # in results and should be preserved.
    keep_ids.update(manifest.get("floating_entities", {}).keys())
    # Also keep any non-checkpoint entities that appear in results (random, heuristic, etc.)
    for row in results:
        for side in ("a", "b"):
            entity = row.get(side, "")
            if not entity.startswith("ckpt:"):
                keep_ids.add(entity)

    # Filter results: keep only rows where both sides are in keep_ids
    clean_results = [
        row for row in results
        if row.get("a") in keep_ids and row.get("b") in keep_ids
    ]

    # Checkpoint files to delete
    files_to_delete = []
    for entry in inactive_entries:
        ckpt_path = league_root / entry.get("path", "")
        if ckpt_path.exists():
            files_to_delete.append(ckpt_path)

    # Report
    print(f"League: {manifest_path}")
    print(f"  Total entries:    {len(entries)}")
    print(f"  Active entries:   {len(active_entries)}")
    print(f"  Inactive entries: {len(inactive_entries)} (to remove)")
    print(f"  Results before:   {len(results)}")
    print(f"  Results after:    {len(clean_results)}")
    print(f"  Checkpoint files to delete: {len(files_to_delete)}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made. Inactive entries that would be removed:")
        for e in inactive_entries:
            print(f"  ckpt:{e['idx']} ({e.get('tag', '?')}) rating={e.get('rating', '?')}")
        return

    # Backup
    backup_path = manifest_path.with_suffix(".json.bak")
    shutil.copy2(manifest_path, backup_path)
    print(f"\nBackup saved to: {backup_path}")

    # Delete checkpoint files
    deleted = 0
    for ckpt_path in files_to_delete:
        ckpt_path.unlink()
        deleted += 1
    if deleted:
        print(f"Deleted {deleted} checkpoint files from disk.")

    # Update manifest
    manifest["entries"] = active_entries
    manifest["results"] = clean_results

    # Save cleaned manifest (before recomputing so we can use the League class)
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(manifest_path)
    print("Saved cleaned manifest.")

    # Recompute ratings using the League class
    from ..train.league import League

    league = League(league_root)
    ratings = league.recompute_ratings()
    print(f"\nRecomputed ratings for {len(ratings)} entities.")

    # Print final leaderboard
    entry_ratings = [
        (e.get("tag", "?"), e.get("idx"), e.get("rating", 0))
        for e in league.manifest["entries"]
    ]
    entry_ratings.sort(key=lambda x: -x[2])
    print("\nFinal league standings:")
    for tag, idx, rating in entry_ratings:
        print(f"  ckpt:{idx} ({tag}): {rating}")


if __name__ == "__main__":
    main()
