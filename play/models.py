"""Discover available opponent models for the play server.

Sources:
1. Built-ins: 'random' (anchored at 1000) and 'heuristic' (anchored at 2500).
   These match the anchors used by the league rating system in
   agent/train/ranking.
2. Every entry in every league.json under
       agent/runs/<run>/checkpoints/league/league.json
   Each entry carries a rating fit from the league's aggregated head-to-head
   match results (anchored to random/heuristic). We always read the latest
   league.json on demand so trained ratings stay in sync as training proceeds.

Each returned model dict has the shape:
    {
        "id": str,           # stable identifier used to start games
        "label": str,        # human-friendly label
        "kind": str,         # 'random' | 'heuristic' | 'net'
        "run": str | None,
        "tag": str | None,
        "ckpt": str | None,
        "rating": float,     # batch-fit rating (or anchor for built-ins)
        "games": int,
        "hidden": int | None,
        "arch": str | None,
        "score_hint": float | None,
        "winrate_vs_heuristic": float | None,
    }

The "elo" field is kept as an alias for "rating" for the frontend.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Optional

RANDOM_ANCHOR_RATING: float = 1000.0
HEURISTIC_ANCHOR_RATING: float = 2500.0
HEURISTIC_OPUS_RATING: float = 2645.0


def _runs_root(workspace_root: pathlib.Path) -> pathlib.Path:
    return workspace_root / "agent/runs"


def _entry_rating(entry: dict, fallback: float) -> float:
    val = entry.get("rating")
    if val is None:
        val = entry.get("elo")
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


def _read_json_resilient(path: pathlib.Path) -> dict[str, Any] | None:
    """Read a league JSON, retrying briefly if a concurrent training process
    is mid-update.

    Training writes via ``write tmp -> os.replace`` which is atomic on POSIX,
    so we should always observe either the old or the new whole file. The
    retries cover transient FS hiccups (e.g. the file being absent for a
    nanosecond between unlink and create on filesystems that don't support
    atomic replace).
    """
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
        except OSError as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
    return None


def _scan_league_json(path: pathlib.Path, run_id: str) -> list[dict[str, Any]]:
    data = _read_json_resilient(path)
    if data is None:
        return []
    out: list[dict[str, Any]] = []
    for entry in data.get("entries", []):
        ckpt_path_str = str(entry.get("path", ""))
        if not ckpt_path_str:
            continue
        ckpt_path = pathlib.Path(ckpt_path_str)
        if not ckpt_path.exists():
            continue
        tag = str(entry.get("tag", f"idx{entry.get('idx', '?')}"))
        idx = int(entry.get("idx", -1))
        model_id = f"net:{run_id}:{idx}"
        rating = _entry_rating(entry, HEURISTIC_ANCHOR_RATING)
        out.append(
            {
                "id": model_id,
                "label": f"{run_id}/{tag} (rating {rating:.0f})",
                "kind": "net",
                "run": run_id,
                "tag": tag,
                "idx": idx,
                "ckpt": str(ckpt_path),
                "rating": rating,
                "elo": rating,
                "games": int(entry.get("games", 0)),
                "hidden": entry.get("hidden"),
                "arch": entry.get("arch"),
                "score_hint": entry.get("score_hint"),
                "winrate_vs_heuristic": entry.get("winrate_vs_heuristic"),
            }
        )
    return out


def _builtins() -> list[dict[str, Any]]:
    return [
        {
            "id": "heuristic_opus",
            "label": f"Heuristic-opus V15 (rating {HEURISTIC_OPUS_RATING:.0f})",
            "kind": "heuristic_opus",
            "run": None,
            "tag": None,
            "ckpt": None,
            "rating": HEURISTIC_OPUS_RATING,
            "elo": HEURISTIC_OPUS_RATING,
            "games": 0,
            "hidden": None,
            "arch": None,
            "score_hint": None,
            "winrate_vs_heuristic": None,
        },
        {
            "id": "heuristic",
            "label": f"Heuristic bot (anchor {HEURISTIC_ANCHOR_RATING:.0f})",
            "kind": "heuristic",
            "run": None,
            "tag": None,
            "ckpt": None,
            "rating": HEURISTIC_ANCHOR_RATING,
            "elo": HEURISTIC_ANCHOR_RATING,
            "games": 0,
            "hidden": None,
            "arch": None,
            "score_hint": None,
            "winrate_vs_heuristic": None,
        },
        {
            "id": "random",
            "label": f"Random bot (anchor {RANDOM_ANCHOR_RATING:.0f})",
            "kind": "random",
            "run": None,
            "tag": None,
            "ckpt": None,
            "rating": RANDOM_ANCHOR_RATING,
            "elo": RANDOM_ANCHOR_RATING,
            "games": 0,
            "hidden": None,
            "arch": None,
            "score_hint": None,
            "winrate_vs_heuristic": None,
        },
    ]


def discover_models(workspace_root: pathlib.Path) -> list[dict[str, Any]]:
    """Return all available opponent models, including built-ins."""
    models: list[dict[str, Any]] = _builtins()

    runs_root = _runs_root(workspace_root)
    if not runs_root.exists():
        return models

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        league_json = run_dir / "checkpoints" / "league" / "league.json"
        if league_json.exists():
            models.extend(_scan_league_json(league_json, run_dir.name))

    return models


def model_by_id(
    models: list[dict[str, Any]],
    model_id: str,
) -> Optional[dict[str, Any]]:
    for m in models:
        if m["id"] == model_id:
            return m
    return None


def model_entity_id(model: dict[str, Any]) -> str:
    """Stable entity id for the rating system.

    Anchored builtins use their kind name; net checkpoints use the
    full model id (which already encodes run + idx).
    """
    if model["kind"] in ("random", "heuristic", "heuristic_opus"):
        return str(model["kind"])
    return str(model["id"])
