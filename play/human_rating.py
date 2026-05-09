"""Persistent human game history and win tracking.

Stores the history of all games played by a human, along with game/win
counts used for placement thresholds. Rating computation is handled
entirely by play/ratings.py using the per-PC calibrated system.

Storage (JSON):

    {
        "history": [...],
        "games": <int>,
        "wins": <int>,
        "username": <str>,
    }
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import threading
from typing import Any

RANDOM_ANCHOR_RATING: float = 1000.0
DEFAULT_INITIAL_RATING: float = 1500.0

# Minimum number of wins required before the human is "placed" and their
# rating is shown on the leaderboard.
PLACEMENT_WINS_REQUIRED: int = 5


class HumanRatingStore:
    """Thread-safe JSON-backed human game history.

    Records game outcomes and persists them. Rating computation is done
    externally by play/ratings.py using the calibrated per-PC system.
    """

    def __init__(
        self,
        path: pathlib.Path,
        initial_rating: float = DEFAULT_INITIAL_RATING,
        human_entity: str = "human",
    ) -> None:
        self._path = pathlib.Path(path)
        self._human_entity = human_entity
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)
        else:
            self._data = self._fresh_data()
            self._save_locked()
        self._migrate_profile_keys_from_legacy_file()
        self._data.setdefault("history", [])
        self._data.setdefault("games", 0)
        # Always recompute wins from history to stay consistent with the
        # current definition (1st-place finishes only).
        self._data["wins"] = self._count_wins_from_history()

    def _count_wins_from_history(self) -> int:
        """Count total game wins (1st place finishes) from history."""
        total = 0
        for entry in self._data.get("history", []):
            if entry.get("legacy"):
                continue
            if int(entry.get("human_rank", -1)) == 0:
                total += 1
        return total

    def _fresh_data(self) -> dict[str, Any]:
        return {
            "history": [],
            "games": 0,
            "wins": 0,
        }

    def _migrate_profile_keys_from_legacy_file(self) -> None:
        """Copy ``google_sub`` into ``username`` from older persisted files."""
        if "username" not in self._data and "google_sub" in self._data:
            self._data["username"] = str(self._data["google_sub"])
            self._save_locked()

    def _save_locked(self) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)

    def set_profile(self, username: str) -> None:
        """Persist the screen name beside the rating table (no verification)."""
        with self._lock:
            self._data["username"] = str(username)
            self._save_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            wins = int(self._data.get("wins", 0))
            placed = wins >= PLACEMENT_WINS_REQUIRED
            out: dict[str, Any] = {
                "games": int(self._data["games"]),
                "wins": wins,
                "placed": placed,
                "history": list(self._data["history"]),
            }
            if "username" in self._data:
                out["username"] = str(self._data["username"])
            elif "google_sub" in self._data:
                out["username"] = str(self._data["google_sub"])
            return out

    def record_game(
        self,
        opponents: list[dict[str, Any]],
        human_rank: int,
        ranks: list[int],
        final_scores: list[dict[str, int]],
        human_seat: int,
        seed: int,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one finished game and persist.

        Each opponent dict requires ``seat``, ``entity_id``, ``model_id``,
        ``label`` and ``rating`` (the opponent's rating at game time).

        Returns: {"games": int, "wins": int, "per_opponent": [...]}
        """
        with self._lock:
            num_opponents = len(opponents)
            weight = 1.0 / max(num_opponents, 1)
            per_opp: list[dict[str, Any]] = []
            for opp in opponents:
                opp_seat = int(opp["seat"])
                entity_id = str(opp["entity_id"])
                opp_rating = float(opp.get("rating", DEFAULT_INITIAL_RATING))
                opp_rank = ranks[opp_seat]
                if human_rank < opp_rank:
                    score = 1.0
                else:
                    score = 0.0
                per_opp.append(
                    {
                        "seat": opp_seat,
                        "entity_id": entity_id,
                        "model_id": str(opp.get("model_id", "")),
                        "label": str(opp.get("label", "")),
                        "opp_rating": opp_rating,
                        "score": score,
                    }
                )

            self._data["games"] = int(self._data["games"]) + 1

            if human_rank == 0:
                self._data["wins"] = int(self._data.get("wins", 0)) + 1

            entry: dict[str, Any] = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "seed": int(seed),
                "human_seat": int(human_seat),
                "human_rank": int(human_rank),
                "ranks": [int(r) for r in ranks],
                "final_scores": final_scores,
                "opponents": per_opp,
            }
            if meta:
                entry["meta"] = meta
            self._data["history"].append(entry)
            self._save_locked()

            return {
                "games": self._data["games"],
                "wins": self._data["wins"],
                "per_opponent": per_opp,
            }
