"""Persistent human rating using the batch-fit Bradley-Terry/Elo system.

Mirrors the league rating system in
agent/train/ranking. We maintain an
aggregated head-to-head match table for the human against every opponent we
have ever played, and after each finished game we refit the human's rating
from the full table by maximum likelihood, anchored to fixed ratings:

    random    -> 1000
    heuristic -> 2500
    each net  -> the rating reported in that net's source league.json

Because every refit uses the full result history, the rating is independent
of the order games were played in. There is no per-game k-factor update.

Storage (JSON) at play/human_elo.json:

    {
        "rating_system": "anchored_bt",
        "anchors": {"random": 1000.0, "heuristic": 2500.0},
        "results": [
            {"a": "human", "b": "<entity_id>",
             "wins_a": .., "wins_b": .., "ties": .., "games": ..},
            ...
        ],
        "history": [...],
        "rating": <float>,
        "games": <int>
    }

Multi-player games: we decompose into pairwise human-vs-opponent records
based on final ranking (1 = win, 0 = loss). Ties are not possible because
Splendor uses a fewer-cards tiebreaker when points are equal.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import pathlib
import threading
from typing import Any, Iterable

RANDOM_ANCHOR_RATING: float = 1000.0
HEURISTIC_ANCHOR_RATING: float = 2500.0
ELO_SCALE: float = 400.0
HUMAN_ENTITY: str = "human"
DEFAULT_INITIAL_RATING: float = 1500.0

# Bayesian regularization: imagine the human starts with this many "ghost"
# games at 50 percent against a virtual reference opponent rated at
# ``PRIOR_MEAN_RATING``. With Bradley-Terry these ghost games give a proper
# prior over the human's rating (much stronger than a Gaussian prior on the
# rating, because Gaussians don't scale with the logistic likelihood). With
# 6 ghost games, a single real win against heuristic (2500) settles the
# rating around the mid-1700s instead of running off to +infinity.
PRIOR_MEAN_RATING: float = 1500.0
PRIOR_GHOST_GAMES: float = 0.0

# Minimum number of wins required before the human is "placed" and their
# rating is shown on the leaderboard.
PLACEMENT_WINS_REQUIRED: int = 5


def _expected_score(r_a: float, r_b: float) -> float:
    """Probability that A beats B under Elo with scale 400."""
    return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / ELO_SCALE))


def _canonical(a: str, b: str, wins_a: float, wins_b: float, ties: float):
    if a <= b:
        return a, b, float(wins_a), float(wins_b), float(ties)
    return b, a, float(wins_b), float(wins_a), float(ties)


def _add_match(
    results: list[dict[str, Any]],
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
    ties: float,
) -> None:
    a, b, wa, wb, t = _canonical(a, b, wins_a, wins_b, ties)
    if wa + wb + t <= 0:
        return
    for row in results:
        if row["a"] == a and row["b"] == b:
            row["wins_a"] = float(row.get("wins_a", 0.0)) + wa
            row["wins_b"] = float(row.get("wins_b", 0.0)) + wb
            row["ties"] = float(row.get("ties", 0.0)) + t
            row["games"] = float(row.get("games", 0.0)) + (wa + wb + t)
            return
    results.append(
        {
            "a": a,
            "b": b,
            "wins_a": wa,
            "wins_b": wb,
            "ties": t,
            "games": wa + wb + t,
        }
    )


def fit_human_rating(
    results: Iterable[dict[str, Any]],
    anchors: dict[str, float],
    initial: float = DEFAULT_INITIAL_RATING,
    prior_mean: float = PRIOR_MEAN_RATING,
    prior_ghost_games: float = PRIOR_GHOST_GAMES,
    human_entity: str = HUMAN_ENTITY,
) -> float:
    """Maximum a posteriori rating for the single free human entity id.

    All other entities referenced in ``results`` must have an anchor in
    ``anchors``. We add ``prior_ghost_games`` virtual 50 percent results
    against a reference opponent at ``prior_mean``; this is a Bradley-Terry-
    consistent prior that prevents the rating from blowing up to +/- infinity
    when the real game record is one-sided. ``prior_ghost_games`` of about 4
    to 8 keeps the rating sensible after the first few games.

    Uses bisection on the (monotone-decreasing) MAP score-residual gradient.
    Returns ``initial`` when the human has no recorded matches.
    """
    matches: list[tuple[float, float, float, float]] = []
    for row in results:
        a = str(row["a"])
        b = str(row["b"])
        wa = float(row.get("wins_a", 0.0))
        wb = float(row.get("wins_b", 0.0))
        ties = float(row.get("ties", 0.0))
        n = wa + wb + ties
        if n <= 0:
            continue
        if a == human_entity:
            opp = b
            human_score = wa + 0.5 * ties
        elif b == human_entity:
            opp = a
            human_score = wb + 0.5 * ties
        else:
            continue
        if opp not in anchors:
            continue
        matches.append((float(anchors[opp]), human_score, n - human_score, n))

    if not matches:
        return float(initial)

    ghost_n = max(0.0, float(prior_ghost_games))
    ghost_score = 0.5 * ghost_n
    ghost_opp = float(prior_mean)

    def gradient(r: float) -> float:
        g = 0.0
        for opp_r, human_score, _, n in matches:
            p = _expected_score(r, opp_r)
            g += human_score - n * p
        if ghost_n > 0:
            g += ghost_score - ghost_n * _expected_score(r, ghost_opp)
        return g

    lo, hi = -5000.0, 8000.0
    g_lo = gradient(lo)
    g_hi = gradient(hi)
    if g_lo <= 0:
        return lo
    if g_hi >= 0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gradient(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class HumanRatingStore:
    """Thread-safe JSON-backed human rating record.

    Anchors track the union of all opponents we have ever played against,
    using the rating that opponent had at game time. For built-in random and
    heuristic bots we use the canonical anchors.
    """

    def __init__(
        self,
        path: pathlib.Path,
        initial_rating: float = DEFAULT_INITIAL_RATING,
        human_entity: str = HUMAN_ENTITY,
    ) -> None:
        self._path = pathlib.Path(path)
        self._initial_rating = float(initial_rating)
        self._human_entity = human_entity
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)
            self._migrate_legacy_in_place()
        else:
            self._data = self._fresh_data()
            self._save_locked()
        self._data.setdefault("rating_system", "anchored_bt")
        self._migrate_profile_keys_from_legacy_file()
        self._data.setdefault(
            "anchors",
            {
                "random": RANDOM_ANCHOR_RATING,
                "heuristic": HEURISTIC_ANCHOR_RATING,
            },
        )
        self._data.setdefault("results", [])
        self._data.setdefault("history", [])
        self._data.setdefault("rating", self._initial_rating)
        self._data.setdefault("games", 0)
        # Always recompute wins from history to stay consistent with the
        # current definition (1st-place finishes only).
        self._data["wins"] = self._count_wins_from_history()

    def _count_wins_from_history(self) -> int:
        """Count total game wins (1st place finishes) from history.

        Always recomputed on load to stay consistent with the current
        definition regardless of what was previously persisted.
        """
        total = 0
        for entry in self._data.get("history", []):
            if entry.get("legacy"):
                continue
            if int(entry.get("human_rank", -1)) == 0:
                total += 1
        return total

    def _fresh_data(self) -> dict[str, Any]:
        return {
            "rating_system": "anchored_bt",
            "anchors": {
                "random": RANDOM_ANCHOR_RATING,
                "heuristic": HEURISTIC_ANCHOR_RATING,
            },
            "results": [],
            "history": [],
            "rating": self._initial_rating,
            "games": 0,
            "wins": 0,
        }

    def _migrate_profile_keys_from_legacy_file(self) -> None:
        """Copy ``google_sub`` into ``username`` from older persisted files."""
        if "username" not in self._data and "google_sub" in self._data:
            self._data["username"] = str(self._data["google_sub"])
            self._save_locked()

    def _migrate_legacy_in_place(self) -> None:
        """Older files used per-game online Elo updates (`elo`, `history`).

        We keep the existing history but reset to a clean rating because the
        results table is unrecoverable. New games will start filling in the
        new rating system.
        """
        if "results" in self._data:
            return
        legacy = self._data
        self._data = self._fresh_data()
        for entry in legacy.get("history", []):
            self._data["history"].append({**entry, "legacy": True})
        self._data["games"] = int(legacy.get("games", 0))
        self._save_locked()

    def _save_locked(self) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)

    def _set_anchor_locked(self, entity_id: str, rating: float) -> None:
        # Canonical anchors are immutable; for nets we keep the latest rating
        # observed at game time (a freshly-trained checkpoint may have a new
        # rating relative to last time we played it).
        if entity_id in ("random", "heuristic"):
            return
        self._data["anchors"][entity_id] = float(rating)

    def set_profile(self, username: str) -> None:
        """Persist the screen name beside the rating table (no verification)."""
        with self._lock:
            self._data["username"] = str(username)
            self._save_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            wins = int(self._data.get("wins", 0))
            placed = wins >= PLACEMENT_WINS_REQUIRED
            rating = float(self._data["rating"])
            out: dict[str, Any] = {
                "rating_system": str(self._data.get("rating_system", "anchored_bt")),
                "rating": rating if placed else None,
                "games": int(self._data["games"]),
                "wins": wins,
                "placed": placed,
                "anchors": dict(self._data["anchors"]),
                "history": list(self._data["history"]),
                "results": list(self._data["results"]),
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
        """Record one finished game, refit the human's rating, and persist.

        Each opponent dict requires ``seat``, ``entity_id``, ``model_id``,
        ``label`` and ``rating`` (the opponent's rating at game time).
        """
        with self._lock:
            old_rating = float(self._data["rating"])
            num_opponents = len(opponents)
            # Normalize pairwise results by number of opponents so that a
            # single 4-player game contributes the same total weight as a
            # single 2-player game (each pairwise result is scaled by
            # 1/num_opponents).
            weight = 1.0 / max(num_opponents, 1)
            per_opp: list[dict[str, Any]] = []
            for opp in opponents:
                opp_seat = int(opp["seat"])
                entity_id = str(opp["entity_id"])
                opp_rating = float(opp.get("rating", HEURISTIC_ANCHOR_RATING))
                self._set_anchor_locked(entity_id, opp_rating)
                opp_rank = ranks[opp_seat]
                if human_rank < opp_rank:
                    score = 1.0
                    wins_h, wins_o = weight, 0.0
                else:
                    score = 0.0
                    wins_h, wins_o = 0.0, weight
                _add_match(
                    self._data["results"],
                    self._human_entity,
                    entity_id,
                    wins_h,
                    wins_o,
                    0.0,
                )
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

            new_rating = fit_human_rating(
                self._data["results"],
                self._data["anchors"],
                initial=old_rating,
                human_entity=self._human_entity,
            )
            self._data["rating"] = new_rating
            self._data["games"] = int(self._data["games"]) + 1

            # Count wins for placement threshold.
            # A win = human finished 1st (rank 0, beat all opponents).
            # A tie does NOT count toward the 5-win placement threshold.
            if human_rank == 0:
                self._data["wins"] = int(self._data.get("wins", 0)) + 1

            entry = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "seed": int(seed),
                "human_seat": int(human_seat),
                "human_rank": int(human_rank),
                "ranks": [int(r) for r in ranks],
                "final_scores": final_scores,
                "old_rating": old_rating,
                "new_rating": new_rating,
                "delta": new_rating - old_rating,
                "opponents": per_opp,
            }
            if meta:
                entry["meta"] = meta
            self._data["history"].append(entry)
            self._save_locked()

            return {
                "old_rating": old_rating,
                "new_rating": new_rating,
                "delta": new_rating - old_rating,
                "games": self._data["games"],
                "per_opponent": per_opp,
            }


HumanEloStore = HumanRatingStore
