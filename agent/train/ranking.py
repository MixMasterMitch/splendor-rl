"""Bradley-Terry rating system with per-player-count pairwise records.

Result rows store wins per player count:
    {"a": "ckpt:42", "b": "random", "wins_a_2p": 10, "wins_b_2p": 6, "wins_a_3p": ...}

Rating computation:
1. Fit separate ratings per player count (anchor: random=1000).
2. Calibrate each to a common scale using 1x/2x/3x multipliers for 2p/3p/4p.
3. Combined rating = weighted average of calibrated per-PC ratings,
   weighted by actual number of games played at each player count.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Optional, Sequence

import torch

RANDOM_ANCHOR_RATING = 1000.0
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_ANCHORS = {"random": RANDOM_ANCHOR_RATING}
RATING_SCALE = 1000.0
_MIN_PROB = 1e-9

PLAYER_COUNTS = (2, 3, 4)
# Fixed calibration multipliers for scaling per-PC ratings to a common scale.
# 2p=1x (baseline), 3p=2x, 4p=3x.
CALIBRATION_SCALE = {2: 1.0, 3: 2.0, 4: 3.0}


@dataclasses.dataclass(frozen=True)
class MatchResult:
    a: str
    b: str
    wins_a: float
    wins_b: float

    @property
    def total_games(self) -> float:
        return self.wins_a + self.wins_b

    @property
    def score_a(self) -> float:
        return self.wins_a


def canonical_match(
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
) -> MatchResult:
    if a <= b:
        return MatchResult(a=a, b=b, wins_a=wins_a, wins_b=wins_b)
    return MatchResult(a=b, b=a, wins_a=wins_b, wins_b=wins_a)


def add_match_result(
    results: list[dict],
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
    ties: float = 0.0,
    num_players: int = 2,
) -> None:
    """Record a pairwise result into the results list.

    Results are stored with per-player-count fields as integers:
        wins_a_2p, wins_b_2p, ties_2p, wins_a_3p, wins_b_3p, ties_3p, ...

    For the Bradley-Terry fit, ties contribute 0.5 to each side's score.
    """
    if a == b:
        return
    wa_raw = round(wins_a)
    wb_raw = round(wins_b)
    ties_raw = round(ties)
    if wa_raw + wb_raw + ties_raw <= 0:
        return

    # Canonicalize order
    if a <= b:
        ca, cb = a, b
        wa, wb = wa_raw, wb_raw
    else:
        ca, cb = b, a
        wa, wb = wb_raw, wa_raw

    pc = num_players
    key_a = f"wins_a_{pc}p"
    key_b = f"wins_b_{pc}p"
    key_t = f"ties_{pc}p"

    for row in results:
        if row["a"] == ca and row["b"] == cb:
            row[key_a] = row.get(key_a, 0) + wa
            row[key_b] = row.get(key_b, 0) + wb
            if ties_raw > 0:
                row[key_t] = row.get(key_t, 0) + ties_raw
            return

    new_row: dict = {"a": ca, "b": cb, key_a: wa, key_b: wb}
    if ties_raw > 0:
        new_row[key_t] = ties_raw
    results.append(new_row)


def _extract_pc_results(
    results: Sequence[dict], pc: int
) -> list[MatchResult]:
    """Extract MatchResult list for a specific player count.

    Ties contribute 0.5 to each side's score for the Bradley-Terry fit.
    """
    key_a = f"wins_a_{pc}p"
    key_b = f"wins_b_{pc}p"
    key_t = f"ties_{pc}p"
    out: list[MatchResult] = []
    for row in results:
        wa = float(row.get(key_a, 0))
        wb = float(row.get(key_b, 0))
        ties = float(row.get(key_t, 0))
        effective_a = wa + 0.5 * ties
        effective_b = wb + 0.5 * ties
        if effective_a + effective_b <= 0:
            continue
        out.append(MatchResult(a=row["a"], b=row["b"], wins_a=effective_a, wins_b=effective_b))
    return out


def expected_score(rating_a: torch.Tensor, rating_b: torch.Tensor) -> torch.Tensor:
    logits = (rating_a - rating_b) * (math.log(10.0) / RATING_SCALE)
    return torch.sigmoid(logits)


def fit_ratings_for_pc(
    results: Sequence[dict],
    pc: int,
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    max_iter: int = 200,
    prior_sigma: float = 600.0,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings for a single player count.

    A Gaussian prior (L2 regularization) with std=prior_sigma pulls ratings
    toward DEFAULT_INITIAL_RATING, preventing divergence when an entity has
    a perfect record against some opponents.
    """
    anchors_map = dict(DEFAULT_ANCHORS if anchors is None else anchors)
    initial_map = {} if initial is None else dict(initial)

    matches = _extract_pc_results(results, pc)
    if not matches:
        return dict(anchors_map)

    participants: set[str] = set(anchors_map)
    for m in matches:
        participants.add(m.a)
        participants.add(m.b)

    free_ids = sorted(pid for pid in participants if pid not in anchors_map)
    ratings = dict(anchors_map)
    if not free_ids:
        return ratings

    init_values = [
        float(initial_map.get(pid, DEFAULT_INITIAL_RATING))
        for pid in free_ids
    ]
    params = torch.nn.Parameter(torch.tensor(init_values, dtype=torch.float64))
    opt = torch.optim.LBFGS(
        [params],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    index = {pid: i for i, pid in enumerate(free_ids)}
    anchor_tensors = {
        pid: torch.tensor(value, dtype=torch.float64)
        for pid, value in anchors_map.items()
    }
    prior_mean = torch.tensor(DEFAULT_INITIAL_RATING, dtype=torch.float64)
    prior_var = prior_sigma ** 2

    def _rating(pid: str) -> torch.Tensor:
        if pid in anchor_tensors:
            return anchor_tensors[pid]
        return params[index[pid]]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float64)
        for match in matches:
            prob_a = expected_score(_rating(match.a), _rating(match.b)).clamp(
                _MIN_PROB, 1.0 - _MIN_PROB
            )
            score_a = torch.tensor(match.score_a, dtype=torch.float64)
            score_b = torch.tensor(match.total_games - match.score_a, dtype=torch.float64)
            loss = loss - score_a * torch.log(prob_a) - score_b * torch.log1p(-prob_a)
        # Gaussian prior: penalize deviation from prior_mean
        loss = loss + 0.5 * ((params - prior_mean) ** 2).sum() / prior_var
        loss.backward()
        return loss

    opt.step(closure)
    out = dict(anchors_map)
    solved = params.detach().cpu().tolist()
    for pid, value in zip(free_ids, solved, strict=True):
        out[pid] = float(value)
    return out


def calibrate_rating(raw: float, pc: int) -> float:
    """Scale a raw per-PC rating to the common (2p-equivalent) scale."""
    return RANDOM_ANCHOR_RATING + (raw - RANDOM_ANCHOR_RATING) * CALIBRATION_SCALE[pc]


def _count_games_per_entity_per_pc(
    results: Sequence[dict],
) -> dict[str, dict[int, float]]:
    """Count actual games per entity per player count.

    In a K-player game, one physical game produces (K-1) pairwise result
    entries per participant.  Each result row for player count `pc`
    contributes (wins_a + wins_b + ties) pairwise entries — divide by (pc - 1)
    to get actual game count.
    """
    counts: dict[str, dict[int, float]] = {}
    for row in results:
        a, b = row["a"], row["b"]
        for pc in PLAYER_COUNTS:
            wa = float(row.get(f"wins_a_{pc}p", 0))
            wb = float(row.get(f"wins_b_{pc}p", 0))
            ties = float(row.get(f"ties_{pc}p", 0))
            pairwise = wa + wb + ties
            if pairwise <= 0:
                continue
            actual = pairwise / (pc - 1)
            counts.setdefault(a, {}).setdefault(pc, 0.0)
            counts[a][pc] += actual
            counts.setdefault(b, {}).setdefault(pc, 0.0)
            counts[b][pc] += actual
    return counts


def compute_ratings(
    results: Sequence[dict],
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
) -> dict[str, dict]:
    """Compute per-PC and combined ratings for all entities.

    The combined rating is a weighted average of calibrated per-PC ratings,
    weighted by the actual number of games played at each player count.

    Returns: {entity: {"rating_2p": ..., "rating_3p": ..., "rating_4p": ...,
                        "calibrated_2p": ..., "calibrated_3p": ..., "calibrated_4p": ...,
                        "rating": combined}}
    """
    per_pc: dict[int, dict[str, float]] = {}
    for pc in PLAYER_COUNTS:
        per_pc[pc] = fit_ratings_for_pc(results, pc, anchors=anchors, initial=initial)

    # Count actual games per entity per player count for weighting
    games_per_entity_pc = _count_games_per_entity_per_pc(results)

    # Collect all entities
    all_entities: set[str] = set()
    for pc_ratings in per_pc.values():
        all_entities.update(pc_ratings.keys())

    out: dict[str, dict] = {}
    for entity in all_entities:
        entry: dict = {}
        calibrated_sum = 0.0
        weight_sum = 0.0
        for pc in PLAYER_COUNTS:
            raw = per_pc[pc].get(entity)
            if raw is not None:
                entry[f"rating_{pc}p"] = raw
                cal = calibrate_rating(raw, pc)
                entry[f"calibrated_{pc}p"] = cal
                # Weight by actual games at this player count
                weight = games_per_entity_pc.get(entity, {}).get(pc, 0.0)
                if weight <= 0:
                    # Entity appears in ratings (e.g. anchor) but has no
                    # recorded games at this pc — use equal weight fallback.
                    weight = 1.0
                calibrated_sum += cal * weight
                weight_sum += weight
        if weight_sum > 0:
            entry["rating"] = calibrated_sum / weight_sum
        else:
            entry["rating"] = None
        out[entity] = entry
    return out


# ---------------------------------------------------------------------------
# Legacy compatibility: fit_anchored_ratings still works for code that calls it
# with the old flat results format. It treats all results as 2p.
# ---------------------------------------------------------------------------

def fit_anchored_ratings(
    results: Sequence[dict],
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    max_iter: int = 200,
) -> dict[str, float]:
    """Legacy API: fit ratings from flat results (treats as 2p or uses per-PC fields).

    If results have per-PC fields (wins_a_2p etc), fits combined ratings.
    If results have old-style fields (wins_a, wins_b, ties), treats as 2p.
    """
    # Check if results use new format
    has_new_format = any(
        any(k.startswith("wins_a_") and k.endswith("p") for k in row)
        for row in results
    )

    if has_new_format:
        # Use new per-PC system
        ratings_data = compute_ratings(results, anchors=anchors, initial=initial)
        return {entity: data["rating"] for entity, data in ratings_data.items()
                if data["rating"] is not None}

    # Legacy flat format: convert to 2p and fit
    anchors_map = dict(DEFAULT_ANCHORS if anchors is None else anchors)
    initial_map = {} if initial is None else dict(initial)

    # Convert old format rows to MatchResult
    participants: set[str] = set(anchors_map)
    clean_results: list[MatchResult] = []
    for row in results:
        a_str = str(row["a"])
        b_str = str(row["b"])
        if a_str == b_str:
            continue
        wa = float(row.get("wins_a", 0.0)) + 0.5 * float(row.get("ties", 0.0))
        wb = float(row.get("wins_b", 0.0)) + 0.5 * float(row.get("ties", 0.0))
        if wa + wb <= 0:
            continue
        if a_str <= b_str:
            clean_results.append(MatchResult(a=a_str, b=b_str, wins_a=wa, wins_b=wb))
        else:
            clean_results.append(MatchResult(a=b_str, b=a_str, wins_a=wb, wins_b=wa))
        participants.add(a_str)
        participants.add(b_str)

    free_ids = sorted(pid for pid in participants if pid not in anchors_map)
    ratings = dict(anchors_map)
    if not free_ids:
        return ratings

    init_values = [
        float(initial_map.get(pid, DEFAULT_INITIAL_RATING))
        for pid in free_ids
    ]
    params = torch.nn.Parameter(torch.tensor(init_values, dtype=torch.float64))
    opt = torch.optim.LBFGS(
        [params],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    index = {pid: i for i, pid in enumerate(free_ids)}
    anchor_tensors = {
        pid: torch.tensor(value, dtype=torch.float64)
        for pid, value in anchors_map.items()
    }

    def _rating(pid: str) -> torch.Tensor:
        if pid in anchor_tensors:
            return anchor_tensors[pid]
        return params[index[pid]]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float64)
        for match in clean_results:
            prob_a = expected_score(_rating(match.a), _rating(match.b)).clamp(
                _MIN_PROB, 1.0 - _MIN_PROB
            )
            score_a = torch.tensor(match.score_a, dtype=torch.float64)
            score_b = torch.tensor(match.total_games - match.score_a, dtype=torch.float64)
            loss = loss - score_a * torch.log(prob_a) - score_b * torch.log1p(-prob_a)
        loss.backward()
        return loss

    opt.step(closure)
    out = dict(anchors_map)
    solved = params.detach().cpu().tolist()
    for pid, value in zip(free_ids, solved, strict=True):
        out[pid] = float(value)
    return out
