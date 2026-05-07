from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Sequence

import torch

RANDOM_ANCHOR_RATING = 1000.0
HEURISTIC_ANCHOR_RATING = 2500.0
DEFAULT_ANCHORS = {
    "random": RANDOM_ANCHOR_RATING,
    "heuristic": HEURISTIC_ANCHOR_RATING,
}
RATING_SCALE = 1000.0
_MIN_PROB = 1e-9


@dataclasses.dataclass(frozen=True)
class MatchResult:
    a: str
    b: str
    wins_a: float
    wins_b: float
    ties: float

    @property
    def total_games(self) -> float:
        return self.wins_a + self.wins_b + self.ties

    @property
    def score_a(self) -> float:
        return self.wins_a + 0.5 * self.ties


def canonical_match(
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
    ties: float,
) -> MatchResult:
    if a <= b:
        return MatchResult(a=a, b=b, wins_a=wins_a, wins_b=wins_b, ties=ties)
    return MatchResult(a=b, b=a, wins_a=wins_b, wins_b=wins_a, ties=ties)


def add_match_result(
    results: list[dict],
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
    ties: float,
) -> None:
    # Drop self-play results — they carry no rating information and can
    # arise in multiplayer games where the same entity occupies multiple seats.
    if a == b:
        return
    match = canonical_match(a, b, wins_a, wins_b, ties)
    if match.total_games <= 0:
        return
    for row in results:
        if row["a"] == match.a and row["b"] == match.b:
            row["wins_a"] = float(row.get("wins_a", 0.0)) + match.wins_a
            row["wins_b"] = float(row.get("wins_b", 0.0)) + match.wins_b
            row["ties"] = float(row.get("ties", 0.0)) + match.ties
            row["games"] = float(row.get("games", 0.0)) + match.total_games
            return
    results.append(
        {
            "a": match.a,
            "b": match.b,
            "wins_a": match.wins_a,
            "wins_b": match.wins_b,
            "ties": match.ties,
            "games": match.total_games,
        }
    )


def expected_score(rating_a: torch.Tensor, rating_b: torch.Tensor) -> torch.Tensor:
    logits = (rating_a - rating_b) * (math.log(10.0) / RATING_SCALE)
    return torch.sigmoid(logits)


def fit_anchored_ratings(
    results: Sequence[dict],
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    max_iter: int = 200,
) -> dict[str, float]:
    anchors_map = dict(DEFAULT_ANCHORS if anchors is None else anchors)
    initial_map = {} if initial is None else dict(initial)

    participants: set[str] = set(anchors_map)
    clean_results: list[MatchResult] = []
    for row in results:
        a_str = str(row["a"])
        b_str = str(row["b"])
        # Skip self-play rows (same entity on both sides)
        if a_str == b_str:
            continue
        match = canonical_match(
            a_str,
            b_str,
            float(row.get("wins_a", 0.0)),
            float(row.get("wins_b", 0.0)),
            float(row.get("ties", 0.0)),
        )
        if match.total_games <= 0:
            continue
        participants.add(match.a)
        participants.add(match.b)
        clean_results.append(match)

    free_ids = sorted(pid for pid in participants if pid not in anchors_map)
    ratings = dict(anchors_map)
    if not free_ids:
        return ratings

    anchor_mean = sum(anchors_map.values()) / max(len(anchors_map), 1)
    init_values = [
        float(initial_map.get(pid, anchor_mean))
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
