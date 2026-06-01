from __future__ import annotations

import pytest

from play import ratings as RT


def _opus_opponents() -> list[dict[str, object]]:
    return [
        {"seat": 1, "entity_id": "heuristic_opus"},
        {"seat": 2, "entity_id": "heuristic_opus"},
        {"seat": 3, "entity_id": "heuristic_opus"},
    ]


def test_4p_table_win_rate_expands_to_pairwise_bt_wins() -> None:
    blob = {"history": []}
    for _ in range(4):
        blob["history"].append(
            {
                "human_rank": 0,
                "ranks": [0, 1, 2, 3],
                "opponents": _opus_opponents(),
            }
        )
    for _ in range(6):
        blob["history"].append(
            {
                "human_rank": 1,
                "ranks": [1, 0, 2, 3],
                "opponents": _opus_opponents(),
            }
        )

    results = RT._build_human_per_pc_results(blob)

    assert results[4] == [("heuristic_opus", 12.0, 6.0)]


def test_4p_human_fit_interprets_40_percent_vs_three_opus_as_stronger_than_opus() -> None:
    blob = {"history": []}
    for _ in range(400):
        blob["history"].append(
            {
                "human_rank": 0,
                "ranks": [0, 1, 2, 3],
                "opponents": _opus_opponents(),
            }
        )
    for _ in range(600):
        blob["history"].append(
            {
                "human_rank": 1,
                "ranks": [1, 0, 2, 3],
                "opponents": _opus_opponents(),
            }
        )
    anchors = {pc: dict(vals) for pc, vals in RT.REFERENCE_ANCHORS_PER_PC.items()}

    human = RT._compute_human_ratings(blob, anchors)

    assert human["games_4p"] == 1000.0
    assert human["rating_4p"] == pytest.approx(3917.0, abs=5.0)
    assert human["rating_4p"] > RT.calibrate_rating(
        RT.REFERENCE_ANCHORS_PER_PC[4]["heuristic_opus"], 4
    )
