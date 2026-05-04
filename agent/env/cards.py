"""Load and validate Splendor card and noble data from CSVs.

CSV layout documented in RULES.md. Both CSVs use color column order
`Black, Blue, Green, Red, White`. Internally we remap to the canonical order
`(White, Blue, Green, Red, Black)` indexed 0..4 as `COLOR_NAMES` below.

This module exposes typed, frozen data structures so the engine can rely on
stable indices throughout training.
"""

from __future__ import annotations

import csv
import dataclasses
import os
from typing import List, Tuple

COLOR_NAMES: Tuple[str, ...] = ("White", "Blue", "Green", "Red", "Black")
NUM_COLORS: int = 5
GOLD_INDEX: int = 5
NUM_TOKEN_KINDS: int = 6

COLOR_W, COLOR_B, COLOR_G, COLOR_R, COLOR_K = 0, 1, 2, 3, 4

_CSV_COLOR_ORDER: Tuple[str, ...] = ("Black", "Blue", "Green", "Red", "White")
_CSV_TO_CANON = {"White": 0, "Blue": 1, "Green": 2, "Red": 3, "Black": 4}


@dataclasses.dataclass(frozen=True)
class Card:
    card_id: int
    level: int
    bonus: int
    points: int
    cost: Tuple[int, int, int, int, int]


@dataclasses.dataclass(frozen=True)
class Noble:
    noble_id: int
    name: str
    points: int
    requirement: Tuple[int, int, int, int, int]


def _csv_row_to_cost(row: dict) -> Tuple[int, int, int, int, int]:
    cost = [0, 0, 0, 0, 0]
    for csv_name in _CSV_COLOR_ORDER:
        cost[_CSV_TO_CANON[csv_name]] = int(row[csv_name])
    return tuple(cost)


def _bonus_from_name(name: str) -> int:
    return _CSV_TO_CANON[name]


def load_cards(path: str) -> List[Card]:
    cards: List[Card] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            cards.append(
                Card(
                    card_id=i,
                    level=int(row["Level"]),
                    bonus=_bonus_from_name(row["Color"]),
                    points=int(row["PV"]),
                    cost=_csv_row_to_cost(row),
                )
            )
    return cards


def load_nobles(path: str) -> List[Noble]:
    nobles: List[Noble] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            nobles.append(
                Noble(
                    noble_id=i,
                    name=row["Name"],
                    points=int(row["PV"]),
                    requirement=_csv_row_to_cost(row),
                )
            )
    return nobles


def token_supply_for_players(num_players: int) -> Tuple[int, int, int, int, int, int]:
    """Returns (W, B, G, R, K, Gold) supply counts."""
    if num_players == 2:
        per_color = 4
    elif num_players == 3:
        per_color = 5
    elif num_players == 4:
        per_color = 7
    else:
        raise ValueError(f"num_players must be 2, 3, or 4; got {num_players}")
    return (per_color, per_color, per_color, per_color, per_color, 5)


def num_nobles_for_players(num_players: int) -> int:
    if num_players not in (2, 3, 4):
        raise ValueError(f"num_players must be 2, 3, or 4; got {num_players}")
    return num_players + 1


def validate_cards(cards: List[Card]) -> None:
    """Validates checks listed in RULES.md section 11."""
    assert len(cards) == 90, f"expected 90 cards, got {len(cards)}"
    per_level: dict = {1: 0, 2: 0, 3: 0}
    per_level_bonus: dict = {}
    for c in cards:
        assert c.level in (1, 2, 3), f"bad level {c.level}"
        assert 0 <= c.bonus < NUM_COLORS, f"bad bonus {c.bonus}"
        assert 0 <= c.points <= 5, f"bad points {c.points}"
        assert all(0 <= x < 128 for x in c.cost), f"cost must be int8"
        per_level[c.level] += 1
        key = (c.level, c.bonus)
        per_level_bonus[key] = per_level_bonus.get(key, 0) + 1
    assert per_level == {1: 40, 2: 30, 3: 20}, f"per-level: {per_level}"
    expected_per_level_bonus = {1: 8, 2: 6, 3: 4}
    for (lvl, bonus), cnt in per_level_bonus.items():
        assert cnt == expected_per_level_bonus[lvl], (
            f"level {lvl} bonus {bonus}: got {cnt}, expected {expected_per_level_bonus[lvl]}"
        )


def validate_nobles(nobles: List[Noble]) -> None:
    assert len(nobles) == 10, f"expected 10 nobles, got {len(nobles)}"
    pair_count = 0
    triple_count = 0
    sigs = set()
    for n in nobles:
        assert n.points == 3, f"nobles always worth 3 PV; got {n.points}"
        nonzero = [c for c in n.requirement if c > 0]
        assert all(c >= 0 for c in n.requirement)
        sig = tuple(n.requirement)
        assert sig not in sigs, f"duplicate noble requirement {sig}"
        sigs.add(sig)
        if len(nonzero) == 2 and all(c == 4 for c in nonzero) and sum(n.requirement) == 8:
            pair_count += 1
        elif len(nonzero) == 3 and all(c == 3 for c in nonzero) and sum(n.requirement) == 9:
            triple_count += 1
        else:
            raise AssertionError(f"noble {n.name} has invalid pattern {n.requirement}")
    assert pair_count == 5, f"expected 5 4+4 nobles, got {pair_count}"
    assert triple_count == 5, f"expected 5 3+3+3 nobles, got {triple_count}"


def validate_token_supply() -> None:
    for np_ in (2, 3, 4):
        ts = token_supply_for_players(np_)
        assert ts[GOLD_INDEX] == 5
        per_color = {2: 4, 3: 5, 4: 7}[np_]
        assert all(ts[i] == per_color for i in range(NUM_COLORS)), ts


def _default_data_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_default_cards() -> List[Card]:
    return load_cards(os.path.join(_default_data_dir(), "splendor_cards.csv"))


def load_default_nobles() -> List[Noble]:
    return load_nobles(os.path.join(_default_data_dir(), "splendor_nobles.csv"))


def load_and_validate_all() -> Tuple[List[Card], List[Noble]]:
    cards = load_default_cards()
    nobles = load_default_nobles()
    validate_cards(cards)
    validate_nobles(nobles)
    validate_token_supply()
    return cards, nobles


CARDS: List[Card] = load_default_cards()
NOBLES: List[Noble] = load_default_nobles()
validate_cards(CARDS)
validate_nobles(NOBLES)
validate_token_supply()
