"""Flat action encoding for the Splendor engine.

The engine uses a single action index space of size NUM_ACTIONS for all three
phases (main, discard, noble-pick). Legality is controlled by per-phase masks.

Layout (indices):
- [0, 10)  TAKE3: 10 "take 3 different colors" actions (one per 3-subset of 5)
- [10, 15) TAKE2: 5 "take 2 of the same color" actions
- [15, 27) RESERVE_GRID: 12 actions (tier in 0..2, slot in 0..3), idx = 15 + tier*4 + slot
- [27, 30) RESERVE_BLIND: 3 (tier in 0..2)
- [30, 42) BUY_GRID: 12 (tier in 0..2, slot in 0..3)
- [42, 45) BUY_RESERVED: 3 (reserve slot in 0..2)
- [45, 46) PASS: 1
- [46, 52) DISCARD: 6 (token kind 0..5 incl. gold)
- [52, 57) PICK_NOBLE: 5 (noble slot in 0..4)

Tiers are 0-indexed: tier 0 is Level 1, tier 1 is Level 2, tier 2 is Level 3.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

TAKE3_BASE: int = 0
TAKE3_COUNT: int = 10
TAKE2_BASE: int = 10
TAKE2_COUNT: int = 5
RESERVE_GRID_BASE: int = 15
RESERVE_GRID_COUNT: int = 12
RESERVE_BLIND_BASE: int = 27
RESERVE_BLIND_COUNT: int = 3
BUY_GRID_BASE: int = 30
BUY_GRID_COUNT: int = 12
BUY_RESERVED_BASE: int = 42
BUY_RESERVED_COUNT: int = 3
PASS_ACTION: int = 45

MAIN_ACTIONS_END: int = 46

DISCARD_BASE: int = 46
DISCARD_COUNT: int = 6
PICK_NOBLE_BASE: int = 52
PICK_NOBLE_COUNT: int = 5

NUM_ACTIONS: int = PICK_NOBLE_BASE + PICK_NOBLE_COUNT

NUM_TIERS: int = 3
NUM_GRID_SLOTS: int = 4
MAX_RESERVED: int = 3
MAX_NOBLE_SLOTS: int = 5

TAKE3_COMBOS: Tuple[Tuple[int, int, int], ...] = tuple(
    combinations(range(5), 3)
)


def take3_index(combo_idx: int) -> int:
    return TAKE3_BASE + combo_idx


def take2_index(color: int) -> int:
    assert 0 <= color < 5
    return TAKE2_BASE + color


def reserve_grid_index(tier: int, slot: int) -> int:
    assert 0 <= tier < NUM_TIERS
    assert 0 <= slot < NUM_GRID_SLOTS
    return RESERVE_GRID_BASE + tier * NUM_GRID_SLOTS + slot


def reserve_blind_index(tier: int) -> int:
    assert 0 <= tier < NUM_TIERS
    return RESERVE_BLIND_BASE + tier


def buy_grid_index(tier: int, slot: int) -> int:
    return BUY_GRID_BASE + tier * NUM_GRID_SLOTS + slot


def buy_reserved_index(rslot: int) -> int:
    assert 0 <= rslot < MAX_RESERVED
    return BUY_RESERVED_BASE + rslot


def discard_index(token_kind: int) -> int:
    assert 0 <= token_kind < 6
    return DISCARD_BASE + token_kind


def pick_noble_index(nslot: int) -> int:
    assert 0 <= nslot < MAX_NOBLE_SLOTS
    return PICK_NOBLE_BASE + nslot


def action_name(a: int) -> str:
    if TAKE3_BASE <= a < TAKE3_BASE + TAKE3_COUNT:
        combo = TAKE3_COMBOS[a - TAKE3_BASE]
        return f"take3({','.join('WBGRK'[c] for c in combo)})"
    if TAKE2_BASE <= a < TAKE2_BASE + TAKE2_COUNT:
        return f"take2({'WBGRK'[a - TAKE2_BASE]})"
    if RESERVE_GRID_BASE <= a < RESERVE_GRID_BASE + RESERVE_GRID_COUNT:
        x = a - RESERVE_GRID_BASE
        return f"reserve_grid(t{x // 4 + 1},s{x % 4})"
    if RESERVE_BLIND_BASE <= a < RESERVE_BLIND_BASE + RESERVE_BLIND_COUNT:
        return f"reserve_blind(t{a - RESERVE_BLIND_BASE + 1})"
    if BUY_GRID_BASE <= a < BUY_GRID_BASE + BUY_GRID_COUNT:
        x = a - BUY_GRID_BASE
        return f"buy_grid(t{x // 4 + 1},s{x % 4})"
    if BUY_RESERVED_BASE <= a < BUY_RESERVED_BASE + BUY_RESERVED_COUNT:
        return f"buy_reserved(r{a - BUY_RESERVED_BASE})"
    if a == PASS_ACTION:
        return "pass"
    if DISCARD_BASE <= a < DISCARD_BASE + DISCARD_COUNT:
        return f"discard({'WBGRKg'[a - DISCARD_BASE]})"
    if PICK_NOBLE_BASE <= a < PICK_NOBLE_BASE + PICK_NOBLE_COUNT:
        return f"pick_noble(n{a - PICK_NOBLE_BASE})"
    return f"invalid({a})"
