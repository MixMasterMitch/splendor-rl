"""Action encoding invariants."""

from __future__ import annotations

from agent.env import actions as A


def test_action_space_layout():
    assert A.NUM_ACTIONS == 57
    assert A.MAIN_ACTIONS_END == 46
    assert len(A.TAKE3_COMBOS) == 10
    seen = set(A.TAKE3_COMBOS)
    assert len(seen) == 10


def test_encoder_roundtrips_are_injective():
    idxs = []
    for i in range(A.TAKE3_COUNT):
        idxs.append(A.take3_index(i))
    for c in range(5):
        idxs.append(A.take2_index(c))
    for t in range(3):
        for s in range(4):
            idxs.append(A.reserve_grid_index(t, s))
    for t in range(3):
        idxs.append(A.reserve_blind_index(t))
    for t in range(3):
        for s in range(4):
            idxs.append(A.buy_grid_index(t, s))
    for r in range(3):
        idxs.append(A.buy_reserved_index(r))
    idxs.append(A.PASS_ACTION)
    for k in range(6):
        idxs.append(A.discard_index(k))
    for n in range(5):
        idxs.append(A.pick_noble_index(n))
    assert idxs == list(range(A.NUM_ACTIONS))
