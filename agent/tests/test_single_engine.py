"""Rule unit tests for the single-game reference engine."""

from __future__ import annotations

import random

from agent.env import actions as A
from agent.env import cards as C
from agent.env import single_engine as E


def _fresh(num_players: int = 2, seed: int = 7) -> E.GameState:
    return E.create_game(num_players, random.Random(seed))


def test_create_game_initial_state():
    s = _fresh(2)
    assert s.num_players == 2
    assert s.gem_pool == [4, 4, 4, 4, 4, 5]
    assert len([n for n in s.nobles if n >= 0]) == 3
    for t in range(3):
        assert all(s.grid[t][sl] >= 0 for sl in range(4))
    assert sum(len(d) for d in s.decks) == 90 - 12
    assert s.phase == 0
    assert not s.ended


def test_take3_and_take2_gem_flow():
    s = _fresh(2)
    mask = E.legal_action_mask(s)
    assert mask[A.take3_index(0)]
    assert mask[A.take2_index(0)]
    p_before = list(s.gem_pool)
    E.apply_action(s, A.take3_index(0))
    assert sum(s.players[0].tokens) == 3
    assert s.gem_pool[0] == p_before[0] - 1
    assert s.gem_pool[1] == p_before[1] - 1
    assert s.gem_pool[2] == p_before[2] - 1


def test_take2_requires_4_tokens():
    s = _fresh(2)
    s.gem_pool[0] = 3
    mask = E.legal_action_mask(s)
    assert not mask[A.take2_index(0)]
    s.gem_pool[0] = 4
    mask = E.legal_action_mask(s)
    assert mask[A.take2_index(0)]


def test_reserve_and_gold():
    s = _fresh(2)
    E.apply_action(s, A.reserve_grid_index(0, 0))
    p = s.players[0]
    assert p.reserved[0] >= 0
    assert p.tokens[C.GOLD_INDEX] == 1
    assert s.gem_pool[C.GOLD_INDEX] == 4


def test_buy_card_flow_with_cheap_card():
    s = _fresh(2, seed=13)
    p = s.players[0]
    cheapest_tier0 = min((E._card(s.grid[0][sl]) for sl in range(4)), key=lambda c: sum(c.cost))
    for i in range(5):
        p.tokens[i] = cheapest_tier0.cost[i]
    cid = None
    for sl in range(4):
        if s.grid[0][sl] == cheapest_tier0.card_id:
            cid = (0, sl)
            break
    assert cid is not None
    mask = E.legal_action_mask(s)
    assert mask[A.buy_grid_index(cid[0], cid[1])]
    E.apply_action(s, A.buy_grid_index(cid[0], cid[1]))
    assert p.points == cheapest_tier0.points
    assert p.bonuses[cheapest_tier0.bonus] == 1
    for i in range(5):
        assert p.tokens[i] == 0
    for i in range(5):
        assert s.gem_pool[i] == 4 + cheapest_tier0.cost[i]


def test_discard_phase_triggers_over_10_tokens():
    s = _fresh(2)
    p = s.players[0]
    p.tokens = [3, 3, 3, 0, 0, 0]
    s.gem_pool = [10, 10, 10, 10, 10, 5]
    E.apply_action(s, A.take2_index(0))
    assert s.phase == 1
    assert s.current_player == 0
    assert sum(p.tokens) == 11
    mask = E.legal_action_mask(s)
    assert mask[A.discard_index(0)]
    E.apply_action(s, A.discard_index(0))
    assert sum(p.tokens) == 10
    assert s.phase == 0
    assert s.current_player == 1


def test_game_ends_after_full_round_once_someone_hits_15():
    s = _fresh(2, seed=1)
    s.players[0].points = 15
    s.last_round_trigger_player = 0
    mask_p0 = E.legal_action_mask(s)
    E.apply_action(s, mask_p0.index(True))
    assert not s.ended
    assert s.current_player == 1
    mask = E.legal_action_mask(s)
    a = mask.index(True)
    E.apply_action(s, a)
    assert s.ended


def test_winner_tiebreak_by_fewer_cards():
    s = _fresh(2)
    s.players[0].points = 15
    s.players[1].points = 15
    s.players[0].bonuses = [1, 1, 1, 1, 1]
    s.players[1].bonuses = [1, 1, 1, 0, 0]
    s.ended = True
    w = E.winners(s)
    assert w == [1]
