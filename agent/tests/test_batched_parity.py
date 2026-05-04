"""Parity test between single-game and batched engines.

For a handful of random seeds and a random legal-action policy, steps both
engines in lockstep and asserts that observable state matches at every turn.
Focus is on the 2-player flow.
"""

from __future__ import annotations

import random

import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.env import single_engine as E


def _aligned_reset(single_seed: int, batched: BE.BatchedEngine) -> E.GameState:
    """Reset both engines such that decks/nobles/first-player match.

    We accomplish alignment by deep-copying visible state from the batched
    engine into a new single-game state.
    """
    b = 0
    nP = batched.num_players
    decks = []
    for t in range(BE.NUM_TIERS):
        top = int(batched.deck_top[b, t])
        ids = [int(x) for x in batched.deck_perm[b, t, :top].tolist() if x >= 0]
        decks.append(ids)
    nobles = [int(x) for x in batched.noble_ids[b].tolist()]
    grid = [[int(batched.grid_card[b, t, s]) for s in range(BE.NUM_GRID_SLOTS)] for t in range(BE.NUM_TIERS)]
    gem_pool = [int(x) for x in batched.gem_pool[b].tolist()]
    cp = int(batched.current_player[b])
    state = E.GameState(
        num_players=nP,
        gem_pool=gem_pool,
        grid=grid,
        decks=decks,
        nobles=nobles,
        players=[E.PlayerState.empty() for _ in range(nP)],
        current_player=cp,
        phase=0,
        turn_count=0,
        last_round_trigger_player=-1,
        round_actions_since_trigger=0,
        ended=False,
    )
    return state


def _assert_states_match(single: E.GameState, batched: BE.BatchedEngine) -> None:
    b = 0
    nP = single.num_players
    assert single.gem_pool == [int(x) for x in batched.gem_pool[b].tolist()], (
        f"gem_pool mismatch: single={single.gem_pool} batched={batched.gem_pool[b].tolist()}"
    )
    for t in range(BE.NUM_TIERS):
        batched_grid = [int(batched.grid_card[b, t, s]) for s in range(BE.NUM_GRID_SLOTS)]
        assert single.grid[t] == batched_grid, f"grid t={t} single={single.grid[t]} batched={batched_grid}"
    assert single.current_player == int(batched.current_player[b])
    assert single.phase == int(batched.phase[b])
    for p in range(nP):
        sp = single.players[p]
        assert sp.tokens == [int(x) for x in batched.tokens[b, p].tolist()], (
            f"tokens[{p}] mismatch: single={sp.tokens} batched={batched.tokens[b, p].tolist()}"
        )
        assert sp.bonuses == [int(x) for x in batched.bonuses[b, p].tolist()]
        assert sp.points == int(batched.points[b, p])


def test_parity_random_rollout():
    for seed in [1, 17, 42]:
        batched = BE.BatchedEngine(batch_size=1, num_players=2, device="cpu", seed=seed)
        single = _aligned_reset(seed, batched)

        rng = random.Random(seed * 31 + 7)
        for turn in range(40):
            if single.ended:
                break
            s_mask = E.legal_action_mask(single)
            b_mask = batched.legal_action_mask()[0].tolist()
            # Intersection of legal actions (both engines should agree for aligned states)
            assert s_mask == b_mask, (
                f"turn {turn}: legal mask differs\nsingle: {[i for i,x in enumerate(s_mask) if x]}\nbatched: {[i for i,x in enumerate(b_mask) if x]}"
            )
            legal_idx = [i for i, x in enumerate(s_mask) if x]
            if not legal_idx:
                break
            a = rng.choice(legal_idx)
            E.apply_action(single, a)
            batched.apply(torch.tensor([a], dtype=torch.int64))
            _assert_states_match(single, batched)
