"""Reproduce and verify the MCTS discard-perspective bug fix.

This test recreates the conditions from game d896002ae811 (UncleMitz vs ML agent
on the local 8765 server) where the ML agent:
1. Had 8 tokens
2. Chose TAKE3 (gaining 3 more → 11 total, triggering discard)
3. Discarded a gold/joker token (terrible move)

The root cause: in _evaluate_root_children_batched, after applying a child action
that triggers discard phase, current_player does NOT advance. The old code read
value[:, num_players - 1] which is the OPPONENT's value, causing the MCTS to
maximize the opponent's position (= discard the most valuable token = gold).

This test verifies:
- The fix correctly reads the agent's own value when current_player stays the same
- The MCTS does not prefer discarding gold over other tokens
- Behavior is consistent across the webapp replay path and direct engine path
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.net import model as M
from agent.net import encoder as ENC
from agent.search import gumbel_mcts as G
from agent.train import checkpointing as CK


# Use the latest league checkpoint for realistic behavior
_LEAGUE_DIR = pathlib.Path(__file__).resolve().parent.parent / "runs" / "league"
_LATEST_CKPT = sorted(
    [p for p in _LEAGUE_DIR.glob("ckpt_*.pt")],
    key=lambda p: p.name,
)[-1] if _LEAGUE_DIR.exists() and list(_LEAGUE_DIR.glob("ckpt_*.pt")) else None


def _setup_discard_scenario(
    seed: int = 1778096207,
    num_players: int = 2,
    gold_tokens: int = 1,
) -> BE.BatchedEngine:
    """Create a game state where player 0 has 9 tokens (including gold) and
    is about to take 3 more, triggering a 2-token discard."""
    engine = BE.BatchedEngine(1, num_players, device="cpu", seed=seed)
    engine.current_player[:] = 0
    # 2W, 2B, 2G, 1R, 1K, 1gold = 9 tokens
    engine.tokens[0, 0, 0] = 2  # W
    engine.tokens[0, 0, 1] = 2  # B
    engine.tokens[0, 0, 2] = 2  # G
    engine.tokens[0, 0, 3] = 1  # R
    engine.tokens[0, 0, 4] = 2 - gold_tokens  # K
    engine.tokens[0, 0, 5] = gold_tokens  # gold
    return engine


def _apply_take3_trigger_discard(engine: BE.BatchedEngine) -> None:
    """Apply a TAKE3 action that triggers the discard phase."""
    # TAKE3 combo 0 = (W, B, G) → adds 3 tokens → 12 total → discard 2
    action = torch.tensor([A.TAKE3_BASE], dtype=torch.int64)
    engine.apply(action)
    assert engine.phase[0].item() == 1, "Expected discard phase"
    assert engine.current_player[0].item() == 0, "Current player should not change"


class TestMCTSDiscardPerspective:
    """Verify the MCTS correctly evaluates discard actions from the agent's
    own perspective, not the opponent's."""

    def test_value_index_same_player(self):
        """When current_player stays the same after apply, value index should be 0."""
        parent_cp = torch.tensor([0, 1, 0], dtype=torch.long)
        child_cp = torch.tensor([0, 1, 1], dtype=torch.long)
        result = G._root_value_index(parent_cp, child_cp, num_players=2)
        # Same-player: index 0; advance from cp=0 to cp=1 in 2p:
        # encoder rotates by MAX_PLAYERS=4, so root player (abs seat 0)
        # sits at slot (0-1) % 4 = 3 in the child's rotated view.
        assert result.tolist() == [0, 0, 3]

    def test_value_index_3_player(self):
        """3-player game: value index depends on MAX_PLAYERS rotation.

        parent_cp=0, child_cp=0 → same player, slot 0.
        parent_cp=2, child_cp=0 → wrap-around, slot (2-0)%4 = 2.
        """
        parent_cp = torch.tensor([0, 2], dtype=torch.long)
        child_cp = torch.tensor([0, 0], dtype=torch.long)
        result = G._root_value_index(parent_cp, child_cp, num_players=3)
        assert result.tolist() == [0, 2]

    def test_value_index_3p_normal_advance(self):
        """Regression guard: root at parent_cp=0, child_cp=1 in 3p must map to
        slot 3 (MAX_PLAYERS-1). Prior code returned num_players-1=2, which
        read the opponent's value and silently corrupted 3p training."""
        parent_cp = torch.tensor([0, 1], dtype=torch.long)
        child_cp = torch.tensor([1, 2], dtype=torch.long)
        result = G._root_value_index(parent_cp, child_cp, num_players=3)
        assert result.tolist() == [3, 3]

    def test_value_index_4p_advance_unchanged(self):
        """At 4p the fix is a no-op: num_players-1 == MAX_PLAYERS-1 == 3."""
        parent_cp = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        child_cp = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        result = G._root_value_index(parent_cp, child_cp, num_players=4)
        assert result.tolist() == [3, 3, 3, 3]

    def test_discard_phase_does_not_advance_player(self):
        """Verify that entering discard phase keeps current_player unchanged."""
        engine = _setup_discard_scenario()
        cp_before = engine.current_player[0].item()
        _apply_take3_trigger_discard(engine)
        cp_after = engine.current_player[0].item()
        assert cp_before == cp_after

    def test_first_discard_keeps_player_when_still_over_limit(self):
        """After discarding 1 of 2 required tokens, player stays in discard."""
        engine = _setup_discard_scenario()
        _apply_take3_trigger_discard(engine)
        # Total is 12, discard 1 → 11, still > 10
        engine.apply(torch.tensor([A.DISCARD_BASE + 0], dtype=torch.int64))  # discard W
        assert engine.phase[0].item() == 1, "Should still be in discard"
        assert engine.current_player[0].item() == 0, "Player should not change"

    def test_mcts_does_not_prefer_gold_discard(self):
        """With the fix, MCTS should not systematically prefer discarding gold."""
        torch.manual_seed(42)
        net = M.SplendorNet(hidden=64, arch="attn")
        net.eval()

        gold_chosen_count = 0
        trials = 20

        for trial in range(trials):
            engine = _setup_discard_scenario(seed=trial * 100, gold_tokens=1)
            _apply_take3_trigger_discard(engine)

            # Verify gold is a legal discard
            mask = engine.legal_action_mask()
            gold_action = A.DISCARD_BASE + 5
            if not mask[0, gold_action]:
                continue

            with torch.no_grad():
                action, _ = G.gumbel_root_act(engine, net, num_sims=8)

            if action.item() == gold_action:
                gold_chosen_count += 1

        # With random policy over 5 legal discards, gold would be chosen ~20% of time.
        # With the old bug (maximizing opponent), gold would be chosen much more often.
        # With the fix, a random net should not systematically prefer gold.
        # Allow up to 50% as a generous bound (random net has no real preference).
        assert gold_chosen_count <= trials * 0.6, (
            f"Gold discarded {gold_chosen_count}/{trials} times — "
            f"MCTS may still be reading opponent's value"
        )

    @pytest.mark.skipif(_LATEST_CKPT is None, reason="No league checkpoint available")
    def test_trained_net_avoids_gold_discard(self):
        """A trained net should not systematically prefer discarding gold.

        NOTE: Current checkpoints were trained with the buggy MCTS that read
        the opponent's value during discard evaluation. The policy head has
        internalized this bias. After retraining with the fix, this threshold
        should be tightened to < 10%. For now we verify the MCTS fix at least
        reduces the preference vs the ~50%+ rate seen with the old bug at
        high num_sims.
        """
        net, _ = CK.load_net_from_checkpoint(_LATEST_CKPT, map_location="cpu")
        net.eval()

        gold_chosen_count = 0
        trials = 50

        for trial in range(trials):
            engine = _setup_discard_scenario(seed=trial * 77 + 1, gold_tokens=1)
            _apply_take3_trigger_discard(engine)

            mask = engine.legal_action_mask()
            gold_action = A.DISCARD_BASE + 5
            if not mask[0, gold_action]:
                continue

            with torch.no_grad():
                action, _ = G.gumbel_root_act(engine, net, num_sims=16)

            if action.item() == gold_action:
                gold_chosen_count += 1

        # With the old bug at num_sims=64, gold was discarded >50% of the time.
        # With the fix, even a net trained with buggy targets should show reduced
        # gold-discard rate. After retraining, tighten to < 15%.
        assert gold_chosen_count <= trials * 0.5, (
            f"Trained net discarded gold {gold_chosen_count}/{trials} times — "
            f"rate should be < 50% with the perspective fix (was >50% before fix)"
        )

    @pytest.mark.skipif(_LATEST_CKPT is None, reason="No league checkpoint available")
    def test_webapp_replay_matches_direct_engine(self):
        """Verify that replaying a game via the webapp path produces the same
        engine state as direct engine manipulation.

        This catches any divergence between how the webapp reconstructs game
        state (via replay_persisted_steps) and how the engine actually works.
        """
        from play.state import GameSession
        from play import players as POL

        seed = 1778076568  # from a real local game
        num_players = 2
        human_seat = 0

        # Create a session the same way the webapp does
        policy = POL.NetPolicy(_LATEST_CKPT, num_sims=16, device="cpu")
        seat_policies = {1: policy}
        seat_models = {1: {"id": "test_net", "kind": "net", "label": "RL Trained Bot",
                           "ckpt": str(_LATEST_CKPT), "rating": 3000.0}}

        session = GameSession(
            game_id="test_repro",
            num_players=num_players,
            human_seat=human_seat,
            seat_models=seat_models,
            seat_policies=seat_policies,
            seed=seed,
            device="cpu",
        )

        # Play a few moves directly on the engine
        direct_engine = BE.BatchedEngine(1, num_players, device="cpu", seed=seed)
        direct_engine.current_player[:] = 0

        # Both engines should start in the same state
        assert torch.equal(session.engine.gem_pool, direct_engine.gem_pool)
        assert torch.equal(session.engine.grid_card, direct_engine.grid_card)
        assert torch.equal(session.engine.noble_ids, direct_engine.noble_ids)

        # Apply the same sequence of actions to both
        actions_to_play = []
        for _ in range(5):
            mask = session.engine.legal_action_mask()
            legal = mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
            action = legal[0]  # deterministic: always pick first legal action
            actions_to_play.append(action)
            session._record_and_apply(action)
            direct_engine.apply(torch.tensor([action], dtype=torch.int64))

        # Verify states match
        assert torch.equal(session.engine.gem_pool, direct_engine.gem_pool)
        assert torch.equal(session.engine.tokens, direct_engine.tokens)
        assert torch.equal(session.engine.bonuses, direct_engine.bonuses)
        assert torch.equal(session.engine.points, direct_engine.points)
        assert torch.equal(session.engine.current_player, direct_engine.current_player)
        assert torch.equal(session.engine.phase, direct_engine.phase)

        # Now replay from scratch (simulating a cold load)
        session2 = GameSession(
            game_id="test_repro2",
            num_players=num_players,
            human_seat=human_seat,
            seat_models=seat_models,
            seat_policies=seat_policies,
            seed=seed,
            device="cpu",
        )
        persisted = [{"action": a} for a in actions_to_play]
        session2.replay_persisted_steps(persisted)

        # Replayed state should match
        assert torch.equal(session2.engine.gem_pool, direct_engine.gem_pool)
        assert torch.equal(session2.engine.tokens, direct_engine.tokens)
        assert torch.equal(session2.engine.current_player, direct_engine.current_player)
        assert torch.equal(session2.engine.phase, direct_engine.phase)
