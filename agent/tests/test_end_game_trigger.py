"""Tests for end-game trigger logic (Splendor last-round rule).

The rule: when a player reaches 15+ prestige points, the current round is
completed so that all players have had equal total turns. Then the game ends.

Key scenarios:
- If the FIRST player in a round triggers, all other players get one more turn.
- If the LAST player in a round triggers, the game ends immediately (round complete).
- If a MIDDLE player triggers (3/4p), only the players after them in the round
  still get their turn.
"""

from __future__ import annotations

import random

import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.env import single_engine as E


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_single(num_players: int = 2, seed: int = 7) -> E.GameState:
    s = E.create_game(num_players, random.Random(seed))
    # Normalize: seat 0 always starts for test predictability
    s.current_player = 0
    s.first_player = 0
    return s


def _fresh_batched(num_players: int = 2, seed: int = 7) -> BE.BatchedEngine:
    engine = BE.BatchedEngine(1, num_players, device="cpu", seed=seed)
    engine.current_player[:] = 0
    engine.first_player[:] = 0
    return engine


def _any_legal_action(state: E.GameState) -> int:
    mask = E.legal_action_mask(state)
    return mask.index(True)


def _any_legal_action_batched(engine: BE.BatchedEngine) -> int:
    mask = engine.legal_action_mask()[0]
    return int(mask.nonzero(as_tuple=False)[0, 0].item())


def _play_one_turn_single(state: E.GameState) -> None:
    """Play one legal action in the single engine."""
    E.apply_action(state, _any_legal_action(state))
    # If we land in discard phase, discard until resolved
    while state.phase == 1 and not state.ended:
        E.apply_action(state, _any_legal_action(state))
    # If we land in noble-pick phase, pick
    while state.phase == 2 and not state.ended:
        E.apply_action(state, _any_legal_action(state))


def _play_one_turn_batched(engine: BE.BatchedEngine) -> None:
    """Play one legal action in the batched engine."""
    engine.apply(torch.tensor([_any_legal_action_batched(engine)], dtype=torch.int64))
    # Resolve sub-phases
    while int(engine.phase[0].item()) != 0 and not bool(engine.ended[0].item()):
        engine.apply(torch.tensor([_any_legal_action_batched(engine)], dtype=torch.int64))


# ---------------------------------------------------------------------------
# 2-Player Tests (Single Engine)
# ---------------------------------------------------------------------------


class TestEndGame2PlayerSingle:
    """2-player end-game trigger in single engine."""

    def test_first_player_triggers_second_gets_one_more_turn(self):
        """Seat 0 triggers → seat 1 gets one more turn → game ends."""
        s = _fresh_single(2, seed=1)
        s.current_player = 0
        s.players[0].points = 15

        # Seat 0 plays (trigger fires)
        _play_one_turn_single(s)
        assert not s.ended, "Game should NOT end immediately when first player triggers"
        assert s.current_player == 1, "Should be seat 1's turn"
        assert s.last_round_trigger_player == 0

        # Seat 1 plays (round complete, game ends)
        _play_one_turn_single(s)
        assert s.ended, "Game should end after seat 1 completes the round"

    def test_second_player_triggers_game_ends_immediately(self):
        """Seat 1 triggers → round already complete → game ends immediately."""
        s = _fresh_single(2, seed=1)
        s.current_player = 1
        s.players[1].points = 15

        # Seat 1 plays (trigger fires, round complete since seat 0 already went)
        _play_one_turn_single(s)
        assert s.ended, "Game should end immediately when last player in round triggers"

    def test_trigger_on_turn_that_reaches_15(self):
        """Trigger fires on the turn a player actually reaches 15 points."""
        s = _fresh_single(2, seed=1)
        s.current_player = 1
        s.players[1].points = 14
        # Give player enough to buy a 1-point card (we'll just set to 15 manually
        # and verify the trigger logic)
        s.players[1].points = 15

        _play_one_turn_single(s)
        assert s.ended


# ---------------------------------------------------------------------------
# 3-Player Tests (Single Engine)
# ---------------------------------------------------------------------------


class TestEndGame3PlayerSingle:
    """3-player end-game trigger in single engine."""

    def test_first_player_triggers_others_get_turns(self):
        """Seat 0 triggers → seats 1 and 2 each get one more turn."""
        s = _fresh_single(3, seed=5)
        s.current_player = 0
        s.players[0].points = 15

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 1

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 2

        _play_one_turn_single(s)
        assert s.ended

    def test_middle_player_triggers(self):
        """Seat 1 triggers → seat 2 gets one more turn, seat 0 does not."""
        s = _fresh_single(3, seed=5)
        s.current_player = 1
        s.players[1].points = 15

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 2

        _play_one_turn_single(s)
        assert s.ended

    def test_last_player_triggers_ends_immediately(self):
        """Seat 2 triggers → round complete → game ends immediately."""
        s = _fresh_single(3, seed=5)
        s.current_player = 2
        s.players[2].points = 15

        _play_one_turn_single(s)
        assert s.ended


# ---------------------------------------------------------------------------
# 4-Player Tests (Single Engine)
# ---------------------------------------------------------------------------


class TestEndGame4PlayerSingle:
    """4-player end-game trigger in single engine."""

    def test_first_player_triggers(self):
        """Seat 0 triggers → seats 1, 2, 3 each get one more turn."""
        s = _fresh_single(4, seed=3)
        s.current_player = 0
        s.players[0].points = 15

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 1

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 2

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 3

        _play_one_turn_single(s)
        assert s.ended

    def test_second_player_triggers(self):
        """Seat 1 triggers → seats 2, 3 get turns, seat 0 does not."""
        s = _fresh_single(4, seed=3)
        s.current_player = 1
        s.players[1].points = 15

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 2

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 3

        _play_one_turn_single(s)
        assert s.ended

    def test_third_player_triggers(self):
        """Seat 2 triggers → seat 3 gets one turn, seats 0 and 1 do not."""
        s = _fresh_single(4, seed=3)
        s.current_player = 2
        s.players[2].points = 15

        _play_one_turn_single(s)
        assert not s.ended
        assert s.current_player == 3

        _play_one_turn_single(s)
        assert s.ended

    def test_last_player_triggers_ends_immediately(self):
        """Seat 3 triggers → round complete → game ends immediately."""
        s = _fresh_single(4, seed=3)
        s.current_player = 3
        s.players[3].points = 15

        _play_one_turn_single(s)
        assert s.ended


# ---------------------------------------------------------------------------
# 2-Player Tests (Batched Engine)
# ---------------------------------------------------------------------------


class TestEndGame2PlayerBatched:
    """2-player end-game trigger in batched engine."""

    def test_first_player_triggers_second_gets_one_more_turn(self):
        """Seat 0 triggers → seat 1 gets one more turn → game ends."""
        engine = _fresh_batched(2, seed=1)
        engine.current_player[:] = 0
        engine.points[0, 0] = 15

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())
        assert int(engine.current_player[0].item()) == 1
        assert int(engine.last_trigger[0].item()) == 0

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())

    def test_second_player_triggers_game_ends_immediately(self):
        """Seat 1 triggers → game ends immediately."""
        engine = _fresh_batched(2, seed=1)
        engine.current_player[:] = 1
        engine.points[0, 1] = 15

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())


# ---------------------------------------------------------------------------
# 3-Player Tests (Batched Engine)
# ---------------------------------------------------------------------------


class TestEndGame3PlayerBatched:
    """3-player end-game trigger in batched engine."""

    def test_first_player_triggers(self):
        engine = _fresh_batched(3, seed=5)
        engine.current_player[:] = 0
        engine.points[0, 0] = 15

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())
        assert int(engine.current_player[0].item()) == 1

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())
        assert int(engine.current_player[0].item()) == 2

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())

    def test_middle_player_triggers(self):
        engine = _fresh_batched(3, seed=5)
        engine.current_player[:] = 1
        engine.points[0, 1] = 15

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())
        assert int(engine.current_player[0].item()) == 2

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())

    def test_last_player_triggers_ends_immediately(self):
        engine = _fresh_batched(3, seed=5)
        engine.current_player[:] = 2
        engine.points[0, 2] = 15

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())


# ---------------------------------------------------------------------------
# 4-Player Tests (Batched Engine)
# ---------------------------------------------------------------------------


class TestEndGame4PlayerBatched:
    """4-player end-game trigger in batched engine."""

    def test_first_player_triggers(self):
        engine = _fresh_batched(4, seed=3)
        engine.current_player[:] = 0
        engine.points[0, 0] = 15

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())

    def test_second_player_triggers(self):
        engine = _fresh_batched(4, seed=3)
        engine.current_player[:] = 1
        engine.points[0, 1] = 15

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())

    def test_third_player_triggers(self):
        engine = _fresh_batched(4, seed=3)
        engine.current_player[:] = 2
        engine.points[0, 2] = 15

        _play_one_turn_batched(engine)
        assert not bool(engine.ended[0].item())

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())

    def test_last_player_triggers_ends_immediately(self):
        engine = _fresh_batched(4, seed=3)
        engine.current_player[:] = 3
        engine.points[0, 3] = 15

        _play_one_turn_batched(engine)
        assert bool(engine.ended[0].item())


# ---------------------------------------------------------------------------
# Parity: single vs batched end-game behavior
# ---------------------------------------------------------------------------


class TestEndGameParity:
    """Verify single and batched engines agree on end-game timing."""

    def _run_parity(self, num_players: int, trigger_seat: int, seed: int = 42):
        s = _fresh_single(num_players, seed)
        engine = _fresh_batched(num_players, seed)

        # Align initial state: set current_player and points
        s.current_player = trigger_seat
        s.first_player = 0  # round starts at seat 0
        engine.current_player[:] = trigger_seat
        engine.first_player[:] = 0
        s.players[trigger_seat].points = 15
        engine.points[0, trigger_seat] = 15

        # Play until one of them ends (they should agree)
        for _ in range(num_players + 1):
            if s.ended:
                break
            # Get legal actions from both
            s_mask = E.legal_action_mask(s)
            b_mask = engine.legal_action_mask()[0].tolist()
            # Use first legal action from single engine
            action = s_mask.index(True)
            assert b_mask[action], "Batched engine should agree action is legal"

            E.apply_action(s, action)
            engine.apply(torch.tensor([action], dtype=torch.int64))

            # Resolve sub-phases in lockstep
            while s.phase != 0 and not s.ended:
                sa = _any_legal_action(s)
                E.apply_action(s, sa)
                engine.apply(torch.tensor([sa], dtype=torch.int64))

            assert s.ended == bool(engine.ended[0].item()), (
                f"Parity mismatch: single.ended={s.ended}, "
                f"batched.ended={bool(engine.ended[0].item())}"
            )

        assert s.ended, "Game should have ended"
        assert bool(engine.ended[0].item()), "Batched game should have ended"

    def test_parity_2p_seat0_triggers(self):
        self._run_parity(2, trigger_seat=0)

    def test_parity_2p_seat1_triggers(self):
        self._run_parity(2, trigger_seat=1)

    def test_parity_3p_seat0_triggers(self):
        self._run_parity(3, trigger_seat=0)

    def test_parity_3p_seat1_triggers(self):
        self._run_parity(3, trigger_seat=1)

    def test_parity_3p_seat2_triggers(self):
        self._run_parity(3, trigger_seat=2)

    def test_parity_4p_seat0_triggers(self):
        self._run_parity(4, trigger_seat=0)

    def test_parity_4p_seat1_triggers(self):
        self._run_parity(4, trigger_seat=1)

    def test_parity_4p_seat2_triggers(self):
        self._run_parity(4, trigger_seat=2)

    def test_parity_4p_seat3_triggers(self):
        self._run_parity(4, trigger_seat=3)


# ---------------------------------------------------------------------------
# Turn counting: verify exact number of turns after trigger
# ---------------------------------------------------------------------------


class TestTurnCountAfterTrigger:
    """Verify the exact number of turns played after the trigger."""

    def test_2p_first_triggers_one_extra_turn(self):
        """When seat 0 triggers in 2p, exactly 1 more turn is played."""
        s = _fresh_single(2, seed=10)
        s.current_player = 0
        s.players[0].points = 15

        turns_after_trigger = 0
        _play_one_turn_single(s)  # seat 0's turn (trigger fires)
        while not s.ended:
            _play_one_turn_single(s)
            turns_after_trigger += 1

        assert turns_after_trigger == 1

    def test_2p_second_triggers_zero_extra_turns(self):
        """When seat 1 triggers in 2p, 0 more turns are played."""
        s = _fresh_single(2, seed=10)
        s.current_player = 1
        s.players[1].points = 15

        _play_one_turn_single(s)  # seat 1's turn (trigger fires, game ends)
        assert s.ended

    def test_3p_first_triggers_two_extra_turns(self):
        s = _fresh_single(3, seed=10)
        s.current_player = 0
        s.players[0].points = 15

        turns_after_trigger = 0
        _play_one_turn_single(s)
        while not s.ended:
            _play_one_turn_single(s)
            turns_after_trigger += 1

        assert turns_after_trigger == 2

    def test_3p_middle_triggers_one_extra_turn(self):
        s = _fresh_single(3, seed=10)
        s.current_player = 1
        s.players[1].points = 15

        turns_after_trigger = 0
        _play_one_turn_single(s)
        while not s.ended:
            _play_one_turn_single(s)
            turns_after_trigger += 1

        assert turns_after_trigger == 1

    def test_4p_seat1_triggers_two_extra_turns(self):
        s = _fresh_single(4, seed=10)
        s.current_player = 1
        s.players[1].points = 15

        turns_after_trigger = 0
        _play_one_turn_single(s)
        while not s.ended:
            _play_one_turn_single(s)
            turns_after_trigger += 1

        assert turns_after_trigger == 2

    def test_4p_seat2_triggers_one_extra_turn(self):
        s = _fresh_single(4, seed=10)
        s.current_player = 2
        s.players[2].points = 15

        turns_after_trigger = 0
        _play_one_turn_single(s)
        while not s.ended:
            _play_one_turn_single(s)
            turns_after_trigger += 1

        assert turns_after_trigger == 1
