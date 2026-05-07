"""In-memory game session built from engine + policies."""

from __future__ import annotations

import threading
from typing import Any

import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from replay import players as POL
from play import views as V


class GameSession:
    """Single game (B=1 ``BatchedEngine`` + per-seat policies + step log)."""

    def __init__(
        self,
        game_id: str,
        num_players: int,
        human_seat: int,
        seat_models: dict[int, dict[str, Any]],
        seat_policies: dict[int, POL.PlayerPolicy],
        seed: int,
        device: str,
        steps: list[dict[str, Any]] | None = None,
        aborted: bool = False,
        rating_update: dict[str, Any] | None = None,
        initial_state_override: dict[str, Any] | None = None,
    ) -> None:
        if num_players not in (2, 3, 4):
            raise ValueError(f"num_players must be 2,3,4; got {num_players}")
        if not 0 <= human_seat < num_players:
            raise ValueError(f"human_seat out of range: {human_seat}")
        self.game_id = game_id
        self.num_players = num_players
        self.human_seat = human_seat
        self.seat_models = seat_models
        self.seat_policies = seat_policies
        self.seed = seed
        self.device = device
        self.saved_num_sims = 8
        self.lock = threading.Lock()
        self.engine = BE.BatchedEngine(1, num_players, device=device, seed=seed)
        # In human play, seat 0 always goes first (deterministic turn order).
        # The random first-player selection is only for training.
        self.engine.current_player[:] = 0
        self.initial_state = (
            initial_state_override
            if initial_state_override is not None
            else V.batched_to_snapshot(self.engine, 0)
        )
        self.steps: list[dict[str, Any]] = [] if steps is None else list(steps)
        self.aborted = aborted
        self.rating_update = rating_update
        self.ai_thinking_since: str | None = None

    def player_descriptors(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for seat in range(self.num_players):
            if seat == self.human_seat:
                out.append(
                    {
                        "seat": seat,
                        "kind": "human",
                        "label": "You",
                        "model_id": None,
                        "rating": None,
                    }
                )
            else:
                m = self.seat_models[seat]
                rating = float(m.get("rating", 1500.0))
                out.append(
                    {
                        "seat": seat,
                        "kind": m["kind"],
                        "label": m["label"],
                        "model_id": m["id"],
                        "rating": rating,
                    }
                )
        return out

    def current_seat(self) -> int:
        return int(self.engine.current_player[0].item())

    def current_phase(self) -> int:
        return int(self.engine.phase[0].item())

    def ended(self) -> bool:
        return bool(self.engine.ended[0].item())

    def legal_actions(self) -> list[int]:
        mask = self.engine.legal_action_mask()
        return mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()

    def _record_and_apply(self, action: int) -> None:
        legal = self.legal_actions()
        seat_before = self.current_seat()
        phase_before = self.current_phase()
        action_t = torch.tensor([int(action)], dtype=torch.int64, device=self.engine.device)
        self.engine.apply(action_t)
        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "player": seat_before,
                "phase": phase_before,
                "action": int(action),
                "action_name": A.action_name(int(action)),
                "action_detail": V.action_detail(int(action)),
                "legal_actions": legal,
                "state_after": V.batched_to_snapshot(self.engine, 0),
            }
        )

    def replay_persisted_steps(self, persisted_steps: list[dict[str, Any]]) -> None:
        """Replay stored actions onto a fresh engine (used after cold load)."""
        self.steps = []
        for s in persisted_steps:
            self._record_and_apply(int(s["action"]))

    def step_ai_until_human_or_end(self, max_ai_steps: int = 200) -> None:
        steps_done = 0
        while not self.ended() and self.current_seat() != self.human_seat:
            policy = self.seat_policies[self.current_seat()]
            action_t = policy.choose(self.engine)
            action = int(action_t[0].item())
            self._record_and_apply(action)
            steps_done += 1
            if steps_done >= max_ai_steps:
                break

    def apply_human_action(self, action: int) -> None:
        if self.ended():
            raise ValueError("game already ended")
        if self.current_seat() != self.human_seat:
            raise ValueError("not human's turn")
        legal = self.legal_actions()
        if int(action) not in legal:
            raise ValueError(f"action {action} is not legal; legal={legal}")
        self._record_and_apply(action)

    def final_scores(self) -> list[dict[str, int]]:
        out: list[dict[str, int]] = []
        for p in range(self.num_players):
            out.append(
                {
                    "points": int(self.engine.points[0, p].item()),
                    "cards": int(self.engine.bonuses[0, p].sum().item()),
                    "nobles": int(self.engine.nobles_claimed[0, p].item()),
                }
            )
        return out

    def ranks(self) -> list[int]:
        scores = self.final_scores()
        keys = [(-s["points"], s["cards"]) for s in scores]
        order = sorted(range(self.num_players), key=lambda i: keys[i])
        ranks_list = [0] * self.num_players
        prev_key: tuple[int, int] | None = None
        prev_rank = 0
        for rank, seat in enumerate(order):
            if prev_key is not None and keys[seat] == prev_key:
                ranks_list[seat] = prev_rank
            else:
                ranks_list[seat] = rank
                prev_key = keys[seat]
                prev_rank = rank
        return ranks_list

    def view(self) -> dict[str, Any]:
        ended = self.ended() or self.aborted
        snap = V.batched_to_snapshot(self.engine, 0) if not self.steps else self.steps[-1]["state_after"]
        if ended:
            status = "ended"
            legal: list[int] | None = None
            winners: list[int] | None = None
            final = self.final_scores()
            if not self.aborted:
                rk = self.ranks()
                winners = [i for i, r in enumerate(rk) if r == 0]
        elif self.current_seat() == self.human_seat:
            status = "human_turn"
            legal = self.legal_actions()
            winners = None
            final = None
        else:
            status = "ai_thinking"
            legal = None
            winners = None
            final = None

        seat = self.human_seat
        redact = V.redact_snapshot_for_human
        redacted_steps: list[dict[str, Any]] = [
            {**s, "state_after": redact(s["state_after"], seat)} for s in self.steps
        ]

        init = self.initial_state
        view_snap = snap
        if not self.steps:
            view_snap = init

        result = {
            "game_id": self.game_id,
            "num_players": self.num_players,
            "human_seat": self.human_seat,
            "players": self.player_descriptors(),
            "initial_state": redact(init, seat),
            "cards": V.cards_table(),
            "nobles": V.nobles_table(),
            "steps": redacted_steps,
            "snapshot": redact(view_snap, seat),
            "legal_actions": legal,
            "current_player": self.current_seat() if not ended else None,
            "phase": self.current_phase() if not ended else None,
            "status": status,
            "winners": winners,
            "final_scores": final,
            "seed": self.seed,
            "rating_update": self.rating_update,
            "elo_update": self.rating_update,  # backward compat for old readers
            "aborted": self.aborted,
        }
        if status == "ai_thinking" and self.ai_thinking_since is not None:
            result["ai_thinking_since"] = self.ai_thinking_since
        return result
