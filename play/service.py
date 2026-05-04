"""Core play API used by ``play_server`` and future Lambda adapters."""

from __future__ import annotations

import pathlib
import threading
import time
import uuid
from typing import Any

from play import auth as AU
from play import human_elo as HE
from play import models as MD
from replay import players as POL
from play import ratings as RT
from play.state import GameSession
from play.store import GameStatus, JsonPlayStore


_GAME_SCHEMA_VERSION = 1

_NET_CACHE: dict[tuple[str, int, str], POL.PlayerPolicy] = {}
_NET_CACHE_LOCK = threading.Lock()


def build_policy_cached(
    model: dict[str, Any],
    num_sims: int,
    seed: int,
    device: str,
) -> POL.PlayerPolicy:
    kind = model["kind"]
    if kind == "random":
        return POL.RandomPolicy(seed=seed)
    if kind == "heuristic":
        return POL.HeuristicPolicy()
    if kind == "heuristic_opus":
        return POL.HeuristicOpusPolicy()
    if kind == "net":
        ckpt = str(model["ckpt"])
        key = (ckpt, num_sims, device)
        with _NET_CACHE_LOCK:
            cached = _NET_CACHE.get(key)
            if cached is not None:
                return cached
            policy: POL.PlayerPolicy = (
                POL.GreedyNetPolicy(ckpt, device=device)
                if num_sims <= 0
                else POL.NetPolicy(ckpt, num_sims=num_sims, device=device)
            )
            _NET_CACHE[key] = policy
            return policy
    raise ValueError(f"unsupported model kind: {kind!r}")


def _iso_utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _seat_models_from_json(raw: dict[str, Any]) -> dict[int, dict[str, Any]]:
    sm = raw.get("seat_models") or {}
    out: dict[int, dict[str, Any]] = {}
    for k, v in sm.items():
        out[int(k)] = dict(v)  # type: ignore[arg-type]
    return out


def seat_models_to_json(sm: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(k): dict(v) for k, v in sorted(sm.items())}


class PlayService:
    def __init__(
        self,
        workspace_root: pathlib.Path,
        play_store: JsonPlayStore,
        device: str = "cpu",
    ) -> None:
        self.workspace_root = workspace_root
        self.play_store = play_store
        self.device = device
        self._session_lock = threading.Lock()
        self._sessions: dict[str, GameSession] = {}

    def list_models(self) -> list[dict[str, Any]]:
        return MD.discover_models(self.workspace_root)

    def human_rating_store(self, identity: AU.UserIdentity) -> HE.HumanRatingStore:
        path = self.play_store.user_rating_path(identity.username)
        return HE.HumanRatingStore(path, human_entity=AU.human_entity_id(identity))

    def me(self, identity: AU.UserIdentity) -> dict[str, Any]:
        hr = self.human_rating_store(identity)
        hr.set_profile(identity.username)
        snap = hr.snapshot()
        return {
            "username": identity.username,
            "rating": snap["rating"],
            "elo": snap["elo"],
            "games": snap["games"],
            "rating_system": snap.get("rating_system"),
            "anchors": snap.get("anchors"),
            "history": snap.get("history"),
            "results": snap.get("results"),
        }

    def leaderboard(self) -> dict[str, Any]:
        return RT.leaderboard_response(self.workspace_root, self.play_store)

    def _record_from_session_views(self, session: GameSession) -> dict[str, Any]:
        return {
            "version": _GAME_SCHEMA_VERSION,
            "game_id": session.game_id,
            "num_players": session.num_players,
            "human_seat": session.human_seat,
            "seed": session.seed,
            "num_sims": session.saved_num_sims,
            "seat_models": seat_models_to_json(session.seat_models),
            "steps": list(session.steps),
            "initial_state": session.initial_state,
            "aborted": session.aborted,
            "elo_update": session.elo_update,
        }

    def _finalize_rating_if_needed(
        self,
        session: GameSession,
        identity: AU.UserIdentity,
    ) -> None:
        if session.aborted or not session.ended() or session.elo_update is not None:
            return
        ranks = session.ranks()
        latest_models = self.list_models()
        opponents_payload: list[dict[str, Any]] = []
        for seat in range(session.num_players):
            if seat == session.human_seat:
                continue
            m = session.seat_models[seat]
            latest = MD.model_by_id(latest_models, m["id"])
            source = latest if latest is not None else m
            rating = float(source.get("rating", source.get("elo", MD.HEURISTIC_ANCHOR_RATING)))
            if m["kind"] == "random":
                rating = MD.RANDOM_ANCHOR_RATING
            elif m["kind"] == "heuristic":
                rating = MD.HEURISTIC_ANCHOR_RATING
            elif m["kind"] == "heuristic_opus":
                rating = MD.HEURISTIC_OPUS_RATING
            opponents_payload.append(
                {
                    "seat": seat,
                    "entity_id": MD.model_entity_id(m),
                    "model_id": m["id"],
                    "label": m["label"],
                    "rating": rating,
                }
            )
        hr = self.human_rating_store(identity)
        update = hr.record_game(
            opponents=opponents_payload,
            human_rank=ranks[session.human_seat],
            ranks=ranks,
            final_scores=session.final_scores(),
            human_seat=session.human_seat,
            seed=session.seed,
            meta={"game_id": session.game_id},
        )
        session.elo_update = {
            "old_elo": update["old_rating"],
            "new_elo": update["new_rating"],
            "old_rating": update["old_rating"],
            "new_rating": update["new_rating"],
            "delta": update["delta"],
            "games": update["games"],
            "per_opponent": update["per_opponent"],
        }

    def _touch_session_save(
        self,
        identity: AU.UserIdentity,
        session: GameSession,
        status: GameStatus,
    ) -> None:
        merged = dict(self.play_store.load_game(session.game_id) or {})
        merged.update(self._record_from_session_views(session))
        merged["user_sub"] = identity.username
        merged["version"] = _GAME_SCHEMA_VERSION
        merged["status"] = status
        if "created_at" not in merged:
            merged["created_at"] = _iso_utc_now()
        self.play_store.save_game(merged)

    @staticmethod
    def _assert_owner(record: dict[str, Any], identity: AU.UserIdentity) -> None:
        if record.get("user_sub") != identity.username:
            raise PermissionError("game belongs to another user")

    def _session_from_record(self, record: dict[str, Any]) -> GameSession:
        num_players = int(record["num_players"])
        human_seat = int(record["human_seat"])
        seed = int(record["seed"])
        num_sims = int(record.get("num_sims", 64))
        seat_models = _seat_models_from_json(record)
        seat_policies: dict[int, POL.PlayerPolicy] = {}
        for seat, m in seat_models.items():
            seat_policies[seat] = build_policy_cached(
                m, num_sims=num_sims, seed=seed + seat, device=self.device
            )
        session = GameSession(
            game_id=str(record["game_id"]),
            num_players=num_players,
            human_seat=human_seat,
            seat_models=seat_models,
            seat_policies=seat_policies,
            seed=seed,
            device=self.device,
            initial_state_override=dict(record["initial_state"]),
        )
        session.saved_num_sims = num_sims
        session.replay_persisted_steps(list(record.get("steps") or []))
        session.aborted = bool(record.get("aborted", False))
        eu = record.get("elo_update")
        session.elo_update = dict(eu) if isinstance(eu, dict) else eu
        return session

    def get_or_load_session(self, game_id: str, identity: AU.UserIdentity) -> GameSession:
        record = self.play_store.load_game(game_id)
        if record is None:
            raise KeyError(game_id)
        self._assert_owner(record, identity)
        with self._session_lock:
            cached = self._sessions.get(game_id)
            if cached is not None and len(cached.steps) == len(record.get("steps") or []):
                return cached
        session = self._session_from_record(record)
        with self._session_lock:
            self._sessions[game_id] = session
        return session

    def validate_and_build_opponents(
        self,
        num_players: int,
        human_seat: int,
        opponents: dict[int, str],
        num_sims: int,
        policy_seed_basis: int,
    ) -> dict[int, dict[str, Any]]:
        if num_players not in (2, 3, 4):
            raise ValueError(f"num_players must be 2,3,4; got {num_players}")
        if not 0 <= human_seat < num_players:
            raise ValueError(
                f"human_seat {human_seat} out of range for num_players={num_players}"
            )
        expected_seats = {s for s in range(num_players) if s != human_seat}
        if set(opponents.keys()) != expected_seats:
            raise ValueError(
                f"opponents must specify exactly seats {sorted(expected_seats)}; "
                f"got {sorted(opponents.keys())}"
            )

        all_models = self.list_models()
        seat_models: dict[int, dict[str, Any]] = {}
        for seat, model_id in opponents.items():
            m = MD.model_by_id(all_models, model_id)
            if m is None:
                raise ValueError(f"unknown model_id: {model_id!r}")
            if m["kind"] == "human":
                raise ValueError("human-vs-human games are not supported")
            if num_players > 2 and m["kind"] == "net":
                raise ValueError("net checkpoints are only allowed in 2-player games")
            seat_models[seat] = m
            try:
                build_policy_cached(
                    m, num_sims=num_sims, seed=policy_seed_basis + seat, device=self.device
                )
            except FileNotFoundError as e:
                raise ValueError(
                    f"checkpoint for {model_id!r} no longer exists "
                    f"(may have been pruned by training): {e}"
                ) from e
        return seat_models

    def user_has_in_flight_game(self, identity: AU.UserIdentity) -> bool:
        rows = self.play_store.list_games_for_user(
            identity.username,
            status=("human_turn", "ai_thinking"),
        )
        return len(rows) > 0

    def create_game(self, identity: AU.UserIdentity, body: dict[str, Any]) -> GameSession:
        if self.user_has_in_flight_game(identity):
            raise ValueError(
                "you already have an in-flight game; finish it before starting another",
            )
        num_players = int(body.get("num_players", 2))
        human_seat = int(body.get("human_seat", 0))
        opponents_raw = body.get("opponents", {})
        opponents: dict[int, str] = {int(k): str(v) for k, v in opponents_raw.items()}
        seed_raw = body.get("seed")
        seed = int(seed_raw if seed_raw is not None else int(time.time()) & 0xFFFFFFFF)
        num_sims = int(body.get("num_sims", 64))

        seat_models = self.validate_and_build_opponents(
            num_players, human_seat, opponents, num_sims, policy_seed_basis=seed
        )
        seat_policies: dict[int, POL.PlayerPolicy] = {}
        for seat, m in seat_models.items():
            seat_policies[seat] = build_policy_cached(
                m, num_sims=num_sims, seed=seed + seat, device=self.device
            )

        game_id = uuid.uuid4().hex[:12]
        session = GameSession(
            game_id=game_id,
            num_players=num_players,
            human_seat=human_seat,
            seat_models=seat_models,
            seat_policies=seat_policies,
            seed=seed,
            device=self.device,
        )
        session.saved_num_sims = num_sims
        session.step_ai_until_human_or_end()
        record_full = {
            **self._record_from_session_views(session),
            "user_sub": identity.username,
            "status": self._infer_status(session),
            "created_at": _iso_utc_now(),
        }
        self.play_store.save_game(record_full)
        with self._session_lock:
            self._sessions[game_id] = session
        return session

    @staticmethod
    def _infer_status(session: GameSession) -> GameStatus:
        if session.aborted:
            return "aborted"
        if session.ended():
            return "completed"
        if session.current_seat() == session.human_seat:
            return "human_turn"
        return "ai_thinking"

    def list_games_summary(
        self,
        identity: AU.UserIdentity,
        status_filter: str | None,
    ) -> list[dict[str, Any]]:
        want: tuple[GameStatus, ...] | None
        if status_filter is None or status_filter == "all":
            want = None
        elif status_filter == "in_flight":
            want = ("human_turn", "ai_thinking")
        elif status_filter == "ended":
            want = ("completed", "aborted")
        elif status_filter == "active":
            want = ("human_turn", "ai_thinking")
        elif status_filter in ("completed", "aborted"):
            want = (status_filter,)  # type: ignore[assignment]
        else:
            raise ValueError(
                "invalid status query; use in_flight, ended, completed, aborted, active, all"
            )
        rows = self.play_store.list_games_for_user(identity.username, status=want)
        summaries: list[dict[str, Any]] = []
        for row in rows:
            gid = str(row["game_id"])
            st = row.get("status", "in_flight")
            summaries.append(
                {
                    "game_id": gid,
                    "num_players": int(row.get("num_players", 0)),
                    "human_seat": int(row.get("human_seat", 0)),
                    "seed": int(row.get("seed", 0)),
                    "status": st,
                    "step_count": len(row.get("steps") or []),
                    "updated_at": row.get("updated_at"),
                }
            )
        return summaries

    def apply_human_action(self, identity: AU.UserIdentity, game_id: str, action: int) -> GameSession:
        session = self.get_or_load_session(game_id, identity)
        with session.lock:
            if session.aborted or session.ended():
                raise ValueError("game is not playable")
            session.apply_human_action(int(action))
            session.step_ai_until_human_or_end()
            self._finalize_rating_if_needed(session, identity)
            self._touch_session_save(identity, session, self._infer_status(session))
        return session

    def get_view(self, identity: AU.UserIdentity, game_id: str) -> dict[str, Any]:
        session = self.get_or_load_session(game_id, identity)
        with session.lock:
            return session.view()
