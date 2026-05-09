"""Core play API used by ``play_server`` and future Lambda adapters."""

from __future__ import annotations

import logging
import pathlib
import threading
import time
import uuid
from typing import Any, Protocol, Iterable

from play import auth as AU
from play import human_rating as HE
from play import models as MD
from play import players as POL
from play import ratings as RT
from play.llm.policy import LLMBedrockPolicy
from play.llm.rate_limiter import LLMRateLimiter, RateLimitExceeded
from play.state import GameSession
from play.store import GameStatus

logger = logging.getLogger(__name__)


class PlayStore(Protocol):
    """Protocol defining the store interface accepted by PlayService.

    Both JsonPlayStore and DynamoPlayStore satisfy this protocol,
    allowing PlayService to work with either backend without code changes.

    Requirements: 2.7, 3.1
    """

    def load_game(self, game_id: str) -> dict[str, Any] | None: ...
    def save_game(self, record: dict[str, Any]) -> None: ...
    def list_games_for_user(
        self, username: str, status: Iterable[GameStatus] | None = None
    ) -> list[dict[str, Any]]: ...
    def load_user_rating_blob(self, username: str) -> dict[str, Any] | None: ...
    def save_user_rating_blob(self, username: str, data: dict[str, Any]) -> None: ...
    def list_all_user_rating_blobs(self) -> list[dict[str, Any]]: ...


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
    if kind == "llm_bedrock":
        # Do NOT cache LLM policies — each game gets its own instance
        return LLMBedrockPolicy(
            model_id=model["id"],
            bedrock_model_id=model["bedrock_model_id"],
            region="us-west-2",
            debug=True,
        )
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
        play_store: PlayStore,
        device: str = "cpu",
    ) -> None:
        self.workspace_root = workspace_root
        self.play_store = play_store
        self.device = device
        self._session_lock = threading.Lock()
        self._sessions: dict[str, GameSession] = {}
        self._llm_rate_limiter = LLMRateLimiter()

    def list_models(self) -> list[dict[str, Any]]:
        all_models = MD.discover_models(self.workspace_root)
        # Only expose built-ins + the single highest-rated net checkpoint.
        builtins = [m for m in all_models if m["kind"] != "net"]
        nets = [m for m in all_models if m["kind"] == "net"]
        if nets:
            best_net = max(nets, key=lambda m: m.get("rating", 0))
            builtins.append(best_net)
        # Enrich with unified ratings computed from all match data
        # (league eval games + human interactive games).
        ratings = RT.combined_ratings(self.workspace_root, self.play_store)
        # Load floating entities from league.json for game counts.
        floating = RT._load_floating_entities(self.workspace_root)
        for m in builtins:
            entity_id = MD.model_entity_id(m)
            lookup_id = RT._normalize_entity(entity_id)
            if lookup_id in ratings:
                m["rating"] = round(ratings[lookup_id])
            # Pull game count from floating_entities (e.g. bedrock_claude_sonnet)
            fe = floating.get(entity_id)
            if fe is not None:
                m["games"] = int(fe.get("games", m.get("games", 0)))
                if lookup_id not in ratings:
                    m["rating"] = round(float(fe.get("rating", m.get("rating", MD.DEFAULT_INITIAL_RATING))))
        return builtins

    def human_rating_store(self, identity: AU.UserIdentity) -> HE.HumanRatingStore:
        # For DynamoPlayStore, this method is overridden by the Lambda handler.
        # This default implementation works with JsonPlayStore (local file store).
        path = self.play_store.user_rating_path(identity.username)  # type: ignore[attr-defined]
        return HE.HumanRatingStore(path, human_entity=AU.human_entity_id(identity))

    def me(self, identity: AU.UserIdentity) -> dict[str, Any]:
        hr = self.human_rating_store(identity)
        hr.set_profile(identity.username)
        snap = hr.snapshot()
        # Compute calibrated combined rating from history.
        anchors = RT.bot_anchors_per_pc(self.workspace_root)
        combined = RT.combined_rating_for_blob(snap, anchors)
        rating = round(combined) if combined is not None else None
        # Only show rating if placed.
        if not snap["placed"]:
            rating = None
        return {
            "username": identity.username,
            "rating": rating,
            "games": snap["games"],
            "wins": snap["wins"],
            "placed": snap["placed"],
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
            "rating_update": session.rating_update,
        }

    def _finalize_rating_if_needed(
        self,
        session: GameSession,
        identity: AU.UserIdentity,
    ) -> None:
        if session.aborted or not session.ended() or session.rating_update is not None:
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
            rating = float(source.get("rating", MD.DEFAULT_INITIAL_RATING))
            if m["kind"] == "random":
                rating = MD.RANDOM_ANCHOR_RATING
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

        # Compute calibrated combined rating BEFORE recording the game.
        anchors = RT.bot_anchors_per_pc(self.workspace_root)
        snap_before = hr.snapshot()
        old_rating = RT.combined_rating_for_blob(snap_before, anchors)

        update = hr.record_game(
            opponents=opponents_payload,
            human_rank=ranks[session.human_seat],
            ranks=ranks,
            final_scores=session.final_scores(),
            human_seat=session.human_seat,
            seed=session.seed,
            meta={"game_id": session.game_id},
        )

        # Compute calibrated combined rating AFTER recording the game.
        snap_after = hr.snapshot()
        new_rating = RT.combined_rating_for_blob(snap_after, anchors)

        # Fall back to default if calibrated computation fails (e.g. no
        # opponents in the league manifest).
        if old_rating is None:
            old_rating = float(HE.DEFAULT_INITIAL_RATING)
        if new_rating is None:
            new_rating = float(HE.DEFAULT_INITIAL_RATING)

        session.rating_update = {
            "old_rating": old_rating,
            "new_rating": new_rating,
            "delta": new_rating - old_rating,
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
        is_ended = record.get("status") in ("completed", "aborted")
        seat_policies: dict[int, POL.PlayerPolicy] = {}
        for seat, m in seat_models.items():
            try:
                seat_policies[seat] = build_policy_cached(
                    m, num_sims=num_sims, seed=seed + seat, device=self.device
                )
            except FileNotFoundError:
                if is_ended:
                    # Game is already over — policy will never be called.
                    # Use a placeholder so the session can still be viewed.
                    seat_policies[seat] = POL.RandomPolicy(seed=seed + seat)
                else:
                    raise
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
        eu = record.get("rating_update") or record.get("elo_update")
        session.rating_update = dict(eu) if isinstance(eu, dict) else eu
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
        llm_count = 0
        for seat, model_id in opponents.items():
            m = MD.model_by_id(all_models, model_id)
            if m is None:
                raise ValueError(f"unknown model_id: {model_id!r}")
            if m["kind"] == "human":
                raise ValueError("human-vs-human games are not supported")

            if m["kind"] == "llm_bedrock":
                llm_count += 1
                if llm_count > 1:
                    raise ValueError(
                        "only one Claude Sonnet agent is allowed per game"
                    )
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

        # Rate-limit LLM game creation
        has_llm_opponent = any(
            m.get("kind") == "llm_bedrock" for m in seat_models.values()
        )
        if has_llm_opponent:
            try:
                self._llm_rate_limiter.check_and_record(identity.username)
            except RateLimitExceeded as exc:
                raise ValueError(str(exc)) from exc
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

        # After session creation, step AI if needed
        if session.current_seat() != human_seat:
            try:
                self._step_ai_sync(session, identity)
            except FileNotFoundError as e:
                logger.error(
                    "Game creation failed: ML checkpoint unavailable: %s", e,
                )
                raise ValueError(
                    "Cannot start game: the ML model checkpoint is unavailable. "
                    "It may have been removed during a deployment update."
                ) from e

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
            # Determine game result for completed games
            result: str | None = None
            if st == "completed":
                rating_update = row.get("rating_update") or row.get("elo_update")
                if rating_update and "per_opponent" in rating_update:
                    scores = [opp.get("score", 0.5) for opp in rating_update["per_opponent"]]
                    if all(s == 1.0 for s in scores):
                        result = "victory"
                    else:
                        result = "loss"
                else:
                    result = "completed"
            elif st == "aborted":
                result = "aborted"
            summaries.append(
                {
                    "game_id": gid,
                    "num_players": int(row.get("num_players", 0)),
                    "human_seat": int(row.get("human_seat", 0)),
                    "seed": int(row.get("seed", 0)),
                    "status": st,
                    "result": result,
                    "step_count": len(row.get("steps") or []),
                    "updated_at": row.get("updated_at"),
                }
            )
        return summaries

    def apply_human_action(
        self, identity: AU.UserIdentity, game_id: str, action: int
    ) -> GameSession:
        """Apply the human's action only. Does NOT step AI.

        Returns the session with the human move applied so the client can
        immediately see the updated game state (scores, tokens, action log).
        The client should then call step_ai() to advance AI seats.
        """
        session = self.get_or_load_session(game_id, identity)
        with session.lock:
            if session.aborted or session.ended():
                raise ValueError("game is not playable")
            session.apply_human_action(int(action))

            # If game ended from the human move alone, finalize immediately.
            if session.ended():
                self._finalize_rating_if_needed(session, identity)

            self._touch_session_save(identity, session, self._infer_status(session))
        return session

    def step_ai(self, identity: AU.UserIdentity, game_id: str) -> GameSession:
        """Step all AI seats synchronously until it's the human's turn or game ends.

        Should be called after apply_human_action when the game is not ended
        and it's not the human's turn. Safe to call when it's already the
        human's turn (no-op).
        """
        session = self.get_or_load_session(game_id, identity)
        with session.lock:
            if session.aborted or session.ended():
                self._touch_session_save(identity, session, self._infer_status(session))
                return session
            if session.current_seat() == session.human_seat:
                # Already human's turn — nothing to do.
                return session

            try:
                self._step_ai_sync(session, identity)
            except FileNotFoundError as e:
                # ML checkpoint unavailable mid-game — abort gracefully.
                logger.error(
                    "Aborting game %s: ML checkpoint unavailable: %s",
                    game_id, e,
                )
                session.aborted = True
                session.abort_reason = (
                    "Game cancelled: the ML model is no longer available. "
                    "This can happen when a new model is deployed during a game."
                )
                self._touch_session_save(identity, session, "aborted")
                return session

            self._finalize_rating_if_needed(session, identity)
            self._touch_session_save(identity, session, self._infer_status(session))
        return session

    def get_view(self, identity: AU.UserIdentity, game_id: str) -> dict[str, Any]:
        session = self.get_or_load_session(game_id, identity)
        with session.lock:
            return session.view()

    def _step_ai_sync(self, session: GameSession, identity: AU.UserIdentity) -> None:
        """Step all AI seats synchronously until it's the human's turn or game ends.

        For LLM policies, calls Bedrock inline (blocking the request).
        For non-LLM policies, uses the fast local choose() method.
        Handles discard/noble-pick sub-phases for LLM seats too.
        """
        import random as _random

        engine = session.engine

        while not session.ended() and session.current_seat() != session.human_seat:
            seat = session.current_seat()
            policy = session.seat_policies.get(seat)

            if isinstance(policy, LLMBedrockPolicy):
                # LLM policy — call synchronously (blocks the request)
                try:
                    action_tensor = policy.choose(engine)
                    session._record_and_apply(int(action_tensor[0].item()))
                except Exception as e:
                    # On any failure, fall back to random legal action
                    logger.warning(
                        "LLM call failed (game=%s, seat=%d): %s — using random fallback",
                        session.game_id, seat, e,
                    )
                    mask = engine.legal_action_mask()
                    legal_indices = mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
                    if isinstance(legal_indices, int):
                        legal_indices = [legal_indices]
                    pick = _random.choice(legal_indices)
                    session._record_and_apply(pick)
                    if session.steps:
                        session.steps[-1]["llm_fallback"] = {
                            "reason": "api_error",
                            "attempts": 1,
                            "raw_responses": [str(e)],
                            "latency_ms": 0,
                        }
            else:
                # Non-LLM policy — fast local call
                action_tensor = policy.choose(engine)
                session._record_and_apply(int(action_tensor[0].item()))
