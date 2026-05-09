"""AWS Lambda handler for the Splendor play API.

Adapts API Gateway HTTP API v2 events to PlayService method calls.
Uses a Docker container image Lambda to include PyTorch CPU.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import traceback
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_svc: Any = None
_dynamo_store: Any = None


def _get_dynamo_store():
    """Get or create the DynamoPlayStore singleton."""
    global _dynamo_store
    if _dynamo_store is not None:
        return _dynamo_store

    from play.dynamo_store import DynamoPlayStore

    games_table = os.environ.get("GAMES_TABLE", "SplendorGames")
    users_table = os.environ.get("USERS_TABLE", "SplendorUsers")
    region = os.environ.get("AWS_REGION", None)

    _dynamo_store = DynamoPlayStore(
        games_table_name=games_table,
        users_table_name=users_table,
        region=region,
    )
    return _dynamo_store


def _get_service():
    """Lazily initialize PlayService with DynamoPlayStore."""
    global _svc
    if _svc is not None:
        return _svc

    from play.service import PlayService

    store = _get_dynamo_store()

    workspace_root = pathlib.Path("/tmp/workspace")
    workspace_root.mkdir(parents=True, exist_ok=True)

    svc = PlayService(
        workspace_root=workspace_root,
        play_store=store,
        device="cpu",
    )

    # Override list_models to use S3-backed manifest.
    list_models_fn, list_models_for_display_fn = _make_manifest_list_models()
    svc.list_models = list_models_fn
    svc.list_models_for_display = list_models_for_display_fn

    # Override human_rating_store to use DynamoDB-backed implementation.
    svc.human_rating_store = _make_dynamo_human_rating_store(store)

    # Monkey-patch build_policy_cached to handle S3 checkpoint downloads
    import play.service as _play_service
    _original_build_policy = _play_service.build_policy_cached

    def _lambda_build_policy_cached(model, num_sims, seed, device):
        if model.get("kind") == "net" and not model.get("ckpt"):
            local_path = _ensure_checkpoint_local(model)
            if local_path:
                model = {**model, "ckpt": local_path}
            else:
                model_id = model.get("id", "unknown")
                logger.error(
                    "Checkpoint not found for model %s.", model_id,
                )
                raise FileNotFoundError(
                    f"ML checkpoint unavailable for model {model_id!r}. "
                    f"The checkpoint may not have been included in the "
                    f"container image build."
                )
        return _original_build_policy(model, num_sims, seed, device)

    _play_service.build_policy_cached = _lambda_build_policy_cached

    _svc = svc
    return _svc


def _load_manifest() -> list[dict[str, Any]]:
    """Load active league entries directly from league.json.

    Only entries with "active": true are returned as playable models.
    """
    league_dir = _LEAGUE_JSON_PATH.parent
    if not _LEAGUE_JSON_PATH.exists():
        return []

    with open(_LEAGUE_JSON_PATH) as f:
        data = json.load(f)

    manifest = []
    for e in data.get("entries", []):
        if not e.get("active"):
            continue

        ckpt_path = league_dir / e["path"]
        manifest.append({
            "run": "league",
            "tag": e.get("tag", f"idx{e['idx']}"),
            "idx": e["idx"],
            "rating": float(e.get("rating", 1500.0)),
            "hidden": e.get("hidden"),
            "arch": e.get("arch"),
            "local_path": str(ckpt_path),
        })

    if manifest:
        logger.info(f"Loaded {len(manifest)} active entries from league.json")
    return manifest


_league_data_cache: dict[str, Any] | None = None

# Path to bundled league.json (baked into container image)
_LEAGUE_JSON_PATH = pathlib.Path(__file__).resolve().parent.parent / "agent" / "runs" / "league" / "league.json"


def _load_league_data() -> dict[str, Any]:
    """Load league.json from bundled file (baked into container image).

    Cached after first load (league data doesn't change during a Lambda
    invocation lifecycle).
    """
    global _league_data_cache
    if _league_data_cache is not None:
        return _league_data_cache

    if _LEAGUE_JSON_PATH.exists():
        with open(_LEAGUE_JSON_PATH) as f:
            data = json.load(f)
        logger.info(f"Loaded bundled league data from {_LEAGUE_JSON_PATH}")
        _league_data_cache = data
        return data

    logger.warning(f"League data not found at {_LEAGUE_JSON_PATH}")
    _league_data_cache = {}
    return {}


_unified_ratings_cache: dict[str, float] | None = None


def _compute_unified_ratings() -> dict[str, float]:
    """Return unified ratings for all entities (bots + humans).

    Bot ratings come directly from league.json (pre-computed, no fitting).
    Cached for the Lambda container lifecycle.
    """
    global _unified_ratings_cache
    if _unified_ratings_cache is not None:
        return _unified_ratings_cache

    from play.ratings import _normalize_entity

    league_data = _load_league_data()
    unified: dict[str, float] = {"random": 1000.0}

    # Bot ratings: read directly from league.json (already computed by training pipeline)
    if league_data:
        for entry in league_data.get("entries", []):
            idx = int(entry.get("idx", -1))
            entity = f"ckpt:{idx}"
            rating = entry.get("rating")
            if rating is not None:
                unified[entity] = float(rating)
        for entity, fe in league_data.get("floating_entities", {}).items():
            rating = fe.get("rating")
            if rating is not None:
                unified[entity] = float(rating)

    # Human ratings: use stored per-user rating (already computed at game end)
    store = _get_dynamo_store()
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        rating = blob.get("rating")
        if rating is not None:
            unified[f"human:{uname}"] = float(rating)

    _unified_ratings_cache = unified
    return unified


def _enrich_models_with_ratings(models: list[dict[str, Any]]) -> None:
    """Enrich a list of model dicts with unified ratings and game counts in-place."""
    from play.models import model_entity_id, DEFAULT_INITIAL_RATING
    from play.ratings import _normalize_entity

    unified = _compute_unified_ratings()
    league_data = _load_league_data()
    floating: dict[str, dict] = league_data.get("floating_entities", {}) if league_data else {}

    for m in models:
        entity_id = model_entity_id(m)
        lookup_id = _normalize_entity(entity_id)
        if lookup_id in unified:
            m["rating"] = unified[lookup_id]
        # Pull game count from floating_entities (e.g. bedrock_claude_sonnet)
        fe = floating.get(entity_id)
        if fe is not None:
            m["games"] = int(fe.get("games", m.get("games", 0)))
            if lookup_id not in unified:
                m["rating"] = float(fe.get("rating", m.get("rating", DEFAULT_INITIAL_RATING)))


def _enrich_game_view_ratings(view: dict[str, Any]) -> None:
    """Enrich player ratings in a game view dict with unified ratings.

    Fixes ratings for games that were persisted with hardcoded defaults
    (e.g. bedrock_claude_sonnet at 1500).
    """
    from play.ratings import _normalize_entity

    unified = _compute_unified_ratings()
    players = view.get("players")
    if not players:
        return
    for p in players:
        if p.get("kind") == "human":
            continue
        model_id = p.get("model_id", "")
        bot_kind = p.get("kind", "")
        # Derive entity_id the same way model_entity_id does
        if bot_kind in ("random", "heuristic", "heuristic_opus"):
            entity_id = bot_kind
        else:
            entity_id = model_id
        lookup_id = _normalize_entity(entity_id)
        if lookup_id in unified:
            p["rating"] = unified[lookup_id]


def _make_manifest_list_models():
    """Create list_models (all agents) and list_models_for_display (best only).

    list_models() returns built-ins + ALL net models (used for game creation
    and stale-game checking so games against non-best agents aren't deleted).

    list_models_for_display() returns built-ins + only the single highest-rated
    net model (used for the /api/agents endpoint and leaderboard).
    """
    from play.models import _builtins as _models_builtins

    _cached_all: list[dict[str, Any]] | None = None
    _cached_display: list[dict[str, Any]] | None = None

    def _net_models() -> list[dict[str, Any]]:
        from play.models import DEFAULT_INITIAL_RATING as _HAR
        net_models: list[dict[str, Any]] = []
        for entry in _load_manifest():
            run = entry.get("run", "unknown")
            idx = entry.get("idx", 0)
            tag = entry.get("tag", f"idx{idx}")
            rating = float(entry.get("rating", _HAR))
            net_models.append({
                "id": f"net:{run}:{idx}",
                "label": "RL Trained Bot",
                "kind": "net",
                "run": run, "tag": tag, "idx": idx,
                "rating": rating,
                "games": int(entry.get("games", 0)),
                "hidden": entry.get("hidden"), "arch": entry.get("arch"),
                "description": "Trained neural network that learned to play through self-play",
                "_local_path": entry.get("local_path", ""),
                "_s3_key": entry.get("s3_key", ""),
            })
        return net_models

    def list_models() -> list[dict[str, Any]]:
        """Return ALL models (built-ins + all net models)."""
        nonlocal _cached_all
        if _cached_all is not None:
            return _cached_all
        models = _models_builtins()
        models.extend(_net_models())
        _enrich_models_with_ratings(models)
        _cached_all = models
        return models

    def list_models_for_display() -> list[dict[str, Any]]:
        """Return built-ins + only the best net model (for frontend/leaderboard)."""
        nonlocal _cached_display
        if _cached_display is not None:
            return _cached_display
        models = _models_builtins()
        nets = _net_models()
        if nets:
            best_net = max(nets, key=lambda m: m["rating"])
            models.append(best_net)
        _enrich_models_with_ratings(models)
        _cached_display = models
        return models

    return list_models, list_models_for_display


def _ensure_checkpoint_local(model: dict[str, Any]) -> str | None:
    """Resolve a net model's checkpoint to a local path.

    Checkpoints are baked into the container image (or on disk for local dev).
    Returns the local path, or None if the checkpoint file doesn't exist.
    """
    if model.get("kind") != "net":
        return None

    # Check for explicit local_path from manifest
    local_path_str = model.get("_local_path", "")
    if local_path_str:
        local_path = pathlib.Path(local_path_str)
        if local_path.exists():
            return str(local_path)

    logger.warning(f"Checkpoint not found for model {model.get('id', 'unknown')}")
    return None


def _make_dynamo_human_rating_store(store):
    """Create a human_rating_store function that uses DynamoDB."""
    from play import auth as AU
    from play import human_rating as HE

    def human_rating_store(identity: AU.UserIdentity) -> HE.HumanRatingStore:
        """Create a HumanRatingStore backed by /tmp with DynamoDB sync."""
        username = identity.username
        human_entity = AU.human_entity_id(identity)

        # Load existing data from DynamoDB
        blob = store.load_user_rating_blob(username)

        # Write to /tmp so HumanRatingStore can read it
        tmp_dir = pathlib.Path("/tmp/users")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        user_path = tmp_dir / f"{username}.json"

        if blob is not None:
            with open(user_path, "w") as f:
                json.dump(blob, f)
        elif user_path.exists():
            # Remove stale file from previous invocation
            user_path.unlink()

        # Create the store (it will read from the file or create fresh)
        hr = HE.HumanRatingStore(user_path, human_entity=human_entity)

        # Monkey-patch _save_locked to also persist to DynamoDB
        original_save = hr._save_locked

        def _save_to_dynamo():
            original_save()
            store.save_user_rating_blob(username, hr._data)

        hr._save_locked = _save_to_dynamo
        return hr

    return human_rating_store


def _lambda_me(identity) -> dict[str, Any]:
    """Handle /me endpoint using DynamoDB directly."""
    from play.ratings import bot_anchors_per_pc, combined_rating_for_blob

    svc = _get_service()

    hr = svc.human_rating_store(identity)
    hr.set_profile(identity.username)
    snap = hr.snapshot()

    # Compute calibrated combined rating from history.
    anchors = bot_anchors_per_pc(svc.workspace_root)
    combined = combined_rating_for_blob(snap, anchors)
    rating = round(combined) if combined is not None else None
    if not snap["placed"]:
        rating = None

    return {
        "username": identity.username,
        "rating": rating,
        "games": snap["games"],
        "wins": snap["wins"],
        "placed": snap["placed"],
    }


def _lambda_leaderboard() -> dict[str, Any]:
    """Handle /leaderboard endpoint using DynamoDB directly.

    Bot ratings come directly from league.json (pre-computed, no fitting).
    Human ratings are computed via fast single-variable bisection per PC.
    """
    from play import models as MD
    from play.ratings import (
        _normalize_entity, _bot_ratings_from_league, _compute_human_ratings,
        human_leaderboard_rows, PLAYER_COUNTS, RANDOM_ANCHOR_RATING,
    )

    store = _get_dynamo_store()
    svc = _get_service()
    all_models = svc.list_models_for_display()

    # Load league data from S3 (cached in module-level dict).
    league_data = _load_league_data()

    # Extract bot ratings directly from league.json (no LBFGS fitting)
    bot_data = _bot_ratings_from_league(league_data) if league_data else {}
    floating: dict[str, dict] = league_data.get("floating_entities", {}) if league_data else {}

    # Agent rows — use pre-computed ratings from league.json
    agents: list[dict[str, Any]] = []
    for m in all_models:
        label = str(m["label"])
        if m["kind"] == "net":
            label = "RL Trained Bot"
        entity_id = MD.model_entity_id(m)
        lookup_id = _normalize_entity(entity_id)

        # Get ratings from league data (already calibrated per-PC)
        entity_data = bot_data.get(lookup_id, {})

        # For the random bot (anchor), rating is 1000 at all player counts.
        if m["kind"] == "random":
            rating_2p = RANDOM_ANCHOR_RATING
            rating_3p = RANDOM_ANCHOR_RATING
            rating_4p = RANDOM_ANCHOR_RATING
        else:
            rating_2p = entity_data.get("rating_2p")
            rating_3p = entity_data.get("rating_3p")
            rating_4p = entity_data.get("rating_4p")

        # Combined rating: average of calibrated per-PC values
        calibrated_vals = [v for v in (rating_2p, rating_3p, rating_4p) if v is not None]
        if calibrated_vals:
            rating = sum(calibrated_vals) / len(calibrated_vals)
        elif "rating" in entity_data:
            rating = entity_data["rating"]
        else:
            rating = float(m.get("rating", 0.0))

        # Game count from floating entities or model definition
        games = int(m.get("games", 0))
        fe = floating.get(entity_id)
        if fe is not None:
            games = int(fe.get("games", games))
            if lookup_id not in bot_data:
                rating = float(fe.get("rating", rating))

        row: dict[str, Any] = {
            "kind": "agent",
            "entity_id": entity_id,
            "label": label,
            "model_id": str(m["id"]),
            "bot_kind": str(m["kind"]),
            "rating": round(rating),
            "games": games,
            "description": m.get("description"),
        }
        if rating_2p is not None:
            row["rating_2p"] = round(rating_2p)
        if rating_3p is not None:
            row["rating_3p"] = round(rating_3p)
        if rating_4p is not None:
            row["rating_4p"] = round(rating_4p)
        agents.append(row)

    # Human rows — compute per-PC ratings via fast bisection
    # Build per-PC anchor maps using RAW (uncalibrated) ratings
    from play.ratings import CALIBRATION_SCALE
    bot_anchors_per_pc: dict[int, dict[str, float]] = {pc: {} for pc in PLAYER_COUNTS}
    for entity, data in bot_data.items():
        for pc in PLAYER_COUNTS:
            raw = data.get(f"raw_{pc}p")
            if raw is not None:
                bot_anchors_per_pc[pc][entity] = raw
    for pc in PLAYER_COUNTS:
        bot_anchors_per_pc[pc]["random"] = RANDOM_ANCHOR_RATING

    from play.ratings import PLACEMENT_WINS_REQUIRED, DEFAULT_INITIAL_RATING as _DEFAULT_RATING
    humans: list[dict[str, Any]] = []
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        wins = int(blob.get("wins", 0))
        if wins < PLACEMENT_WINS_REQUIRED:
            continue
        games = int(blob.get("games", 0))

        human_data = _compute_human_ratings(blob, bot_anchors_per_pc)
        rating_val = human_data.get("rating")
        if rating_val is None:
            rating_val = float(blob.get("rating", _DEFAULT_RATING))

        row = {
            "kind": "human",
            "entity_id": f"human:{uname}",
            "label": uname,
            "username": uname,
            "rating": round(rating_val),
            "games": games,
        }
        for pc in PLAYER_COUNTS:
            val = human_data.get(f"rating_{pc}p")
            if val is not None:
                row[f"rating_{pc}p"] = round(val)
        humans.append(row)

    combined = humans + agents
    combined.sort(key=lambda r: float(r.get("rating", 0.0)), reverse=True)
    return {"entities": combined}


def _parse_event(event: dict[str, Any]) -> tuple[str, str, dict[str, str], dict[str, str], str]:
    """Parse API Gateway HTTP API v2 event."""
    request_context = event.get("requestContext", {})
    http_info = request_context.get("http", {})

    method = http_info.get("method", event.get("httpMethod", "GET")).upper()
    path = event.get("rawPath", event.get("path", "/"))
    headers = event.get("headers", {}) or {}
    query_params = event.get("queryStringParameters", {}) or {}
    body = event.get("body", "") or ""
    if event.get("isBase64Encoded", False):
        import base64
        body = base64.b64decode(body).decode("utf-8")

    return method, path, headers, query_params, body


def _parse_json_body(body: str) -> dict[str, Any]:
    if not body or not body.strip():
        return {}
    obj = json.loads(body)
    if not isinstance(obj, dict):
        raise ValueError("JSON body must be an object")
    return obj


def _json_response(status_code: int, body: Any) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _extract_identity(headers: dict[str, str]):
    from play import auth as AU
    return AU.identity_from_headers(headers)


def _get_available_model_ids() -> set[str]:
    """Get the set of currently available model IDs."""
    svc = _get_service()
    return {m["id"] for m in svc.list_models()}


def _dispatch(method: str, path: str, headers: dict[str, str], query_params: dict[str, str], body: str) -> dict[str, Any]:
    # Health — no heavy imports.
    if method == "GET" and path in ("/api/health", "/api/play/health"):
        return _json_response(200, {"ok": True})

    if method == "OPTIONS":
        return _json_response(200, {"ok": True})

    # Strip /api/play/ prefix to normalize paths
    if path.startswith("/api/play/"):
        path = "/api/" + path[len("/api/play/"):]

    if method == "GET":
        if path == "/api/agents":
            svc = _get_service()
            return _json_response(200, svc.list_models_for_display())

        if path == "/api/leaderboard":
            return _json_response(200, _lambda_leaderboard())

        identity = _extract_identity(headers)

        if path == "/api/me":
            return _json_response(200, _lambda_me(identity))

        svc = _get_service()

        if path == "/api/games":
            status_filter = query_params.get("status")
            games = svc.list_games_summary(identity, status_filter)
            # Clean up stale in-flight games whose opponent model is no longer available
            available_ids = _get_available_model_ids()
            store = _get_dynamo_store()
            clean_games: list[dict[str, Any]] = []
            for g in games:
                if g.get("status") in ("human_turn", "ai_thinking"):
                    # Load full record to check seat_models
                    record = store.load_game(g["game_id"])
                    if record:
                        seat_models = record.get("seat_models", {})
                        stale = False
                        for seat_key, model_info in seat_models.items():
                            model_id = model_info.get("id", "")
                            if model_info.get("kind") == "net" and model_id not in available_ids:
                                stale = True
                                break
                        if stale:
                            store.delete_game(g["game_id"])
                            continue
                clean_games.append(g)
            return _json_response(200, clean_games)

        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "games":
            # Check if the opponent model is still available before resuming
            game_id = parts[2]
            store = _get_dynamo_store()
            record = store.load_game(game_id)
            if record is not None:
                status = record.get("status", "")
                if status in ("human_turn", "ai_thinking"):
                    available_ids = _get_available_model_ids()
                    seat_models = record.get("seat_models", {})
                    for seat_key, model_info in seat_models.items():
                        model_id = model_info.get("id", "")
                        if model_info.get("kind") == "net" and model_id not in available_ids:
                            store.delete_game(game_id)
                            raise KeyError(game_id)
            view = svc.get_view(identity, game_id)
            _enrich_game_view_ratings(view)
            return _json_response(200, view)

        return _json_response(404, {"error": f"not found: {path}"})

    if method == "POST":
        identity = _extract_identity(headers)
        parsed_body = _parse_json_body(body)
        svc = _get_service()

        if path == "/api/games":
            session = svc.create_game(identity, parsed_body)
            with session.lock:
                view = session.view()
            _enrich_game_view_ratings(view)
            return _json_response(201, view)

        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "games" and parts[3] == "action":
            action = parsed_body.get("action")
            if not isinstance(action, int):
                raise ValueError("body must include integer 'action'")
            session = svc.apply_human_action(identity, parts[2], action)
            with session.lock:
                view = session.view()
            _enrich_game_view_ratings(view)
            return _json_response(200, view)

        if len(parts) == 4 and parts[0] == "api" and parts[1] == "games" and parts[3] == "step-ai":
            session = svc.step_ai(identity, parts[2])
            with session.lock:
                view = session.view()
            _enrich_game_view_ratings(view)
            return _json_response(200, view)

        return _json_response(404, {"error": f"not found: {path}"})

    if method == "DELETE":
        identity = _extract_identity(headers)
        store = _get_dynamo_store()

        if path == "/api/user":
            deleted_count = store.delete_user_data(identity.username)
            return _json_response(200, {"ok": True, "deleted_games": deleted_count})

        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "games":
            game_id = parts[2]
            record = store.load_game(game_id)
            if record is None:
                return _json_response(404, {"error": "game not found"})
            if record.get("user_sub") != identity.username:
                return _json_response(403, {"error": "game belongs to another user"})
            status = record.get("status", "")
            if status not in ("human_turn", "ai_thinking"):
                return _json_response(400, {"error": "cannot delete a completed/aborted game"})
            store.delete_game(game_id)
            return _json_response(200, {"ok": True})

        return _json_response(404, {"error": f"not found: {path}"})

    return _json_response(404, {"error": f"not found: {path}"})


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point."""
    try:
        method, path, headers, query_params, body = _parse_event(event)
        return _dispatch(method, path, headers, query_params, body)
    except ValueError as e:
        return _json_response(400, {"error": str(e)})
    except KeyError as e:
        return _json_response(404, {"error": f"not found: {e}"})
    except PermissionError as e:
        return _json_response(403, {"error": str(e)})
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return _json_response(500, {"error": "internal server error"})
