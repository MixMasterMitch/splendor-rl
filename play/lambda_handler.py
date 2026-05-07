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
                # Checkpoint unavailable (e.g., viewing a completed game
                # whose model was removed). Return a dummy policy that
                # won't be called for completed games.
                from replay import players as POL
                return POL.RandomPolicy(seed=seed)
        return _original_build_policy(model, num_sims, seed, device)

    _play_service.build_policy_cached = _lambda_build_policy_cached

    _svc = svc
    return _svc


def _load_manifest() -> list[dict[str, Any]]:
    """Load deployed models manifest from S3, env var, or bundled file."""
    # Try loading from S3 first
    bucket = os.environ.get("MODELS_BUCKET", "")
    manifest_key = os.environ.get("MODELS_MANIFEST", "")
    if bucket and manifest_key:
        try:
            import boto3
            s3 = boto3.client("s3")
            resp = s3.get_object(Bucket=bucket, Key=manifest_key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            # Inject s3_bucket into each entry
            for entry in data:
                entry.setdefault("s3_bucket", bucket)
            logger.info(f"Loaded manifest from s3://{bucket}/{manifest_key} with {len(data)} entries")
            return data
        except Exception as e:
            logger.warning(f"Failed to load manifest from S3: {e}")

    # Fallback: env var with JSON content
    manifest_env = os.environ.get("MODELS_MANIFEST_JSON")
    if manifest_env:
        return json.loads(manifest_env)

    # Fallback: bundled file
    manifest_path = pathlib.Path(__file__).parent / "models_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)

    return []


_league_data_cache: dict[str, Any] | None = None


def _load_league_data() -> dict[str, Any]:
    """Load league.json from S3 for unified rating computation.

    Cached after first load (league data doesn't change during a Lambda
    invocation lifecycle).
    """
    global _league_data_cache
    if _league_data_cache is not None:
        return _league_data_cache

    bucket = os.environ.get("MODELS_BUCKET", "")
    league_key = os.environ.get("LEAGUE_DATA_KEY", "")
    if bucket and league_key:
        try:
            import boto3
            s3 = boto3.client("s3")
            resp = s3.get_object(Bucket=bucket, Key=league_key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            logger.info(f"Loaded league data from s3://{bucket}/{league_key}")
            _league_data_cache = data
            return data
        except Exception as e:
            logger.warning(f"Failed to load league data from S3: {e}")

    _league_data_cache = {}
    return {}


_unified_ratings_cache: dict[str, float] | None = None


def _compute_unified_ratings() -> dict[str, float]:
    """Compute unified ratings from league data + human game results.

    Cached for the Lambda container lifecycle. Used to enrich model ratings
    across all endpoints (agents list, game views, etc.).
    """
    global _unified_ratings_cache
    if _unified_ratings_cache is not None:
        return _unified_ratings_cache

    from play.ratings import _add_eval_results, _normalize_entity
    from agent.train import ranking as R

    league_data = _load_league_data()
    anchors: dict[str, float] = dict(R.DEFAULT_ANCHORS)
    all_results: list[dict[str, Any]] = []

    if league_data:
        all_results.extend(league_data.get("results", []))
        for entry in league_data.get("entries", []):
            idx = int(entry.get("idx", -1))
            entity = f"ckpt:{idx}"
            _add_eval_results(all_results, entity, entry)

    store = _get_dynamo_store()
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        for r in blob.get("results", []):
            normalized = dict(r)
            normalized["a"] = _normalize_entity(normalized["a"])
            normalized["b"] = _normalize_entity(normalized["b"])
            all_results.append(normalized)

    unified: dict[str, float] = dict(anchors)
    if all_results:
        initial: dict[str, float] = {}
        if league_data:
            for entry in league_data.get("entries", []):
                idx = int(entry.get("idx", -1))
                entity = f"ckpt:{idx}"
                rating = entry.get("rating")
                if rating is not None:
                    initial[entity] = float(rating)
        try:
            unified = R.fit_anchored_ratings(
                all_results, anchors=anchors, initial=initial
            )
        except Exception:
            logger.warning("Failed to compute unified ratings", exc_info=True)

    _unified_ratings_cache = unified
    return unified


def _enrich_models_with_ratings(models: list[dict[str, Any]]) -> None:
    """Enrich a list of model dicts with unified ratings in-place."""
    from play.models import model_entity_id
    from play.ratings import _normalize_entity

    unified = _compute_unified_ratings()
    for m in models:
        entity_id = model_entity_id(m)
        lookup_id = _normalize_entity(entity_id)
        if lookup_id in unified:
            m["rating"] = unified[lookup_id]


def _enrich_game_view_ratings(view: dict[str, Any]) -> None:
    """Enrich player ratings in a game view dict with unified ratings.

    Fixes ratings for games that were persisted with hardcoded defaults
    (e.g. bedrock_claude_sonnet at 2500).
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
        from play.models import HEURISTIC_ANCHOR_RATING as _HAR
        net_models: list[dict[str, Any]] = []
        for entry in _load_manifest():
            run = entry.get("run", "unknown")
            idx = entry.get("idx", 0)
            tag = entry.get("tag", f"idx{idx}")
            rating = float(entry.get("rating", _HAR))
            net_models.append({
                "id": f"net:{run}:{idx}",
                "label": "ML Bot",
                "kind": "net",
                "run": run, "tag": tag, "idx": idx,
                "ckpt": None,
                "rating": rating,
                "games": int(entry.get("games", 0)),
                "hidden": entry.get("hidden"), "arch": entry.get("arch"),
                "score_hint": entry.get("score_hint"),
                "winrate_vs_heuristic": entry.get("winrate_vs_heuristic"),
                "_s3_bucket": entry.get("s3_bucket", ""),
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
    """Download a net model's checkpoint from S3 to /tmp if needed.

    Returns the local path, or None if download fails or model is unavailable.
    """
    if model.get("kind") != "net":
        return None

    s3_bucket = model.get("_s3_bucket", "")
    s3_key = model.get("_s3_key", "")
    if not s3_bucket or not s3_key:
        # Fallback: try MODELS_BUCKET env var
        s3_bucket = os.environ.get("MODELS_BUCKET", "")
        if not s3_bucket:
            return None

    local_path = pathlib.Path(f"/tmp/checkpoints/{s3_key}")
    if local_path.exists():
        return str(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import boto3
        s3 = boto3.client("s3")
        logger.info(f"Downloading checkpoint s3://{s3_bucket}/{s3_key} -> {local_path}")
        s3.download_file(s3_bucket, s3_key, str(local_path))
        return str(local_path)
    except Exception as e:
        logger.warning(f"Failed to download checkpoint s3://{s3_bucket}/{s3_key}: {e}")
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
    from play import auth as AU
    from play import human_rating as HE

    store = _get_dynamo_store()
    svc = _get_service()

    hr = svc.human_rating_store(identity)
    hr.set_profile(identity.username)
    snap = hr.snapshot()
    # Strip internal "ties" field from results for the API response.
    results = [
        {k: v for k, v in r.items() if k != "ties"}
        for r in snap.get("results", [])
    ]
    return {
        "username": identity.username,
        "rating": snap["rating"],
        "games": snap["games"],
        "wins": snap["wins"],
        "placed": snap["placed"],
        "rating_system": snap.get("rating_system"),
        "anchors": snap.get("anchors"),
        "history": snap.get("history"),
        "results": results,
    }


def _lambda_leaderboard() -> dict[str, Any]:
    """Handle /leaderboard endpoint using DynamoDB directly.

    Loads league.json from S3 and combines it with user rating blobs to
    compute unified ratings — matching the local server's combined_ratings()
    logic exactly.
    """
    from play import models as MD
    from play.ratings import _add_eval_results, _normalize_entity, human_leaderboard_rows

    store = _get_dynamo_store()
    svc = _get_service()
    all_models = svc.list_models_for_display()

    # Load league data from S3 (cached in module-level dict).
    league_data = _load_league_data()

    # Build unified results pool — same as combined_ratings() in ratings.py.
    from agent.train import ranking as R
    anchors: dict[str, float] = dict(R.DEFAULT_ANCHORS)
    all_results: list[dict[str, Any]] = []

    # 1. League results (checkpoint vs checkpoint, checkpoint vs bots)
    if league_data:
        all_results.extend(league_data.get("results", []))
        for entry in league_data.get("entries", []):
            idx = int(entry.get("idx", -1))
            entity = f"ckpt:{idx}"
            _add_eval_results(all_results, entity, entry)

    # 2. Human game results from DynamoDB user rating blobs
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        for r in blob.get("results", []):
            normalized = dict(r)
            normalized["a"] = _normalize_entity(normalized["a"])
            normalized["b"] = _normalize_entity(normalized["b"])
            all_results.append(normalized)

    # Compute unified ratings.
    unified_ratings: dict[str, float] = dict(anchors)
    if all_results:
        # Collect initial guesses from league entries for faster convergence.
        initial: dict[str, float] = {}
        if league_data:
            for entry in league_data.get("entries", []):
                idx = int(entry.get("idx", -1))
                entity = f"ckpt:{idx}"
                rating = entry.get("rating")
                if rating is not None:
                    initial[entity] = float(rating)
        try:
            unified_ratings = R.fit_anchored_ratings(
                all_results, anchors=anchors, initial=initial
            )
        except Exception:
            logger.warning("Failed to compute unified ratings", exc_info=True)

    # Agent rows — use unified ratings when available, else static default.
    agents: list[dict[str, Any]] = []
    for m in all_models:
        label = str(m["label"])
        if m["kind"] == "net":
            label = "ML Bot"
        entity_id = MD.model_entity_id(m)
        lookup_id = _normalize_entity(entity_id)
        rating = unified_ratings.get(lookup_id, float(m.get("rating", 0.0)))
        agents.append({
            "kind": "agent",
            "entity_id": entity_id,
            "label": label,
            "model_id": str(m["id"]),
            "bot_kind": str(m["kind"]),
            "rating": rating,
            "games": int(m.get("games", 0)),
        })

    # Human rows — reuse the same logic as the local server.
    humans = human_leaderboard_rows(store)

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
