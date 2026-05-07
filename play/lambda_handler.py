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


def _make_manifest_list_models():
    """Create list_models (all agents) and list_models_for_display (best only).

    list_models() returns built-ins + ALL net models (used for game creation
    and stale-game checking so games against non-best agents aren't deleted).

    list_models_for_display() returns built-ins + only the single highest-rated
    net model (used for the /api/agents endpoint and leaderboard).
    """
    from play.models import (
        HEURISTIC_ANCHOR_RATING,
        HEURISTIC_OPUS_RATING,
        RANDOM_ANCHOR_RATING,
    )

    _cached_all: list[dict[str, Any]] | None = None
    _cached_display: list[dict[str, Any]] | None = None

    def _builtins() -> list[dict[str, Any]]:
        return [
            {
                "id": "heuristic_opus",
                "label": "Heuristic Opus Bot",
                "kind": "heuristic_opus",
                "run": None, "tag": None, "ckpt": None,
                "rating": HEURISTIC_OPUS_RATING,
                "games": 0, "hidden": None, "arch": None,
                "score_hint": None, "winrate_vs_heuristic": None,
            },
            {
                "id": "heuristic",
                "label": "Heuristic Bot",
                "kind": "heuristic",
                "run": None, "tag": None, "ckpt": None,
                "rating": HEURISTIC_ANCHOR_RATING,
                "games": 0, "hidden": None, "arch": None,
                "score_hint": None, "winrate_vs_heuristic": None,
            },
            {
                "id": "random",
                "label": "Random Bot",
                "kind": "random",
                "run": None, "tag": None, "ckpt": None,
                "rating": RANDOM_ANCHOR_RATING,
                "games": 0, "hidden": None, "arch": None,
                "score_hint": None, "winrate_vs_heuristic": None,
            },
        ]

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
        models = _builtins()
        models.extend(_net_models())
        _cached_all = models
        return models

    def list_models_for_display() -> list[dict[str, Any]]:
        """Return built-ins + only the best net model (for frontend/leaderboard)."""
        nonlocal _cached_display
        if _cached_display is not None:
            return _cached_display
        models = _builtins()
        nets = _net_models()
        if nets:
            best_net = max(nets, key=lambda m: m["rating"])
            models.append(best_net)
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
    from play import human_elo as HE

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
    from play import human_elo as HE

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
    """Handle /leaderboard endpoint using DynamoDB directly."""
    from play import human_elo as HE
    from play import models as MD

    store = _get_dynamo_store()
    svc = _get_service()
    all_models = svc.list_models_for_display()

    # Agent rows from manifest models
    agents: list[dict[str, Any]] = []
    for m in all_models:
        agents.append({
            "kind": "agent",
            "entity_id": MD.model_entity_id(m),
            "label": str(m["label"]),
            "model_id": str(m["id"]),
            "bot_kind": str(m["kind"]),
            "rating": float(m.get("rating", 0.0)),
            "games": int(m.get("games", 0)),
        })

    # Human rows from DynamoDB
    humans: list[dict[str, Any]] = []
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        wins = int(blob.get("wins", 0))
        if wins < HE.PLACEMENT_WINS_REQUIRED:
            continue
        rating = float(blob.get("rating", blob.get("elo", HE.DEFAULT_INITIAL_RATING)))
        games = int(blob.get("games", 0))
        humans.append({
            "kind": "human",
            "entity_id": f"human:{uname}",
            "label": uname,
            "username": uname,
            "rating": rating,
            "games": games,
        })

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
            return _json_response(200, svc.get_view(identity, game_id))

        return _json_response(404, {"error": f"not found: {path}"})

    if method == "POST":
        identity = _extract_identity(headers)
        parsed_body = _parse_json_body(body)
        svc = _get_service()

        if path == "/api/games":
            session = svc.create_game(identity, parsed_body)
            with session.lock:
                return _json_response(201, session.view())

        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "games" and parts[3] == "action":
            action = parsed_body.get("action")
            if not isinstance(action, int):
                raise ValueError("body must include integer 'action'")
            session = svc.apply_human_action(identity, parts[2], action)
            with session.lock:
                return _json_response(200, session.view())

        if len(parts) == 4 and parts[0] == "api" and parts[1] == "games" and parts[3] == "step-ai":
            session = svc.step_ai(identity, parts[2])
            with session.lock:
                return _json_response(200, session.view())

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
