"""Leaderboard rows from rated agents plus persisted human rating files.

Ratings are computed by combining ALL available match data — league eval
games (checkpoints vs bots) and human interactive games — into a single
Bradley-Terry MLE fit anchored to random=1000 and heuristic=2500.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from play import human_rating as HE
from play import models as MD
from play.store import JsonPlayStore

logger = logging.getLogger(__name__)


def _add_eval_results(
    results: list[dict[str, Any]], entity: str, entry: dict[str, Any]
) -> None:
    """Synthesize pairwise results from per-entry eval winrate stats.

    League entries store rank_winrate_vs_heuristic_opus (and similar) from
    batch evaluations. We convert these into pairwise result records so they
    contribute to the unified rating fit.
    """
    # Number of games per eval batch (from the rank eval, typically 512 games
    # split across opponents). Use a conservative estimate.
    EVAL_GAMES = 256

    for opponent in ("heuristic_opus",):
        wr_key = f"rank_winrate_vs_{opponent}"
        wr = entry.get(wr_key)
        if wr is None:
            continue
        wr = float(wr)
        wins_a = round(wr * EVAL_GAMES)
        wins_b = EVAL_GAMES - wins_a
        results.append({
            "a": entity,
            "b": opponent,
            "wins_a": float(wins_a),
            "wins_b": float(wins_b),
            "ties": 0.0,
            "games": float(EVAL_GAMES),
        })


def _normalize_entity(entity_id: str) -> str:
    """Map model IDs to league entity IDs so results connect properly.

    The human rating system uses model IDs like "net:league:1926" while the
    league uses "ckpt:1926". This normalizes to the league format.
    """
    if entity_id.startswith("net:league:"):
        idx = entity_id.split(":")[-1]
        return f"ckpt:{idx}"
    # Other net model IDs: "net:<run>:<idx>"
    if entity_id.startswith("net:") and entity_id.count(":") == 2:
        _, _run, idx = entity_id.split(":")
        return f"ckpt:{idx}"
    return entity_id


def combined_ratings(
    workspace_root: pathlib.Path, store: JsonPlayStore
) -> dict[str, float]:
    """Fit ratings from the union of league results and human game results.

    Merges all pairwise match data into one pool and runs a single
    fit_anchored_ratings call. This gives every entity (checkpoints, bots,
    LLM agents, humans) a rating on the same unified scale.

    Anchors: random=1000, heuristic=2500 (fixed).
    """
    from agent.train import league as LG
    from agent.train import ranking as R

    anchors: dict[str, float] = dict(R.DEFAULT_ANCHORS)
    all_results: list[dict[str, Any]] = []

    # 1. League results (checkpoint vs checkpoint, checkpoint vs bots)
    league_root = workspace_root / "agent" / "runs" / "league"
    if league_root.exists():
        try:
            league = LG.League(league_root)
            all_results.extend(league.manifest.get("results", []))
            # Also synthesize pairwise results from per-entry eval stats.
            # Each entry has rank_winrate_vs_heuristic_opus etc. from batch
            # evals that aren't recorded as explicit pairwise results.
            for entry in league.manifest.get("entries", []):
                idx = int(entry.get("idx", -1))
                entity = f"ckpt:{idx}"
                _add_eval_results(all_results, entity, entry)
        except Exception:
            logger.debug("Failed to load league results", exc_info=True)

    # 2. Human game results (human vs bots/checkpoints)
    #    Human results use model IDs like "net:league:1926" while the league
    #    uses "ckpt:1926". Normalize to league format so they connect.
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        for r in blob.get("results", []):
            normalized = dict(r)
            normalized["a"] = _normalize_entity(normalized["a"])
            normalized["b"] = _normalize_entity(normalized["b"])
            all_results.append(normalized)

    if not all_results:
        return dict(anchors)

    # Collect initial guesses from league entries for faster convergence.
    initial: dict[str, float] = {}
    if league_root.exists():
        try:
            league = LG.League(league_root)
            for entry in league.manifest.get("entries", []):
                idx = int(entry.get("idx", -1))
                entity = f"ckpt:{idx}"
                rating = entry.get("rating")
                if rating is not None:
                    initial[entity] = float(rating)
            # Also use floating entity ratings as initial guesses
            for entity, fe in league.manifest.get("floating_entities", {}).items():
                rating = fe.get("rating")
                if rating is not None:
                    initial[entity] = float(rating)
        except Exception:
            pass

    try:
        return R.fit_anchored_ratings(
            all_results, anchors=anchors, initial=initial
        )
    except Exception:
        logger.debug("Failed to compute combined ratings", exc_info=True)
        return dict(anchors)


# Public alias used by service.py
league_ratings = combined_ratings


def _load_floating_entities(workspace_root: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Load floating_entities from league.json (e.g. bedrock_claude_sonnet, heuristic_opus)."""
    from agent.train import league as LG

    league_root = workspace_root / "agent" / "runs" / "league"
    if not league_root.exists():
        return {}
    try:
        league = LG.League(league_root)
        return league.manifest.get("floating_entities", {})
    except Exception:
        return {}


def agent_leaderboard_rows(
    workspace_root: pathlib.Path, store: JsonPlayStore
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_models = MD.discover_models(workspace_root)

    # Single unified rating fit from all match data.
    ratings = combined_ratings(workspace_root, store)

    # Load floating entities from league.json for game counts and fallback ratings.
    floating = _load_floating_entities(workspace_root)

    # For ML (net) models, only include the highest-rated one.
    net_models = [m for m in all_models if m["kind"] == "net"]
    non_net_models = [m for m in all_models if m["kind"] != "net"]

    best_net: dict[str, Any] | None = None
    if net_models:
        best_net = max(net_models, key=lambda m: float(m.get("rating", 0.0)))

    models_to_show = non_net_models
    if best_net is not None:
        models_to_show.append(best_net)

    for m in models_to_show:
        label = str(m["label"])
        if m["kind"] == "net":
            label = "ML Bot"
        entity_id = MD.model_entity_id(m)
        # Use the unified rating if the entity has match data, otherwise
        # fall back to the static rating from the model definition.
        # Normalize the entity ID to match the league format used in the fit.
        lookup_id = _normalize_entity(entity_id)
        rating = ratings.get(lookup_id, float(m.get("rating", 0.0)))

        # For game count, prefer floating_entities (from league.json) which
        # tracks games played via llm_rating_games.py. Fall back to model def.
        games = int(m.get("games", 0))
        fe = floating.get(entity_id)
        if fe is not None:
            games = int(fe.get("games", games))
            # Also use floating entity rating as fallback if not in unified fit
            if lookup_id not in ratings:
                rating = float(fe.get("rating", rating))

        rows.append(
            {
                "kind": "agent",
                "entity_id": entity_id,
                "label": label,
                "model_id": str(m["id"]),
                "bot_kind": str(m["kind"]),
                "rating": rating,
                "games": games,
            }
        )
    return rows


def human_leaderboard_rows(store: JsonPlayStore) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        wins = int(blob.get("wins", 0))
        if wins < HE.PLACEMENT_WINS_REQUIRED:
            continue
        rating = float(blob.get("rating", HE.DEFAULT_INITIAL_RATING))
        games = int(blob.get("games", 0))
        label = uname
        out.append(
            {
                "kind": "human",
                "entity_id": f"human:{uname}",
                "label": label,
                "username": uname,
                "rating": rating,
                "games": games,
            }
        )
    return out


def leaderboard_response(workspace_root: pathlib.Path, store: JsonPlayStore) -> dict[str, Any]:
    agents = agent_leaderboard_rows(workspace_root, store)
    humans = human_leaderboard_rows(store)
    entities = humans + agents
    entities.sort(key=lambda r: float(r.get("rating", 0.0)), reverse=True)
    return {"entities": entities}
