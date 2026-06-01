"""Fast leaderboard ratings using fixed bot ratings from league.json.

Bot ratings are read directly from the league manifest (pre-computed by the
training pipeline). Human ratings are computed per-PC via single-variable
bisection against fixed bot anchors — O(1) per human, no multi-entity MLE.

Per-PC ratings are calibrated to a common scale using reference-anchor-derived
multipliers (heuristic_opus aligned across all PCs) and combined into a single
rating weighted by games played at each PC.
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Any

from play import models as MD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrored from agent/train/ranking.py to avoid torch import)
# ---------------------------------------------------------------------------

RANDOM_ANCHOR_RATING: float = 1000.0
DEFAULT_INITIAL_RATING: float = 1500.0
RATING_SCALE: float = 1000.0
PLAYER_COUNTS = (2, 3, 4)

# Per-player-count reference anchors (mirrored from agent/train/ranking.py).
# These pin the rating scale to the fixed heuristic triangle.
REFERENCE_ANCHORS_PER_PC: dict[int, dict[str, float]] = {
    2: {"random": 1000.0, "heuristic": 2679.6, "heuristic_opus": 3138.6},
    3: {"random": 1000.0, "heuristic": 2454.6, "heuristic_opus": 2845.4},
    4: {"random": 1000.0, "heuristic": 1487.2, "heuristic_opus": 1823.5},
}

# Calibration scales derived from reference anchors: scale each PC so that
# heuristic_opus maps to the same calibrated value (2p baseline) at every PC.
_CALIBRATION_REF_ENTITY = "heuristic_opus"
_BASELINE_DIFF = REFERENCE_ANCHORS_PER_PC[2][_CALIBRATION_REF_ENTITY] - RANDOM_ANCHOR_RATING
CALIBRATION_SCALE: dict[int, float] = {
    pc: _BASELINE_DIFF / max(1e-9, REFERENCE_ANCHORS_PER_PC[pc][_CALIBRATION_REF_ENTITY] - RANDOM_ANCHOR_RATING)
    for pc in PLAYER_COUNTS
}

# Bayesian regularization for per-PC rating fits.
PRIOR_MEAN_RATING: float = 1500.0
PRIOR_GHOST_GAMES: float = 4.0

# Minimum wins before a human appears on the leaderboard.
PLACEMENT_WINS_REQUIRED: int = 5


def calibrate_rating(raw: float, pc: int) -> float:
    """Scale a raw per-PC rating to the common (2p-equivalent) scale."""
    return RANDOM_ANCHOR_RATING + (raw - RANDOM_ANCHOR_RATING) * CALIBRATION_SCALE[pc]


# ---------------------------------------------------------------------------
# Public helpers for computing calibrated combined ratings
# ---------------------------------------------------------------------------


def bot_anchors_per_pc(workspace_root: pathlib.Path) -> dict[int, dict[str, float]]:
    """Build per-PC anchor maps from the league manifest.

    Returns: {pc: {entity_id: raw_rating}} for use with _compute_human_ratings.
    """
    manifest = _load_league_manifest(workspace_root)
    bot_data = _bot_ratings_from_league(manifest)
    anchors: dict[int, dict[str, float]] = {pc: {} for pc in PLAYER_COUNTS}
    for entity, data in bot_data.items():
        for pc in PLAYER_COUNTS:
            raw = data.get(f"raw_{pc}p")
            if raw is not None:
                anchors[pc][entity] = raw
    for pc in PLAYER_COUNTS:
        anchors[pc]["random"] = RANDOM_ANCHOR_RATING
    return anchors


def combined_rating_for_blob(
    blob: dict[str, Any],
    anchors_per_pc: dict[int, dict[str, float]],
) -> float | None:
    """Compute the calibrated combined rating for a single human blob.

    Returns the weighted-average calibrated rating, or None if no per-PC
    rating could be computed.
    """
    human_data = _compute_human_ratings(blob, anchors_per_pc)
    return human_data.get("rating")


# ---------------------------------------------------------------------------
# Legacy helper kept for lambda_handler.py compatibility
# ---------------------------------------------------------------------------


def _add_eval_results(
    results: list[dict[str, Any]], entity: str, entry: dict[str, Any]
) -> None:
    """Synthesize pairwise results from per-entry eval winrate stats.

    Used by lambda_handler.py which still does its own LBFGS fit.
    """
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


# ---------------------------------------------------------------------------
# Bot ratings from league.json (fixed, no fitting)
# ---------------------------------------------------------------------------


def _load_league_manifest(workspace_root: pathlib.Path) -> dict[str, Any]:
    """Load the league.json manifest directly (no torch dependency).

    Checks workspace_root first, then falls back to the module-relative path
    (for Lambda where workspace_root is /tmp/workspace but league.json is
    bundled alongside the code).
    """
    import json

    league_path = workspace_root / "agent" / "runs" / "league" / "league.json"
    if not league_path.exists():
        # Fallback: module-relative path (bundled in container image)
        league_path = pathlib.Path(__file__).resolve().parent.parent / "agent" / "runs" / "league" / "league.json"
    if not league_path.exists():
        return {}
    try:
        with open(league_path) as f:
            return json.load(f)
    except Exception:
        logger.debug("Failed to load league.json", exc_info=True)
        return {}


def _bot_ratings_from_league(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract per-PC ratings and game counts for all bots from the league manifest.

    NOTE: league.json stores CALIBRATED per-PC ratings (already scaled to the
    common 2p-equivalent scale). We store both the calibrated values (for
    display) and back-compute raw values (for use as anchors in human fitting).

    Returns: {entity_id: {"rating": combined, "rating_2p": calibrated, ...,
                           "raw_2p": uncalibrated, ..., "games": int}}
    """
    out: dict[str, dict[str, Any]] = {}

    # Checkpoint entries
    for entry in manifest.get("entries", []):
        idx = int(entry.get("idx", -1))
        entity = f"ckpt:{idx}"
        data: dict[str, Any] = {"games": int(entry.get("games", 0))}
        for pc in PLAYER_COUNTS:
            calibrated = entry.get(f"rating_{pc}p")
            if calibrated is not None:
                calibrated = float(calibrated)
                data[f"rating_{pc}p"] = calibrated
                # Back-compute raw: calibrated = RANDOM + (raw - RANDOM) * scale
                # => raw = RANDOM + (calibrated - RANDOM) / scale
                scale = CALIBRATION_SCALE[pc]
                raw = RANDOM_ANCHOR_RATING + (calibrated - RANDOM_ANCHOR_RATING) / scale
                data[f"raw_{pc}p"] = raw
        if "rating" in entry:
            data["rating"] = float(entry["rating"])
        out[entity] = data

    # Floating entities (heuristic, heuristic_opus, bedrock_claude_sonnet, etc.)
    for entity, fe in manifest.get("floating_entities", {}).items():
        data: dict[str, Any] = {"games": int(fe.get("games", 0))}
        for pc in PLAYER_COUNTS:
            calibrated = fe.get(f"rating_{pc}p")
            if calibrated is not None:
                calibrated = float(calibrated)
                data[f"rating_{pc}p"] = calibrated
                scale = CALIBRATION_SCALE[pc]
                raw = RANDOM_ANCHOR_RATING + (calibrated - RANDOM_ANCHOR_RATING) / scale
                data[f"raw_{pc}p"] = raw
        if "rating" in fe:
            data["rating"] = float(fe["rating"])
        out[entity] = data

    return out


# ---------------------------------------------------------------------------
# Human rating computation (single-variable bisection per PC)
# ---------------------------------------------------------------------------


def _expected_score(r_a: float, r_b: float) -> float:
    """Probability that A beats B under Bradley-Terry with scale 1000."""
    return 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / RATING_SCALE))


def _fit_human_rating_single(
    results_for_pc: list[tuple[str, float, float]],
    anchors: dict[str, float],
    prior_mean: float = PRIOR_MEAN_RATING,
    prior_ghost_games: float = PRIOR_GHOST_GAMES,
) -> float | None:
    """Fit a single human's rating for one player count via bisection.

    results_for_pc: list of (opponent_entity, wins_human, wins_opponent)
    anchors: {entity: rating} for all opponents (fixed)

    Returns the fitted rating, or None if no data for this PC.
    """
    matches: list[tuple[float, float, float]] = []
    for opp_entity, wins_h, wins_o in results_for_pc:
        opp_rating = anchors.get(opp_entity)
        if opp_rating is None:
            continue
        n = wins_h + wins_o
        if n <= 0:
            continue
        matches.append((opp_rating, wins_h, n))

    if not matches:
        return None

    ghost_n = max(0.0, float(prior_ghost_games))
    ghost_score = 0.5 * ghost_n

    def gradient(r: float) -> float:
        g = 0.0
        for opp_r, human_score, n in matches:
            p = _expected_score(r, opp_r)
            g += human_score - n * p
        if ghost_n > 0:
            g += ghost_score - ghost_n * _expected_score(r, prior_mean)
        return g

    lo, hi = -5000.0, 8000.0
    if gradient(lo) <= 0:
        return lo
    if gradient(hi) >= 0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gradient(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _normalize_entity(entity_id: str) -> str:
    """Map model IDs to league entity IDs so results connect properly."""
    if entity_id.startswith("net:league:"):
        idx = entity_id.split(":")[-1]
        return f"ckpt:{idx}"
    if entity_id.startswith("net:") and entity_id.count(":") == 2:
        _, _run, idx = entity_id.split(":")
        return f"ckpt:{idx}"
    return entity_id


def _build_human_per_pc_results(
    blob: dict[str, Any],
) -> dict[int, list[tuple[str, float, float]]]:
    """Build per-PC pairwise results for a human from their history.

    For each game in history:
    - Determine player count from len(ranks) or len(opponents)+1
    - Human wins: record 1 win against each losing opponent
    - Human loses: record 1 loss against the winner only

    Handles legacy entries that lack ``ranks`` but have ``opponents[].score``
    and ``human_rank``.

    Returns: {pc: [(opponent_entity, wins_human, wins_opponent), ...]}
    """
    # Accumulate per (pc, opponent) -> (wins_h, wins_o)
    accum: dict[tuple[int, str], list[float, float]] = {}

    for entry in blob.get("history", []):
        if entry.get("legacy"):
            continue
        opponents = entry.get("opponents", [])
        if not opponents:
            continue

        ranks = entry.get("ranks")
        if ranks:
            pc = len(ranks)
        else:
            # Legacy entries without ranks: infer PC from opponents list
            pc = len(opponents) + 1

        if pc not in (2, 3, 4):
            continue
        human_rank = int(entry.get("human_rank", -1))

        if human_rank == 0:
            # Human won the table: in the BT encoding, the winner beat each
            # losing opponent once.
            for opp in opponents:
                entity = _normalize_entity(str(opp["entity_id"]))
                key = (pc, entity)
                if key not in accum:
                    accum[key] = [0.0, 0.0]
                accum[key][0] += 1.0
        elif ranks:
            # Human lost and we have full rank info: record 1 loss against
            # the winner (rank 0) only.
            for opp in opponents:
                opp_seat = int(opp["seat"])
                opp_rank = ranks[opp_seat]
                if opp_rank == 0:
                    entity = _normalize_entity(str(opp["entity_id"]))
                    key = (pc, entity)
                    if key not in accum:
                        accum[key] = [0.0, 0.0]
                    accum[key][1] += 1.0
                    break
        else:
            # Legacy entry without ranks: use opponent score field.
            # score=0.0 means that opponent beat the human.
            # In 2p this is unambiguous (the single opponent won).
            # In multiplayer, record a loss against each opponent that beat
            # the human, weighted so total loss weight = 1.
            losers = [opp for opp in opponents if float(opp.get("score", 1.0)) == 0.0]
            if not losers:
                # Fallback: if no score info, assume single opponent won
                losers = opponents[:1]
            loss_weight = 1.0 / len(losers) if losers else 1.0
            for opp in losers:
                entity = _normalize_entity(str(opp["entity_id"]))
                key = (pc, entity)
                if key not in accum:
                    accum[key] = [0.0, 0.0]
                accum[key][1] += loss_weight

    # Group by PC
    per_pc: dict[int, list[tuple[str, float, float]]] = {pc: [] for pc in PLAYER_COUNTS}
    for (pc, entity), (wins_h, wins_o) in accum.items():
        per_pc[pc].append((entity, wins_h, wins_o))

    return per_pc


def _count_human_physical_games_per_pc(blob: dict[str, Any]) -> dict[int, float]:
    """Count physical games in a human history by player count."""
    counts: dict[int, float] = {pc: 0.0 for pc in PLAYER_COUNTS}
    for entry in blob.get("history", []):
        if entry.get("legacy"):
            continue
        opponents = entry.get("opponents", [])
        if not opponents:
            continue

        ranks = entry.get("ranks")
        if ranks:
            pc = len(ranks)
        else:
            pc = len(opponents) + 1

        if pc in counts:
            counts[pc] += 1.0
    return counts


def _compute_human_ratings(
    blob: dict[str, Any],
    bot_anchors_per_pc: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """Compute per-PC and combined rating for a single human.

    Returns: {"rating": combined, "rating_2p": ..., "rating_3p": ..., "rating_4p": ...,
              "games_2p": ..., "games_3p": ..., "games_4p": ...}
    """
    per_pc_results = _build_human_per_pc_results(blob)
    games_per_pc = _count_human_physical_games_per_pc(blob)

    calibrated_sum = 0.0
    weight_sum = 0.0
    out: dict[str, Any] = {}

    for pc in PLAYER_COUNTS:
        results = per_pc_results[pc]
        if not results:
            continue
        anchors = bot_anchors_per_pc.get(pc, {})
        raw = _fit_human_rating_single(results, anchors)
        if raw is None:
            continue
        cal = calibrate_rating(raw, pc)
        out[f"rating_{pc}p"] = cal
        # Weight combined ratings by physical games, not by expanded BT
        # pairwise mass. A 4p table win now contributes three BT wins but is
        # still one played game.
        games_at_pc = games_per_pc.get(pc, 0.0)
        if games_at_pc <= 0:
            games_at_pc = sum(wh + wo for _, wh, wo in results)
        out[f"games_{pc}p"] = games_at_pc
        calibrated_sum += cal * games_at_pc
        weight_sum += games_at_pc

    if weight_sum > 0:
        out["rating"] = calibrated_sum / weight_sum
    else:
        out["rating"] = None

    return out


# ---------------------------------------------------------------------------
# Public API (used by service.py)
# ---------------------------------------------------------------------------


def _load_floating_entities(workspace_root: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Load floating_entities from league.json."""
    manifest = _load_league_manifest(workspace_root)
    return manifest.get("floating_entities", {})

def combined_ratings(
    workspace_root: pathlib.Path, store: Any
) -> dict[str, float]:
    """Return combined ratings for all entities (bots + humans).

    Bot ratings come directly from league.json. Human ratings are computed
    via fast single-variable bisection per PC.
    """
    manifest = _load_league_manifest(workspace_root)
    bot_data = _bot_ratings_from_league(manifest)

    # Build per-PC anchor maps using RAW (uncalibrated) ratings for human fitting
    bot_anchors_per_pc: dict[int, dict[str, float]] = {pc: {} for pc in PLAYER_COUNTS}
    for entity, data in bot_data.items():
        for pc in PLAYER_COUNTS:
            raw = data.get(f"raw_{pc}p")
            if raw is not None:
                bot_anchors_per_pc[pc][entity] = raw
    # Always include random anchor
    for pc in PLAYER_COUNTS:
        bot_anchors_per_pc[pc]["random"] = RANDOM_ANCHOR_RATING

    out: dict[str, float] = {"random": RANDOM_ANCHOR_RATING}

    # Bot combined ratings
    for entity, data in bot_data.items():
        if "rating" in data:
            out[entity] = data["rating"]

    # Human ratings
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        human_data = _compute_human_ratings(blob, bot_anchors_per_pc)
        rating = human_data.get("rating")
        if rating is not None:
            out[f"human:{uname}"] = rating

    return out


def combined_ratings_detailed(
    workspace_root: pathlib.Path, store: Any
) -> dict[str, dict]:
    """Return per-PC and combined ratings for all entities.

    Returns: {entity: {"rating": combined, "rating_2p": ..., "rating_3p": ..., "rating_4p": ...}}
    """
    manifest = _load_league_manifest(workspace_root)
    bot_data = _bot_ratings_from_league(manifest)

    # Build per-PC anchor maps using RAW ratings for human fitting
    bot_anchors_per_pc: dict[int, dict[str, float]] = {pc: {} for pc in PLAYER_COUNTS}
    for entity, data in bot_data.items():
        for pc in PLAYER_COUNTS:
            raw = data.get(f"raw_{pc}p")
            if raw is not None:
                bot_anchors_per_pc[pc][entity] = raw
    for pc in PLAYER_COUNTS:
        bot_anchors_per_pc[pc]["random"] = RANDOM_ANCHOR_RATING

    out: dict[str, dict] = {}

    # Bot ratings: per-PC values from league.json are already calibrated
    for entity, data in bot_data.items():
        entry: dict[str, Any] = {}
        calibrated_sum = 0.0
        weight_sum = 0.0
        for pc in PLAYER_COUNTS:
            cal = data.get(f"rating_{pc}p")  # Already calibrated in league.json
            if cal is not None:
                entry[f"rating_{pc}p"] = cal
                # Use equal weight for bots (they play many games at each PC)
                weight = 1.0
                calibrated_sum += cal * weight
                weight_sum += weight
        if weight_sum > 0:
            entry["rating"] = calibrated_sum / weight_sum
        elif "rating" in data:
            entry["rating"] = data["rating"]
        else:
            entry["rating"] = None
        out[entity] = entry

    # Human ratings
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        human_data = _compute_human_ratings(blob, bot_anchors_per_pc)
        out[f"human:{uname}"] = human_data

    return out


# Public alias used by service.py
league_ratings = combined_ratings


def agent_leaderboard_rows(
    workspace_root: pathlib.Path, store: Any
) -> list[dict[str, Any]]:
    """Build leaderboard rows for bot agents."""
    rows: list[dict[str, Any]] = []
    all_models = MD.discover_models(workspace_root)
    manifest = _load_league_manifest(workspace_root)
    bot_data = _bot_ratings_from_league(manifest)
    floating = manifest.get("floating_entities", {})

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
            label = "RL Trained Bot"
        entity_id = MD.model_entity_id(m)
        lookup_id = _normalize_entity(entity_id)

        # Get ratings from league data (already calibrated per-PC)
        entity_data = bot_data.get(lookup_id, {})

        # Per-PC ratings (already calibrated in league.json)
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
            # Fallback rating for entities not in league entries
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
        rows.append(row)
    return rows


def human_leaderboard_rows(
    workspace_root_or_store: Any,
    store: Any = None,
) -> list[dict[str, Any]]:
    """Build leaderboard rows for human players.

    Accepts either (workspace_root, store) or just (store,) for backward
    compatibility with lambda_handler.py. When called with just a store,
    human ratings fall back to the stored per-user rating (no per-PC refit).
    """
    if store is None:
        # Called as human_leaderboard_rows(store) — legacy lambda path
        actual_store = workspace_root_or_store
        bot_anchors_per_pc: dict[int, dict[str, float]] | None = None
    else:
        # Called as human_leaderboard_rows(workspace_root, store)
        actual_store = store
        workspace_root = workspace_root_or_store
        manifest = _load_league_manifest(workspace_root)
        bot_data = _bot_ratings_from_league(manifest)
        bot_anchors_per_pc = {pc: {} for pc in PLAYER_COUNTS}
        for entity, data in bot_data.items():
            for pc in PLAYER_COUNTS:
                raw = data.get(f"raw_{pc}p")
                if raw is not None:
                    bot_anchors_per_pc[pc][entity] = raw
        for pc in PLAYER_COUNTS:
            bot_anchors_per_pc[pc]["random"] = RANDOM_ANCHOR_RATING

    out: list[dict[str, Any]] = []
    for blob in actual_store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        wins = int(blob.get("wins", 0))
        if wins < PLACEMENT_WINS_REQUIRED:
            continue
        games = int(blob.get("games", 0))

        if bot_anchors_per_pc is not None:
            human_data = _compute_human_ratings(blob, bot_anchors_per_pc)
            rating = human_data.get("rating")
        else:
            human_data = {}
            rating = None

        if rating is None:
            rating = float(blob.get("rating", DEFAULT_INITIAL_RATING))

        row: dict[str, Any] = {
            "kind": "human",
            "entity_id": f"human:{uname}",
            "label": uname,
            "username": uname,
            "rating": round(rating),
            "games": games,
        }
        for pc in PLAYER_COUNTS:
            val = human_data.get(f"rating_{pc}p")
            if val is not None:
                row[f"rating_{pc}p"] = round(val)
        out.append(row)
    return out


def leaderboard_response(workspace_root: pathlib.Path, store: Any) -> dict[str, Any]:
    """Build the full leaderboard response."""
    agents = agent_leaderboard_rows(workspace_root, store)
    humans = human_leaderboard_rows(workspace_root, store)

    entities = humans + agents
    entities.sort(key=lambda r: float(r.get("rating", 0.0)), reverse=True)
    return {"entities": entities}
