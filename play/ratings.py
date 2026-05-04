"""Leaderboard rows from rated agents plus persisted human rating files."""

from __future__ import annotations

import pathlib
from typing import Any

from play import human_elo as HE
from play import models as MD
from play.store import JsonPlayStore


def agent_leaderboard_rows(workspace_root: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in MD.discover_models(workspace_root):
        rows.append(
            {
                "kind": "agent",
                "entity_id": MD.model_entity_id(m),
                "label": str(m["label"]),
                "model_id": str(m["id"]),
                "bot_kind": str(m["kind"]),
                "rating": float(m.get("rating", m.get("elo", 0.0))),
                "elo": float(m.get("rating", m.get("elo", 0.0))),
                "games": int(m.get("games", 0)),
            }
        )
    return rows


def human_leaderboard_rows(store: JsonPlayStore) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for blob in store.list_all_user_rating_blobs():
        uname = str(blob.get("username") or blob.get("google_sub") or "")
        if not uname:
            continue
        rating = float(blob.get("rating", blob.get("elo", HE.DEFAULT_INITIAL_RATING)))
        games = int(blob.get("games", 0))
        label = uname
        out.append(
            {
                "kind": "human",
                "entity_id": f"human:{uname}",
                "label": label,
                "username": uname,
                "rating": rating,
                "elo": rating,
                "games": games,
            }
        )
    return out


def leaderboard_response(workspace_root: pathlib.Path, store: JsonPlayStore) -> dict[str, Any]:
    agents = agent_leaderboard_rows(workspace_root)
    humans = human_leaderboard_rows(store)
    combined = humans + agents
    combined.sort(key=lambda r: float(r.get("rating", 0.0)), reverse=True)
    return {"agents": agents, "humans": humans, "combined": combined}
