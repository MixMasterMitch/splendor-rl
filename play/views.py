"""JSON-friendly views over ``BatchedEngine`` state for interactive play."""

from __future__ import annotations

from typing import Any

import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.env import cards as C


def _ti(t: torch.Tensor) -> int:
    """Extract a scalar tensor as a plain Python int."""
    return int(t.item())


def _tl(t: torch.Tensor) -> list[int]:
    """Convert a 1-D tensor to a Python list of ints."""
    return t.tolist()


def batched_to_snapshot(engine: BE.BatchedEngine, b: int = 0) -> dict[str, Any]:
    """Convert game index b in engine to a JSON-serializable snapshot dict."""
    nP = engine.num_players

    gem_pool: list[int] = _tl(engine.gem_pool[b])

    grid: list[list[int | None]] = []
    for t in range(BE.NUM_TIERS):
        row: list[int | None] = []
        for s in range(BE.NUM_GRID_SLOTS):
            cid = _ti(engine.grid_card[b, t, s])
            row.append(None if cid < 0 else cid)
        grid.append(row)

    deck_counts: list[int] = [_ti(engine.deck_top[b, t]) for t in range(BE.NUM_TIERS)]

    nobles: list[int | None] = []
    for ns in range(BE.MAX_NOBLE_SLOTS):
        nid = _ti(engine.noble_ids[b, ns])
        nobles.append(None if nid < 0 else nid)

    players: list[dict[str, Any]] = []
    for p in range(nP):
        reserved: list[int | None] = []
        reserved_hidden: list[bool] = []
        for r in range(BE.MAX_RESERVED):
            cid = _ti(engine.reserved[b, p, r])
            reserved.append(None if cid < 0 else cid)
            reserved_hidden.append(bool(engine.reserved_hidden[b, p, r].item()))
        players.append(
            {
                "tokens": _tl(engine.tokens[b, p]),
                "bonuses": _tl(engine.bonuses[b, p]),
                "reserved": reserved,
                "reserved_hidden": reserved_hidden,
                "points": _ti(engine.points[b, p]),
                "nobles_claimed": _ti(engine.nobles_claimed[b, p]),
            }
        )

    return {
        "gem_pool": gem_pool,
        "grid": grid,
        "deck_counts": deck_counts,
        "nobles": nobles,
        "current_player": _ti(engine.current_player[b]),
        "phase": _ti(engine.phase[b]),
        "turn_count": _ti(engine.turns_since_trigger[b]),
        "ended": bool(engine.ended[b].item()),
        "players": players,
    }


def action_detail(action: int) -> dict[str, Any]:
    """Decode an action integer into a structured human-friendly dict."""
    if A.TAKE3_BASE <= action < A.TAKE3_BASE + A.TAKE3_COUNT:
        combo = A.TAKE3_COMBOS[action - A.TAKE3_BASE]
        return {"kind": "take3", "colors": list(combo)}
    if A.TAKE2_BASE <= action < A.TAKE2_BASE + A.TAKE2_COUNT:
        return {"kind": "take2", "color": action - A.TAKE2_BASE}
    if A.RESERVE_GRID_BASE <= action < A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT:
        x = action - A.RESERVE_GRID_BASE
        return {"kind": "reserve_grid", "tier": x // A.NUM_GRID_SLOTS, "slot": x % A.NUM_GRID_SLOTS}
    if A.RESERVE_BLIND_BASE <= action < A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT:
        return {"kind": "reserve_blind", "tier": action - A.RESERVE_BLIND_BASE}
    if A.BUY_GRID_BASE <= action < A.BUY_GRID_BASE + A.BUY_GRID_COUNT:
        x = action - A.BUY_GRID_BASE
        return {"kind": "buy_grid", "tier": x // A.NUM_GRID_SLOTS, "slot": x % A.NUM_GRID_SLOTS}
    if A.BUY_RESERVED_BASE <= action < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT:
        return {"kind": "buy_reserved", "slot": action - A.BUY_RESERVED_BASE}
    if action == A.PASS_ACTION:
        return {"kind": "pass"}
    if A.DISCARD_BASE <= action < A.DISCARD_BASE + A.DISCARD_COUNT:
        return {"kind": "discard", "token": action - A.DISCARD_BASE}
    if A.PICK_NOBLE_BASE <= action < A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT:
        return {"kind": "pick_noble", "slot": action - A.PICK_NOBLE_BASE}
    return {"kind": "unknown", "raw": action}


def cards_table() -> list[dict[str, Any]]:
    """All development cards serialized from the loaded ``C.CARDS`` list."""
    return [
        {
            "id": c.card_id,
            "level": c.level,
            "bonus": c.bonus,
            "points": c.points,
            "cost": list(c.cost),
        }
        for c in C.CARDS
    ]


def nobles_table() -> list[dict[str, Any]]:
    """All noble tiles serialized from the loaded ``C.NOBLES`` list."""
    return [
        {
            "id": n.noble_id,
            "name": n.name,
            "points": n.points,
            "requirement": list(n.requirement),
        }
        for n in C.NOBLES
    ]


def redact_snapshot_for_human(snapshot: dict[str, Any], human_seat: int) -> dict[str, Any]:
    """Hide face-down opponent reserved cards while keeping visibility flags."""
    players = snapshot.get("players", [])
    new_players: list[dict[str, Any]] = []
    for seat, player in enumerate(players):
        if seat == human_seat:
            new_players.append(player)
            continue
        reserved = list(player.get("reserved", []))
        hidden = list(player.get("reserved_hidden", []))
        redacted = [
            (None if (r < len(hidden) and hidden[r]) else reserved[r]) for r in range(len(reserved))
        ]
        new_players.append({**player, "reserved": redacted, "reserved_hidden": hidden})
    return {**snapshot, "players": new_players}
