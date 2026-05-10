"""Encodes batched engine state into feature tensors for the network.

Produces two tensors per batch:
- `global_feat` (B, D_GLOBAL): fixed-size dense vector with gems, phase, turn
  counter, number of players, one-hot active seat info, noble slot info, etc.
- `card_feat`   (B, N_CARDS, D_CARD): per-card feature rows for every card
  visible or potentially reserved (grid slots 0..11 + reserved 0..MAX_P*MAX_RES).

The network uses attention over `card_feat` to produce per-card action logits,
then concatenates with global features for a value head.

Perspective: encoding is from the current player's point of view. Other
players' state is attached as additional per-seat features ordered cyclically
from the current player (seat 0 = current).
"""

from __future__ import annotations

import torch

from ..env import actions as A
from ..env import cards as C
from ..env import batched_engine as BE

MAX_PLAYERS = BE.MAX_PLAYERS
NUM_GRID = BE.NUM_TIERS * BE.NUM_GRID_SLOTS  # 12
NUM_RESERVED_CARDS = MAX_PLAYERS * BE.MAX_RESERVED  # 12 (ours + opponents, padded)

N_CARDS: int = NUM_GRID + NUM_RESERVED_CARDS  # 24

D_GEMS = 6
D_BONUSES = 5
D_PLAYER_META = 4  # points, nobles_claimed, reserved_count, is_current
D_SEAT_FEAT = D_GEMS + D_BONUSES + D_PLAYER_META  # 15

D_NOBLES_FLAT = BE.MAX_NOBLE_SLOTS * (C.NUM_COLORS + 1)  # 5 * 6 = 30

D_PC_OH = 3  # one-hot for 2p/3p/4p

D_GLOBAL = (
    D_GEMS  # gem pool
    + 1  # phase (one-hot 3) -> store as int
    + 3  # phase one-hot
    + D_PC_OH  # num_players one-hot (2p/3p/4p)
    + 1  # last_trigger present
    + 1  # turns_since_trigger
    + MAX_PLAYERS * D_SEAT_FEAT
    + D_NOBLES_FLAT
)

D_CARD = (
    1  # present
    + 1  # tier (1-3, 0 if absent)
    + 1  # pv
    + C.NUM_COLORS  # bonus one-hot
    + C.NUM_COLORS  # cost
    + 1  # grid (vs reserved)
    + 1  # reserved by current player
    + 1  # reserved by other
    + 1  # hidden
    + 1  # affordable
)


def _gather_card_feat(
    ids: torch.Tensor,
    tables: dict,
) -> torch.Tensor:
    """ids (B,K) long. Returns (B,K,D_minimal) with cost/bonus/pv/level features."""
    cost = tables["card_cost_pad"][torch.where(ids < 0, torch.full_like(ids, 90), ids)].to(torch.float32)
    bonus_idx = tables["card_bonus_pad"][torch.where(ids < 0, torch.full_like(ids, 90), ids)].to(torch.long)
    pv = tables["card_points_pad"][torch.where(ids < 0, torch.full_like(ids, 90), ids)].to(torch.float32)
    # bonus one-hot
    B, K = ids.shape
    bonus_oh = torch.zeros((B, K, C.NUM_COLORS), device=ids.device)
    valid = ids >= 0
    # scatter_ is safe unconditionally: when valid is all False the update is 0.
    bonus_oh.scatter_(
        2,
        bonus_idx.clamp_min(0).unsqueeze(-1),
        valid.to(torch.float32).unsqueeze(-1),
    )
    return cost, bonus_oh, pv

def encode_state_python(
    engine: BE.BatchedEngine,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (global_feat, card_feat, card_affordable_mask)."""
    device = engine.device
    B = engine.batch_size
    nP = engine.num_players
    cp = engine.current_player.to(torch.long)  # (B,)

    # Global gem pool
    gp = engine.gem_pool.to(torch.float32)  # (B,6)

    # Phase one-hot
    phase = engine.phase.to(torch.long)
    phase_oh = torch.nn.functional.one_hot(phase, num_classes=3).to(torch.float32)

    # Seat features, rotated so seat 0 = current player
    tokens = engine.tokens.to(torch.float32)  # (B,P,6)
    bonuses = engine.bonuses.to(torch.float32)  # (B,P,5)
    points = engine.points.to(torch.float32)  # (B,P)
    nobles_c = engine.nobles_claimed.to(torch.float32)  # (B,P)
    reserved = engine.reserved  # (B,P,3)
    reserved_count = (reserved >= 0).sum(dim=-1).to(torch.float32)  # (B,P)
    is_cur = torch.nn.functional.one_hot(cp, num_classes=MAX_PLAYERS).to(torch.float32)  # (B,P)

    seat_feat = torch.cat(
        [
            tokens,
            bonuses,
            points.unsqueeze(-1),
            nobles_c.unsqueeze(-1),
            reserved_count.unsqueeze(-1),
            is_cur.unsqueeze(-1),
        ],
        dim=-1,
    )  # (B,P,D_SEAT_FEAT)

    # Rotate so current player = seat 0
    seat_idx = (
        (torch.arange(MAX_PLAYERS, device=device).unsqueeze(0) + cp.unsqueeze(-1))
        % MAX_PLAYERS
    )  # (B,P) - indices to gather from original seats
    rot_seat = seat_feat.gather(
        1,
        seat_idx.unsqueeze(-1).expand(-1, -1, seat_feat.shape[-1]).to(torch.long),
    )
    rot_flat = rot_seat.reshape(B, MAX_PLAYERS * D_SEAT_FEAT)

    # Nobles block: for each noble slot, requirement (5) + present flag (1)
    noble_ids = engine.noble_ids.to(torch.long)
    noble_req_pad = engine.tables["noble_req_pad"]
    safe_idx = torch.where(noble_ids < 0, torch.full_like(noble_ids, C.NUM_COLORS * 0 + 10), noble_ids)
    noble_req = noble_req_pad[safe_idx].to(torch.float32)  # (B,5,5)
    noble_present = (noble_ids >= 0).to(torch.float32).unsqueeze(-1)  # (B,5,1)
    nobles_flat = torch.cat([noble_req, noble_present], dim=-1).reshape(B, D_NOBLES_FLAT)

    last_trig_present = (engine.last_trigger >= 0).to(torch.float32).unsqueeze(-1)
    turns_since = engine.turns_since_trigger.to(torch.float32).unsqueeze(-1)
    # One-hot player count: index 0=2p, 1=3p, 2=4p
    pc_idx = torch.full((B,), nP - 2, dtype=torch.long, device=device)
    num_players_oh = torch.nn.functional.one_hot(pc_idx, num_classes=3).to(torch.float32)  # (B, 3)
    phase_scalar = phase.to(torch.float32).unsqueeze(-1)

    global_feat = torch.cat(
        [
            gp,
            phase_scalar,
            phase_oh,
            num_players_oh,
            last_trig_present,
            turns_since,
            rot_flat,
            nobles_flat,
        ],
        dim=-1,
    )

    # Card features: grid (12) + reserved slots (4 seats * 3 = 12) in rotated seat order
    grid_ids = engine.grid_card.reshape(B, NUM_GRID).to(torch.long)
    rot_reserved = engine.reserved.gather(
        1, seat_idx.unsqueeze(-1).expand(-1, -1, BE.MAX_RESERVED).to(torch.long)
    ).reshape(B, NUM_RESERVED_CARDS).to(torch.long)
    rot_reserved_hidden = engine.reserved_hidden.gather(
        1, seat_idx.unsqueeze(-1).expand(-1, -1, BE.MAX_RESERVED).to(torch.long)
    ).reshape(B, NUM_RESERVED_CARDS)

    all_ids = torch.cat([grid_ids, rot_reserved], dim=-1)  # (B, 24)
    present = (all_ids >= 0).to(torch.float32)

    cost, bonus_oh, pv = _gather_card_feat(all_ids, engine.tables)

    # tier feat
    level_pad = torch.cat(
        [engine.tables["card_level"], torch.zeros((1,), dtype=torch.int8, device=device)], dim=0
    ).to(torch.float32)
    safe = torch.where(all_ids < 0, torch.full_like(all_ids, 90), all_ids)
    tier = level_pad[safe].unsqueeze(-1)

    is_grid = torch.zeros((B, N_CARDS), device=device)
    is_grid[:, :NUM_GRID] = 1.0

    is_grid_f = is_grid.unsqueeze(-1)

    # reserved-by-current
    own_mask = torch.zeros((B, N_CARDS), device=device)
    own_mask[:, NUM_GRID : NUM_GRID + BE.MAX_RESERVED] = 1.0
    reserved_by_cur = own_mask.unsqueeze(-1)
    reserved_by_other = torch.zeros_like(reserved_by_cur)
    reserved_by_other[:, NUM_GRID + BE.MAX_RESERVED :] = 1.0

    hidden = torch.zeros((B, N_CARDS), device=device)
    hidden[:, NUM_GRID:] = rot_reserved_hidden.to(torch.float32)
    hidden = hidden.unsqueeze(-1)

    # affordability: re-use engine helper
    afford, _, _ = engine._compute_purchase_for_ids(all_ids)
    afford_f = afford.to(torch.float32).unsqueeze(-1)

    card_feat = torch.cat(
        [
            present.unsqueeze(-1),
            tier,
            pv.unsqueeze(-1),
            bonus_oh,
            cost,
            is_grid_f,
            reserved_by_cur,
            reserved_by_other,
            hidden,
            afford_f,
        ],
        dim=-1,
    )

    return global_feat, card_feat, afford


def encode_state(
    engine: BE.BatchedEngine,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (global_feat, card_feat, card_affordable_mask)."""
    return encode_state_python(engine)


def encode_state_with_legal(
    engine: BE.BatchedEngine,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (global_feat, card_feat, card_affordable_mask, legal_mask)."""
    g, c, afford = encode_state_python(engine)
    legal = engine.legal_action_mask()
    return g, c, afford, legal


def card_action_index_map() -> torch.Tensor:
    """Returns (N_CARDS,) tensor mapping each card-slot index to the BUY_GRID /
    BUY_RESERVED action index it corresponds to, or -1 for reserved-by-other.

    Indices 0..11: grid slots -> BUY_GRID_BASE..BUY_GRID_BASE+11
    Indices 12..14: current player reserved 0..2 -> BUY_RESERVED_BASE..+2
    Indices 15..23: other seat reserved -> -1 (cannot be bought)
    """
    out = torch.full((N_CARDS,), -1, dtype=torch.long)
    for i in range(NUM_GRID):
        out[i] = A.BUY_GRID_BASE + i
    for r in range(BE.MAX_RESERVED):
        out[NUM_GRID + r] = A.BUY_RESERVED_BASE + r
    return out
