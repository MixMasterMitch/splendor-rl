"""Batched Splendor environment on a single CPU device.

Represents B parallel games with a fixed maximum player count P (default 4)
using ragged-free padded tensors. The active seats per game are tracked by
`active_mask` of shape (B, P).

State tensor shapes (all on the device provided in the constructor):
- `gem_pool`            (B, 6)           int8     # W,B,G,R,K,Gold
- `grid_card`           (B, 3, 4)        int16    # card_id; -1 = empty
- `deck_top`            (B, 3)           int8     # index into deck_perm[t]
- `deck_perm`           (B, 3, 40)       int16    # shuffled card ids per tier (unused slots may be -1)
- `noble_ids`           (B, 5)           int8     # noble_id; -1 = claimed/empty
- `tokens`              (B, P, 6)        int8
- `bonuses`             (B, P, 5)        int8
- `reserved`            (B, P, 3)        int16
- `reserved_hidden`     (B, P, 3)        bool
- `points`              (B, P)           int8
- `nobles_claimed`      (B, P)           int8
- `active_mask`         (B, P)           bool
- `current_player`      (B,)             int8
- `phase`               (B,)             int8
- `last_trigger`        (B,)             int8     # -1 if not triggered
- `turns_since_trigger` (B,)             int8
- `ended`               (B,)             bool

Static globals (shared, built once from env.cards):
- `CARD_COST`           (90, 5)          int16    # padded to allow broadcast
- `CARD_BONUS`          (90,)            int8
- `CARD_POINTS`         (90,)            int8
- `CARD_LEVEL`          (90,)            int8     # 1-based
- `NOBLE_REQ`           (10, 5)          int8
- `TAKE3_COMBOS`        (10, 3)          int8
- `CARD_COST_PAD`       (91, 5)          int16    # index -1 mapped to slot 90 = zeros

The engine maintains parity with `single_engine.py` for the main 2-player flow.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import actions as A
from . import cards as C

MAX_PLAYERS: int = 4
NUM_TIERS: int = 3
NUM_GRID_SLOTS: int = A.NUM_GRID_SLOTS
MAX_RESERVED: int = A.MAX_RESERVED
MAX_NOBLE_SLOTS: int = A.MAX_NOBLE_SLOTS
NUM_COLORS: int = C.NUM_COLORS
GOLD: int = C.GOLD_INDEX
NUM_CARDS: int = 90
NUM_NOBLES: int = 10
MAX_TIER_CARDS: int = 40
WINNING_POINTS: int = 15
TOKEN_LIMIT: int = 10

_STATE_TENSOR_ATTRS = (
    "gem_pool",
    "grid_card",
    "deck_perm",
    "deck_top",
    "noble_ids",
    "tokens",
    "bonuses",
    "reserved",
    "reserved_hidden",
    "points",
    "nobles_claimed",
    "active_mask",
    "current_player",
    "first_player",
    "phase",
    "last_trigger",
    "turns_since_trigger",
    "ended",
)

def _build_static_tables(device: torch.device) -> dict:
    card_cost = torch.zeros((NUM_CARDS, NUM_COLORS), dtype=torch.int16)
    card_bonus = torch.zeros((NUM_CARDS,), dtype=torch.int8)
    card_points = torch.zeros((NUM_CARDS,), dtype=torch.int8)
    card_level = torch.zeros((NUM_CARDS,), dtype=torch.int8)
    for c in C.CARDS:
        card_cost[c.card_id] = torch.tensor(c.cost, dtype=torch.int16)
        card_bonus[c.card_id] = c.bonus
        card_points[c.card_id] = c.points
        card_level[c.card_id] = c.level

    card_cost_pad = torch.cat(
        [card_cost, torch.zeros((1, NUM_COLORS), dtype=torch.int16)], dim=0
    )
    card_bonus_pad = torch.cat([card_bonus, torch.zeros((1,), dtype=torch.int8)], dim=0)
    card_points_pad = torch.cat([card_points, torch.zeros((1,), dtype=torch.int8)], dim=0)

    noble_req = torch.zeros((NUM_NOBLES, NUM_COLORS), dtype=torch.int8)
    for n in C.NOBLES:
        noble_req[n.noble_id] = torch.tensor(n.requirement, dtype=torch.int8)
    noble_req_pad = torch.cat(
        [noble_req, torch.zeros((1, NUM_COLORS), dtype=torch.int8)], dim=0
    )

    take3 = torch.tensor(A.TAKE3_COMBOS, dtype=torch.int64)

    # Precompute membership matrix: combo_membership[i, c] = True iff color c in combo i.
    # Used by legal_action_mask edge-case in a branchless form.
    combo_membership = torch.zeros((A.TAKE3_COUNT, NUM_COLORS), dtype=torch.bool)
    for i, combo in enumerate(A.TAKE3_COMBOS):
        for c in combo:
            combo_membership[i, c] = True

    return {
        "card_cost": card_cost.to(device),
        "card_cost_pad": card_cost_pad.to(device),
        "card_bonus": card_bonus.to(device),
        "card_bonus_pad": card_bonus_pad.to(device),
        "card_points": card_points.to(device),
        "card_points_pad": card_points_pad.to(device),
        "card_level": card_level.to(device),
        "noble_req": noble_req.to(device),
        "noble_req_pad": noble_req_pad.to(device),
        "take3_combos": take3.to(device),
        "combo_membership": combo_membership.to(device),
    }


def _gather_card_row(table_pad: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """ids in [-1, NUM_CARDS). -1 maps to the last (padding) row of table_pad."""
    pad_idx = table_pad.shape[0] - 1
    safe = torch.where(ids < 0, torch.full_like(ids, pad_idx), ids.to(torch.long))
    return table_pad[safe]


class BatchedEngine:
    """Vectorized Splendor engine over B parallel games.

    All operations run on `self.device`. Games may have different numbers of
    active seats (2-4); padding seats are indicated by `active_mask=False` and
    are never selected as `current_player`.
    """

    def __init__(
        self,
        batch_size: int,
        num_players: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ):
        assert num_players in (2, 3, 4), f"num_players must be 2,3,4; got {num_players}"
        self.batch_size = batch_size
        self.num_players = num_players
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.tables = _build_static_tables(self.device)
        # Cached aranges for advanced indexing in the hot path (branchless apply).
        self._b_range = torch.arange(batch_size, dtype=torch.long, device=self.device)
        self._allocate_state()
        self.reset_all()

    def _allocate_state(self) -> None:
        B, P = self.batch_size, MAX_PLAYERS
        d = self.device
        self.gem_pool = torch.zeros((B, 6), dtype=torch.int8, device=d)
        self.grid_card = torch.full((B, NUM_TIERS, NUM_GRID_SLOTS), -1, dtype=torch.int16, device=d)
        self.deck_perm = torch.full((B, NUM_TIERS, MAX_TIER_CARDS), -1, dtype=torch.int16, device=d)
        self.deck_top = torch.zeros((B, NUM_TIERS), dtype=torch.int8, device=d)
        self.noble_ids = torch.full((B, MAX_NOBLE_SLOTS), -1, dtype=torch.int8, device=d)
        self.tokens = torch.zeros((B, P, 6), dtype=torch.int8, device=d)
        self.bonuses = torch.zeros((B, P, NUM_COLORS), dtype=torch.int8, device=d)
        self.reserved = torch.full((B, P, MAX_RESERVED), -1, dtype=torch.int16, device=d)
        self.reserved_hidden = torch.zeros((B, P, MAX_RESERVED), dtype=torch.bool, device=d)
        self.points = torch.zeros((B, P), dtype=torch.int8, device=d)
        self.nobles_claimed = torch.zeros((B, P), dtype=torch.int8, device=d)
        self.active_mask = torch.zeros((B, P), dtype=torch.bool, device=d)
        self.current_player = torch.zeros((B,), dtype=torch.int8, device=d)
        self.first_player = torch.zeros((B,), dtype=torch.int8, device=d)
        self.phase = torch.zeros((B,), dtype=torch.int8, device=d)
        self.last_trigger = torch.full((B,), -1, dtype=torch.int8, device=d)
        self.turns_since_trigger = torch.zeros((B,), dtype=torch.int8, device=d)
        self.ended = torch.zeros((B,), dtype=torch.bool, device=d)

    def reset_all(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.generator.manual_seed(seed)
        B = self.batch_size
        P = MAX_PLAYERS
        nP = self.num_players
        d = self.device

        # Token supply
        supply = C.token_supply_for_players(nP)
        gp = torch.tensor(supply, dtype=torch.int8).to(d)
        self.gem_pool[:] = gp.unsqueeze(0).expand(B, -1)

        # Decks per tier; shuffled independently per game via CPU generator
        level_to_cards = [[], [], []]
        for c in C.CARDS:
            level_to_cards[c.level - 1].append(c.card_id)
        for t in range(NUM_TIERS):
            ids = torch.tensor(level_to_cards[t], dtype=torch.int16)
            L = ids.shape[0]
            perms = torch.zeros((B, L), dtype=torch.int16)
            for i in range(B):
                perm_idx = torch.randperm(L, generator=self.generator)
                perms[i] = ids[perm_idx]
            self.deck_perm[:, t, :L] = perms.to(d)
            if L < MAX_TIER_CARDS:
                self.deck_perm[:, t, L:] = -1
            # Reveal top 4 into grid
            for s in range(NUM_GRID_SLOTS):
                self.grid_card[:, t, s] = perms[:, L - 1 - s].to(d)
            self.deck_top[:, t] = (L - NUM_GRID_SLOTS)

        # Nobles
        n_nobles = C.num_nobles_for_players(nP)
        self.noble_ids[:] = -1
        all_nobles = torch.arange(NUM_NOBLES, dtype=torch.int8)
        for i in range(B):
            perm_idx = torch.randperm(NUM_NOBLES, generator=self.generator)
            chosen = all_nobles[perm_idx][:n_nobles]
            self.noble_ids[i, :n_nobles] = chosen.to(d)

        # Players
        self.tokens.zero_()
        self.bonuses.zero_()
        self.reserved.fill_(-1)
        self.reserved_hidden.zero_()
        self.points.zero_()
        self.nobles_claimed.zero_()
        self.active_mask.zero_()
        self.active_mask[:, :nP] = True

        first_players = torch.randint(0, nP, (B,), generator=self.generator).to(torch.int8)
        self.current_player[:] = first_players.to(d)
        self.first_player[:] = first_players.to(d)
        self.phase.zero_()
        self.last_trigger.fill_(-1)
        self.turns_since_trigger.zero_()
        self.ended.zero_()

    def _new_shell(self, batch_size: int) -> "BatchedEngine":
        new = BatchedEngine.__new__(BatchedEngine)
        new.batch_size = batch_size
        new.num_players = self.num_players
        new.device = self.device
        new.generator = self.generator
        new.tables = self.tables
        new._b_range = torch.arange(batch_size, dtype=torch.long, device=self.device)
        return new

    def clone(self) -> "BatchedEngine":
        """Return a mutable copy of the full batched engine state."""
        return self.index_select(self._b_range)

    def index_select(self, idx: torch.Tensor) -> "BatchedEngine":
        """Return a new batched engine containing only the selected games."""
        idx = idx.to(device=self.device, dtype=torch.long)
        new = self._new_shell(int(idx.numel()))
        for name in _STATE_TENSOR_ATTRS:
            setattr(new, name, getattr(self, name).index_select(0, idx))
        return new

    def repeat_interleave(self, repeats: int) -> "BatchedEngine":
        """Repeat each game state `repeats` times along the batch dimension."""
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")
        new = self._new_shell(self.batch_size * repeats)
        for name in _STATE_TENSOR_ATTRS:
            setattr(
                new,
                name,
                getattr(self, name).repeat_interleave(repeats, dim=0),
            )
        return new

    def _gather_player(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather the active player's row from (B, P, ...) -> (B, ...)."""
        B = self.batch_size
        cp = self.current_player.to(torch.long).unsqueeze(-1)
        if tensor.dim() == 2:
            return tensor.gather(1, cp).squeeze(1)
        if tensor.dim() == 3:
            cp2 = cp.unsqueeze(-1).expand(-1, -1, tensor.shape[2])
            return tensor.gather(1, cp2).squeeze(1)
        raise ValueError(tensor.dim())

    def _scatter_player(self, tensor: torch.Tensor, values: torch.Tensor) -> None:
        B = self.batch_size
        cp = self.current_player.to(torch.long).unsqueeze(-1)
        if tensor.dim() == 2:
            tensor.scatter_(1, cp, values.unsqueeze(1))
        elif tensor.dim() == 3:
            cp2 = cp.unsqueeze(-1).expand(-1, -1, tensor.shape[2])
            tensor.scatter_(1, cp2, values.unsqueeze(1))
        else:
            raise ValueError(tensor.dim())

    def _compute_purchase_for_ids(
        self, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a tensor of card_ids shape (B, K) (K candidates per game),
        return (affordable (B,K) bool, spent_colored (B,K,5) int16, spent_gold (B,K) int16).
        `ids == -1` means empty/invalid and is treated as unaffordable.
        """
        B = self.batch_size
        K = ids.shape[-1] if ids.dim() >= 2 else 1
        if ids.dim() == 1:
            ids = ids.unsqueeze(-1)
        cost = _gather_card_row(self.tables["card_cost_pad"], ids).to(torch.int16)  # (B,K,5)
        cp_bonus = self._gather_player(self.bonuses).to(torch.int16)  # (B,5)
        net = (cost - cp_bonus.unsqueeze(1)).clamp_min(0)  # (B,K,5)
        cp_tokens = self._gather_player(self.tokens).to(torch.int16)  # (B,6)
        colored_tokens = cp_tokens[:, :NUM_COLORS]  # (B,5)
        gold_tokens = cp_tokens[:, GOLD]  # (B,)
        spent_colored = torch.minimum(net, colored_tokens.unsqueeze(1))  # (B,K,5)
        deficit = (net - spent_colored).sum(dim=-1)  # (B,K)
        spent_gold = deficit
        affordable = deficit <= gold_tokens.unsqueeze(-1)
        invalid = ids < 0
        affordable = affordable & (~invalid)
        return affordable, spent_colored, spent_gold

    def legal_action_mask(self) -> torch.Tensor:
        """Returns (B, NUM_ACTIONS) bool tensor of legal actions in the current phase."""
        B = self.batch_size
        NA = A.NUM_ACTIONS
        mask = torch.zeros((B, NA), dtype=torch.bool, device=self.device)

        not_ended = ~self.ended
        phase0 = not_ended & (self.phase == 0)
        phase1 = not_ended & (self.phase == 1)
        phase2 = not_ended & (self.phase == 2)

        # Take3: combo c is legal if all colors in combo have gem_pool > 0 OR edge case
        combos = self.tables["take3_combos"]  # (10,3)
        pool_colors = self.gem_pool[:, :NUM_COLORS]  # (B,5)
        pool_has = pool_colors > 0  # (B,5)
        combo_has_all = pool_has[:, combos].all(dim=-1)  # (B,10)
        mask[:, A.TAKE3_BASE : A.TAKE3_BASE + A.TAKE3_COUNT] = combo_has_all & phase0.unsqueeze(-1)

        # Edge case: fewer than 3 piles non-empty. Allow any combo whose available
        # set is a subset of the combo's colors. Computed branchlessly every call.
        num_nonempty = pool_has.sum(dim=-1)  # (B,)
        edge_games = (num_nonempty < 3) & phase0
        combo_membership = self.tables["combo_membership"]  # (10,5) bool
        # bad[b,i,c] = pool_has[b,c] AND NOT combo_membership[i,c]
        bad = pool_has.unsqueeze(1) & (~combo_membership.unsqueeze(0))  # (B,10,5)
        combo_valid = ~bad.any(dim=-1)  # (B,10)
        edge_mask = edge_games.unsqueeze(-1) & combo_valid
        mask[:, A.TAKE3_BASE : A.TAKE3_BASE + A.TAKE3_COUNT] |= edge_mask

        # Take2: gem_pool[c] >= 4 and phase0
        mask[:, A.TAKE2_BASE : A.TAKE2_BASE + A.TAKE2_COUNT] = (
            (pool_colors >= 4) & phase0.unsqueeze(-1)
        )

        # Reserve grid/blind: player reserved_count < 3
        cp_reserved = self._gather_player(self.reserved)  # (B, 3)
        can_reserve = ((cp_reserved >= 0).sum(dim=-1) < MAX_RESERVED) & phase0  # (B,)
        # reserve grid: grid card present
        grid_flat = self.grid_card.reshape(B, NUM_TIERS * NUM_GRID_SLOTS)  # (B,12)
        mask[:, A.RESERVE_GRID_BASE : A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT] = (
            (grid_flat >= 0) & can_reserve.unsqueeze(-1)
        )
        # reserve blind: deck has cards
        deck_has = self.deck_top > 0  # (B,3)
        mask[:, A.RESERVE_BLIND_BASE : A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT] = (
            deck_has & can_reserve.unsqueeze(-1)
        )

        # Buy grid: affordable & phase0
        grid_ids_flat = self.grid_card.reshape(B, NUM_TIERS * NUM_GRID_SLOTS).to(torch.long)
        afford_grid, _, _ = self._compute_purchase_for_ids(grid_ids_flat)  # (B,12)
        mask[:, A.BUY_GRID_BASE : A.BUY_GRID_BASE + A.BUY_GRID_COUNT] = (
            afford_grid & phase0.unsqueeze(-1)
        )

        # Buy reserved
        afford_res, _, _ = self._compute_purchase_for_ids(cp_reserved.to(torch.long))  # (B,3)
        mask[:, A.BUY_RESERVED_BASE : A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT] = (
            afford_res & phase0.unsqueeze(-1)
        )

        # Pass only if nothing else legal
        main_slice = mask[:, A.TAKE3_BASE : A.MAIN_ACTIONS_END]
        nothing_legal = ~main_slice.any(dim=-1)
        mask[:, A.PASS_ACTION] = nothing_legal & phase0

        # Discard: token_k > 0 & phase1
        cp_tokens = self._gather_player(self.tokens)  # (B,6)
        mask[:, A.DISCARD_BASE : A.DISCARD_BASE + A.DISCARD_COUNT] = (
            (cp_tokens > 0) & phase1.unsqueeze(-1)
        )

        # Pick noble: noble slot occupied and bonuses satisfy requirement
        cp_bonus = self._gather_player(self.bonuses)  # (B,5)
        noble_req = _gather_card_row(self.tables["noble_req_pad"], self.noble_ids.to(torch.long))
        # (B,5,5) - bonuses satisfies each slot's req
        slot_ok = (cp_bonus.unsqueeze(1) >= noble_req).all(dim=-1)  # (B,5)
        slot_present = self.noble_ids >= 0  # (B,5)
        mask[:, A.PICK_NOBLE_BASE : A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT] = (
            slot_ok & slot_present & phase2.unsqueeze(-1)
        )

        return mask

    # -----------------------------------------------------------------------
    # Branchless, fixed-shape helpers
    #
    # All helpers take a per-game boolean mask of shape (B,) instead of a
    # variable-length index tensor. They always run the full-batch tensor ops
    # and use torch.where / mask-multiplied deltas to only affect the selected
    # games. This removes all Python-side `.any()` branches and dynamic-shape
    # `.nonzero()` indexing from the hot path and keeps the implementation
    # predictable under batching.
    # -----------------------------------------------------------------------

    def _refill_grid_mask(self, mask: torch.Tensor, tier: torch.Tensor, slot: torch.Tensor) -> None:
        """For games in `mask`, refill grid[b, tier[b], slot[b]] from deck tier[b].

        `tier` and `slot` are full-batch (B,) long tensors; for games outside
        the mask they may hold arbitrary safe values (e.g. 0) and are ignored.
        """
        B = self.batch_size
        b_range = self._b_range
        top = self.deck_top[b_range, tier].to(torch.long)  # (B,)
        has_card = mask & (top > 0)
        new_top = torch.where(has_card, top - 1, top)
        # Fetch candidate next card id from deck permutation.
        perm_flat = self.deck_perm.view(B, NUM_TIERS * MAX_TIER_CARDS)
        perm_idx = tier * MAX_TIER_CARDS + new_top.clamp_min(0)
        cand_card = perm_flat.gather(1, perm_idx.unsqueeze(-1)).squeeze(-1)  # int16
        new_card = torch.where(
            has_card,
            cand_card,
            torch.full_like(cand_card, -1),
        )
        # deck_top update (only for games with a card to pop)
        self.deck_top[b_range, tier] = torch.where(
            has_card, new_top.to(self.deck_top.dtype), self.deck_top[b_range, tier]
        )
        # grid_card update (only for games in mask; otherwise keep current)
        cur_grid = self.grid_card[b_range, tier, slot]
        self.grid_card[b_range, tier, slot] = torch.where(mask, new_card, cur_grid)

    def _claim_noble_mask(self, mask: torch.Tensor, slot_idx: torch.Tensor) -> None:
        """Claim the noble at `slot_idx[b]` for games in `mask`.

        `slot_idx` is a full-batch (B,) long tensor (safe default 0 outside mask).
        Points +3, nobles_claimed +1, noble_ids[slot] cleared. Mask-gated.
        """
        b_range = self._b_range
        cp = self.current_player.to(torch.long)
        delta = mask.to(self.points.dtype)
        self.points[b_range, cp] = self.points[b_range, cp] + delta * 3
        self.nobles_claimed[b_range, cp] = self.nobles_claimed[b_range, cp] + delta
        cur_nid = self.noble_ids[b_range, slot_idx]
        self.noble_ids[b_range, slot_idx] = torch.where(
            mask, torch.full_like(cur_nid, -1), cur_nid
        )

    def _end_turn_mask(self, mask: torch.Tensor) -> None:
        """Rotate turn for games in `mask`, handle trigger, and set ended if full round elapsed."""
        nP = self.num_players

        pts = self.points[self._b_range, self.current_player.to(torch.long)]
        no_trigger = self.last_trigger < 0
        just_hit = pts >= WINNING_POINTS
        new_trigger = mask & no_trigger & just_hit
        self.last_trigger = torch.where(new_trigger, self.current_player, self.last_trigger)

        # Reset phase to 0 for ending games.
        self.phase = torch.where(mask, torch.zeros_like(self.phase), self.phase)
        # Rotate current_player for ending games.
        new_cp = ((self.current_player.to(torch.long) + 1) % nP).to(torch.int8)
        self.current_player = torch.where(mask, new_cp, self.current_player)

        # Increment turns_since_trigger for ending games that have a trigger set.
        has_trigger = self.last_trigger >= 0
        inc = (mask & has_trigger).to(self.turns_since_trigger.dtype)
        self.turns_since_trigger = self.turns_since_trigger + inc

        # Game ends when current_player wraps back to first_player, meaning
        # the round is complete and everyone has had equal turns. This correctly
        # handles the case where the last player in a round triggers — the
        # game ends immediately since the round is already complete.
        stop_at = self.first_player
        end_now = mask & has_trigger & (self.current_player == stop_at)
        self.ended = self.ended | end_now

    def _nobles_and_end_mask(self, mask: torch.Tensor) -> None:
        """For games in `mask`: check qualifying nobles; 2+ -> phase 2; 1 -> auto-claim + end; 0 -> end."""
        B = self.batch_size
        cp = self.current_player.to(torch.long)
        bonuses_cp = self.bonuses[self._b_range, cp]  # (B,5)
        noble_req = _gather_card_row(
            self.tables["noble_req_pad"], self.noble_ids.to(torch.long)
        )  # (B,5,5)
        slot_ok = (bonuses_cp.unsqueeze(1) >= noble_req).all(dim=-1)  # (B,5)
        slot_present = self.noble_ids >= 0
        qualify = slot_ok & slot_present  # (B,5)
        qcount = qualify.sum(dim=-1)  # (B,)

        two_plus = mask & (qcount >= 2)
        one_q = mask & (qcount == 1)
        self.phase = torch.where(two_plus, torch.full_like(self.phase, 2), self.phase)

        # Auto-claim for one_q games (safe default slot 0 otherwise).
        slot_idx = qualify.to(torch.int64).argmax(dim=-1)
        self._claim_noble_mask(one_q, slot_idx)

        # End turn for games where not two_plus (i.e., zero qualifying or just auto-claimed one).
        self._end_turn_mask(mask & ~two_plus)

    def _advance_after_action_mask(self, mask: torch.Tensor) -> None:
        """For games in `mask` that just completed a main action, do over-limit / noble / end-turn flow."""
        cp = self.current_player.to(torch.long)
        total = self.tokens[self._b_range, cp].to(torch.int16).sum(dim=-1)  # (B,)
        over_limit = mask & (total > TOKEN_LIMIT)
        # over-limit -> phase 1
        self.phase = torch.where(over_limit, torch.full_like(self.phase, 1), self.phase)
        self._nobles_and_end_mask(mask & ~over_limit)

    def _pay_and_gain_mask(
        self, mask: torch.Tensor, cid: torch.Tensor
    ) -> None:
        """For games in `mask`, pay card cost (colored + gold) and credit bonus/points for cid.

        `cid` is (B,) long; may hold -1 or arbitrary outside mask (treated as zero-cost).
        """
        B = self.batch_size
        b_range = self._b_range
        cp = self.current_player.to(torch.long)
        cost = _gather_card_row(self.tables["card_cost_pad"], cid).to(torch.int16)  # (B,5)
        bonus = self.bonuses[b_range, cp].to(torch.int16)  # (B,5)
        tokens_row = self.tokens[b_range, cp].to(torch.int16)  # (B,6)
        colored = tokens_row[:, :NUM_COLORS]
        net = (cost - bonus).clamp_min(0)  # (B,5)
        spent_colored = torch.minimum(net, colored)  # (B,5)
        spent_gold = (net - spent_colored).sum(dim=-1)  # (B,)

        m_i16 = mask.to(torch.int16)
        spent_colored = spent_colored * m_i16.unsqueeze(-1)
        spent_gold = spent_gold * m_i16

        # Update pool and tokens per color branchlessly.
        sc_i8 = spent_colored.to(torch.int8)
        for c in range(NUM_COLORS):
            self.gem_pool[:, c] = self.gem_pool[:, c] + sc_i8[:, c]
            self.tokens[b_range, cp, c] = self.tokens[b_range, cp, c] - sc_i8[:, c]
        sg_i8 = spent_gold.to(torch.int8)
        self.gem_pool[:, GOLD] = self.gem_pool[:, GOLD] + sg_i8
        self.tokens[b_range, cp, GOLD] = self.tokens[b_range, cp, GOLD] - sg_i8

        # Credit bonus color and points.
        bonus_color = self.tables["card_bonus_pad"][cid.clamp_min(0)].to(torch.long)  # (B,)
        m_i8 = mask.to(self.bonuses.dtype)
        self.bonuses[b_range, cp, bonus_color] = (
            self.bonuses[b_range, cp, bonus_color] + m_i8
        )
        pts = self.tables["card_points_pad"][cid.clamp_min(0)].to(self.points.dtype) * mask.to(
            self.points.dtype
        )
        self.points[b_range, cp] = self.points[b_range, cp] + pts

    def apply(self, actions: torch.Tensor) -> None:
        self.apply_python(actions)

    def apply_python(self, actions: torch.Tensor) -> None:
        """Apply one action per game in a fully branchless manner.

        `actions` shape (B,) int64 on self.device. Assumes actions are legal
        (callers should sample from the legal mask). No host synchronization.
        """
        assert actions.shape == (self.batch_size,)
        B = self.batch_size
        P = MAX_PLAYERS
        b_range = self._b_range
        a = actions.to(torch.long)
        alive = ~self.ended  # (B,)
        cp = self.current_player.to(torch.long)  # (B,)

        # ----- Per-action-class masks (all shape (B,)) -----
        m_take3 = alive & (a >= A.TAKE3_BASE) & (a < A.TAKE3_BASE + A.TAKE3_COUNT)
        m_take2 = alive & (a >= A.TAKE2_BASE) & (a < A.TAKE2_BASE + A.TAKE2_COUNT)
        m_rg = alive & (a >= A.RESERVE_GRID_BASE) & (a < A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT)
        m_rb = alive & (a >= A.RESERVE_BLIND_BASE) & (a < A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT)
        m_bg = alive & (a >= A.BUY_GRID_BASE) & (a < A.BUY_GRID_BASE + A.BUY_GRID_COUNT)
        m_br = alive & (a >= A.BUY_RESERVED_BASE) & (a < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT)
        m_pass = alive & (a == A.PASS_ACTION)
        m_disc = alive & (a >= A.DISCARD_BASE) & (a < A.DISCARD_BASE + A.DISCARD_COUNT)
        m_pn = alive & (a >= A.PICK_NOBLE_BASE) & (a < A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT)
        m_main = m_take3 | m_take2 | m_rg | m_rb | m_bg | m_br | m_pass
        m_res = m_rg | m_rb
        m_buy = m_bg | m_br
        m_refill = m_rg | m_bg

        zero_l = torch.zeros_like(a)

        # ----- TAKE3 -----
        combo_idx = torch.where(m_take3, a - A.TAKE3_BASE, zero_l)
        combos = self.tables["take3_combos"]  # (10,3) int64
        chosen3 = combos[combo_idx]  # (B,3)
        for ci in range(3):
            c = chosen3[:, ci]  # (B,)
            pool_val = self.gem_pool[b_range, c]
            transfer = (m_take3 & (pool_val > 0)).to(self.gem_pool.dtype)
            self.gem_pool[b_range, c] = self.gem_pool[b_range, c] - transfer
            self.tokens[b_range, cp, c] = self.tokens[b_range, cp, c] + transfer

        # ----- TAKE2 -----
        c_t2 = torch.where(m_take2, a - A.TAKE2_BASE, zero_l)
        delta2 = m_take2.to(self.gem_pool.dtype) * 2
        self.gem_pool[b_range, c_t2] = self.gem_pool[b_range, c_t2] - delta2
        self.tokens[b_range, cp, c_t2] = self.tokens[b_range, cp, c_t2] + delta2

        # ----- Compute common reserve/buy parameters -----
        # Tier/slot for RG and BG.
        x_rg = torch.where(m_rg, a - A.RESERVE_GRID_BASE, zero_l)
        tier_rg = x_rg // NUM_GRID_SLOTS
        slot_rg = x_rg % NUM_GRID_SLOTS
        x_bg = torch.where(m_bg, a - A.BUY_GRID_BASE, zero_l)
        tier_bg = x_bg // NUM_GRID_SLOTS
        slot_bg = x_bg % NUM_GRID_SLOTS

        # Card id from grid for RG/BG.
        cid_rg = self.grid_card[b_range, tier_rg, slot_rg].to(torch.long)
        cid_bg = self.grid_card[b_range, tier_bg, slot_bg].to(torch.long)

        # Tier for RB and top-card lookup.
        tier_rb = torch.where(m_rb, a - A.RESERVE_BLIND_BASE, zero_l)
        top_rb = self.deck_top[b_range, tier_rb].to(torch.long)
        # new_top_rb = top_rb - 1 (guarded to >= 0 for safety outside mask)
        new_top_rb = (top_rb - 1).clamp_min(0)
        cid_rb = self.deck_perm[b_range, tier_rb, new_top_rb].to(torch.long)

        # ----- RESERVE (grid or blind) -----
        # cid for reserve: cid_rg if m_rg, cid_rb if m_rb, else -1.
        neg_one = torch.full_like(cid_rg, -1)
        cid_res = torch.where(m_rg, cid_rg, torch.where(m_rb, cid_rb, neg_one))
        # First empty reserved slot (argmax over empty flag).
        reserved_rows = self.reserved[b_range, cp]  # (B,3)
        empty = reserved_rows < 0
        rs_res = empty.to(torch.int64).argmax(dim=-1)  # (B,)
        cur_res = self.reserved[b_range, cp, rs_res].to(torch.long)
        self.reserved[b_range, cp, rs_res] = torch.where(
            m_res, cid_res, cur_res
        ).to(self.reserved.dtype)
        cur_hidden = self.reserved_hidden[b_range, cp, rs_res]
        # Hidden=True for blind reserve, False for grid reserve; keep current otherwise.
        hidden_new = torch.where(m_rb, torch.ones_like(cur_hidden), torch.zeros_like(cur_hidden))
        self.reserved_hidden[b_range, cp, rs_res] = torch.where(
            m_res, hidden_new, cur_hidden
        )
        # Gold token if available.
        gold_transfer = (m_res & (self.gem_pool[:, GOLD] > 0)).to(self.gem_pool.dtype)
        self.gem_pool[:, GOLD] = self.gem_pool[:, GOLD] - gold_transfer
        self.tokens[b_range, cp, GOLD] = self.tokens[b_range, cp, GOLD] + gold_transfer
        # Deck pop for RB (only if m_rb; otherwise leave deck_top unchanged).
        dec_rb = m_rb.to(self.deck_top.dtype)
        self.deck_top[b_range, tier_rb] = self.deck_top[b_range, tier_rb] - dec_rb

        # ----- BUY (grid or reserved) -----
        rs_br = torch.where(m_br, a - A.BUY_RESERVED_BASE, zero_l)
        cid_br = self.reserved[b_range, cp, rs_br].to(torch.long)
        cid_buy = torch.where(m_bg, cid_bg, torch.where(m_br, cid_br, neg_one))
        self._pay_and_gain_mask(m_buy, cid_buy)
        # Clear reserved slot for BUY_RESERVED.
        cur_br_val = self.reserved[b_range, cp, rs_br].to(torch.long)
        self.reserved[b_range, cp, rs_br] = torch.where(
            m_br, torch.full_like(cur_br_val, -1), cur_br_val
        ).to(self.reserved.dtype)
        cur_br_hidden = self.reserved_hidden[b_range, cp, rs_br]
        self.reserved_hidden[b_range, cp, rs_br] = torch.where(
            m_br, torch.zeros_like(cur_br_hidden), cur_br_hidden
        )

        # ----- Grid refill for RG + BG -----
        # tier/slot for refill: use RG's if m_rg, else BG's if m_bg, else default 0.
        tier_ref = torch.where(m_rg, tier_rg, torch.where(m_bg, tier_bg, zero_l))
        slot_ref = torch.where(m_rg, slot_rg, torch.where(m_bg, slot_bg, zero_l))
        self._refill_grid_mask(m_refill, tier_ref, slot_ref)

        # ----- DISCARD (phase 1) -----
        c_disc = torch.where(m_disc, a - A.DISCARD_BASE, zero_l)
        delta_d = m_disc.to(self.tokens.dtype)
        self.tokens[b_range, cp, c_disc] = self.tokens[b_range, cp, c_disc] - delta_d
        self.gem_pool[b_range, c_disc] = self.gem_pool[b_range, c_disc] + delta_d

        # ----- PICK_NOBLE (phase 2) -----
        ns = torch.where(m_pn, a - A.PICK_NOBLE_BASE, zero_l)
        self._claim_noble_mask(m_pn, ns)

        # ----- Phase transitions (branchless) -----
        # Main actions: over-limit check then nobles/end-turn.
        self._advance_after_action_mask(m_main)
        # Discards: if tokens <= TOKEN_LIMIT now, proceed with noble check + end turn.
        total_after = self.tokens[b_range, cp].to(torch.int16).sum(dim=-1)
        self._nobles_and_end_mask(m_disc & (total_after <= TOKEN_LIMIT))
        # Noble picks: end turn immediately.
        self._end_turn_mask(m_pn)

def snapshot_from_single(
    engine: BatchedEngine, games: list[int] = None
):
    """Debug helper: returns a list of dicts matching single_engine state fields."""
    from . import single_engine as SE

    if games is None:
        games = list(range(engine.batch_size))
    snaps = []
    for b in games:
        nP = engine.num_players
        gs = SE.GameState(
            num_players=nP,
            gem_pool=engine.gem_pool[b].tolist(),
            grid=[engine.grid_card[b, t].tolist() for t in range(NUM_TIERS)],
            decks=[
                [
                    int(x)
                    for x in engine.deck_perm[b, t, : int(engine.deck_top[b, t])].tolist()
                    if x >= 0
                ]
                for t in range(NUM_TIERS)
            ],
            nobles=engine.noble_ids[b].tolist(),
            players=[
                SE.PlayerState(
                    tokens=engine.tokens[b, p].tolist(),
                    bonuses=engine.bonuses[b, p].tolist(),
                    reserved=engine.reserved[b, p].tolist(),
                    reserved_hidden=engine.reserved_hidden[b, p].tolist(),
                    points=int(engine.points[b, p]),
                    nobles=int(engine.nobles_claimed[b, p]),
                )
                for p in range(nP)
            ],
            current_player=int(engine.current_player[b]),
            first_player=int(engine.first_player[b]),
            phase=int(engine.phase[b]),
            turn_count=0,
            last_round_trigger_player=int(engine.last_trigger[b]),
            round_actions_since_trigger=int(engine.turns_since_trigger[b]),
            ended=bool(engine.ended[b]),
        )
        snaps.append(gs)
    return snaps
