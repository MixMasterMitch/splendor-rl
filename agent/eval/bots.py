"""Reference opponent bots that operate on the batched engine.

All bots expose `choose(engine) -> (B,) int64` actions.

- `RandomBot`: uniformly random over legal actions.
- `HeuristicBot`: simple greedy bot that prefers buying the most valuable
  affordable card, else reserves the best long-term card, else takes tokens
  tilted toward cards it's close to affording.
"""

from __future__ import annotations

import torch

from ..env import actions as A
from ..env import batched_engine as BE


class RandomBot:
    def __init__(self, seed: int = 0) -> None:
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        mask = engine.legal_action_mask()  # (B, NA)
        B, NA = mask.shape
        scores = torch.rand(B, NA, generator=self.gen).to(engine.device)
        scores = scores.masked_fill(~mask, -1.0)
        return scores.argmax(dim=-1)


class HeuristicBot:
    """Greedy scoring:
    1. If any affordable card: buy the one with highest (pv + 0.3*bonus_rarity).
    2. Else if can reserve and reserved<3: reserve a tier-3 visible card with
       the highest PV if affordable-with-1-more-bonus, else pass to tokens.
    3. Else take tokens: prefer 3-different that matches cards on grid.
    """

    def __init__(self) -> None:
        pass

    def choose(self, engine: BE.BatchedEngine) -> torch.Tensor:
        mask = engine.legal_action_mask()  # (B, NA)
        device = engine.device
        B, NA = mask.shape
        scores = torch.full((B, NA), -1e9, device=device)

        # Buying: +100 base, + pv * 20
        buy_grid = mask[:, A.BUY_GRID_BASE : A.BUY_GRID_BASE + A.BUY_GRID_COUNT]
        buy_res = mask[:, A.BUY_RESERVED_BASE : A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT]

        grid_ids = engine.grid_card.reshape(B, A.BUY_GRID_COUNT).to(torch.long)
        cp = engine.current_player.to(torch.long).unsqueeze(-1)
        reserved_rows = engine.reserved.gather(
            1, cp.unsqueeze(-1).expand(-1, -1, BE.MAX_RESERVED)
        ).squeeze(1).to(torch.long)

        pv_pad = torch.cat(
            [engine.tables["card_points"], torch.zeros((1,), dtype=torch.int8, device=device)],
            dim=0,
        ).to(torch.float32)
        grid_pv = pv_pad[torch.where(grid_ids < 0, torch.full_like(grid_ids, 90), grid_ids)]
        res_pv = pv_pad[torch.where(reserved_rows < 0, torch.full_like(reserved_rows, 90), reserved_rows)]

        scores[:, A.BUY_GRID_BASE : A.BUY_GRID_BASE + A.BUY_GRID_COUNT] = torch.where(
            buy_grid, 100.0 + 20.0 * grid_pv, torch.full_like(grid_pv, -1e9)
        )
        scores[:, A.BUY_RESERVED_BASE : A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT] = torch.where(
            buy_res, 100.0 + 20.0 * res_pv, torch.full_like(res_pv, -1e9)
        )

        # Reserving: small positive, prefer tier 3 by PV
        rg_mask = mask[:, A.RESERVE_GRID_BASE : A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT]
        tiers = torch.arange(A.BUY_GRID_COUNT, device=device) // BE.NUM_GRID_SLOTS
        reserve_scores = 20.0 + 2.0 * grid_pv + 3.0 * tiers.to(torch.float32).unsqueeze(0)
        scores[:, A.RESERVE_GRID_BASE : A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT] = torch.where(
            rg_mask, reserve_scores, torch.full_like(reserve_scores, -1e9)
        )

        # Take3: modest score
        t3_mask = mask[:, A.TAKE3_BASE : A.TAKE3_BASE + A.TAKE3_COUNT]
        scores[:, A.TAKE3_BASE : A.TAKE3_BASE + A.TAKE3_COUNT] = torch.where(
            t3_mask, torch.full_like(t3_mask, 10.0, dtype=torch.float32), torch.full_like(t3_mask, -1e9, dtype=torch.float32)
        )
        t2_mask = mask[:, A.TAKE2_BASE : A.TAKE2_BASE + A.TAKE2_COUNT]
        scores[:, A.TAKE2_BASE : A.TAKE2_BASE + A.TAKE2_COUNT] = torch.where(
            t2_mask, torch.full_like(t2_mask, 8.0, dtype=torch.float32), torch.full_like(t2_mask, -1e9, dtype=torch.float32)
        )

        # Reserve blind: very low
        rb_mask = mask[:, A.RESERVE_BLIND_BASE : A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT]
        scores[:, A.RESERVE_BLIND_BASE : A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT] = torch.where(
            rb_mask, torch.full_like(rb_mask, 2.0, dtype=torch.float32), torch.full_like(rb_mask, -1e9, dtype=torch.float32)
        )

        # Discard phase: prefer to discard gold LAST; pick most-abundant non-gold token
        cp_tokens = engine.tokens.gather(
            1, cp.unsqueeze(-1).expand(-1, -1, 6)
        ).squeeze(1).to(torch.float32)
        disc_mask = mask[:, A.DISCARD_BASE : A.DISCARD_BASE + A.DISCARD_COUNT]
        disc_scores = cp_tokens.clone()
        disc_scores[:, 5] = disc_scores[:, 5] - 100.0  # avoid discarding gold
        scores[:, A.DISCARD_BASE : A.DISCARD_BASE + A.DISCARD_COUNT] = torch.where(
            disc_mask, disc_scores, torch.full_like(disc_scores, -1e9)
        )

        # Pick noble: any legal slot (all equally good, take lowest index)
        pn_mask = mask[:, A.PICK_NOBLE_BASE : A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT]
        pn_scores = torch.linspace(50.0, 40.0, A.PICK_NOBLE_COUNT, device=device).unsqueeze(0).expand(B, -1)
        scores[:, A.PICK_NOBLE_BASE : A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT] = torch.where(
            pn_mask, pn_scores, torch.full_like(pn_scores, -1e9)
        )

        # Pass: last resort
        pass_mask = mask[:, A.PASS_ACTION]
        scores[:, A.PASS_ACTION] = torch.where(
            pass_mask, torch.full_like(pass_mask, -100.0, dtype=torch.float32), torch.full_like(pass_mask, -1e9, dtype=torch.float32)
        )

        scores = scores.masked_fill(~mask, -1e9)
        return scores.argmax(dim=-1)
