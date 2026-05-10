"""Policy + value network.

Architecture (V4 — per-PC heads):
- MLP trunk over `global_feat` -> hidden vector h_g (dim H)
- Learned player-count embedding (3×H) added to h_g to condition on game mode
- Per-card MLP embedding of `card_feat` -> h_c (B, N_CARDS, H)
- Cross attention: query = h_g projected, keys/values = h_c. Output: h_attn (B, H)
- Per-PC policy heads (3×): MLP over (h_g + h_attn) producing `NUM_ACTIONS` logits.
  Routed by player count. Masked before softmax.
- Per-PC value heads (3×): MLP over (h_g + h_attn) producing `MAX_PLAYERS` scalars
  representing each seat's predicted final placement score. Routed by player count.

The per-PC heads allow each player count (2p/3p/4p) to develop specialized
strategy without interfering with other modes. The shared trunk learns game
fundamentals that transfer across all player counts.

Old checkpoints with shared heads are auto-migrated on load (see checkpointing.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env import actions as A
from ..env import batched_engine as BE
from . import encoder as ENC

NUM_ACTIONS = A.NUM_ACTIONS


class SplendorNet(nn.Module):
    def __init__(
        self,
        hidden: int = 192,
        arch: str = "attn",
        num_heads: int = 4,
        dropout: float = 0.0,
        compile_forward: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.hidden = hidden
        self.arch = arch
        self._compiled = False
        # Learned player-count embedding: index 0=2p, 1=3p, 2=4p.
        # Added to the global trunk output to condition the representation
        # on game mode before attention and heads.
        self.pc_embed = nn.Embedding(3, hidden)
        if arch == "attn":
            self.g_trunk = nn.Sequential(
                nn.Linear(ENC.D_GLOBAL, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.c_embed = nn.Sequential(
                nn.Linear(ENC.D_CARD, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.post_attn = nn.LayerNorm(hidden)
        elif arch == "flat":
            flat_dim = ENC.D_GLOBAL + ENC.N_CARDS * ENC.D_CARD
            self.flat_trunk = nn.Sequential(
                nn.Linear(flat_dim, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
            )
        else:
            raise ValueError(f"unsupported SplendorNet arch: {arch}")
        # Per-player-count policy and value heads.
        # Each PC gets its own 2-layer MLP so it can specialize strategy
        # without interfering with other PCs. Initialized from the same
        # weights at migration time for warm-start continuity.
        self.policy_heads = nn.ModuleDict({
            "0": nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, NUM_ACTIONS)),
            "1": nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, NUM_ACTIONS)),
            "2": nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, NUM_ACTIONS)),
        })
        self.value_heads = nn.ModuleDict({
            "0": nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, BE.MAX_PLAYERS)),
            "1": nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, BE.MAX_PLAYERS)),
            "2": nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, BE.MAX_PLAYERS)),
        })

        self._compile_mode = compile_mode
        if compile_forward:
            self.enable_compile()

    def enable_compile(self) -> None:
        """Wrap forward with torch.compile after the module is on its target device."""
        if self._compiled:
            return
        # Enable the inductor FX graph cache so subsequent runs with the same
        # model shape skip recompilation.
        import os
        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        # Suppress CUDAGraph dynamic shape warnings — with dynamic=True the
        # compiled kernels handle variable batch sizes correctly; the warning
        # is about CUDAGraph caching overhead which is negligible here.
        import torch._inductor.config
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
        # Allow .item() calls in compiled graphs without graph-break warnings.
        torch._dynamo.config.capture_scalar_outputs = True
        # dynamic=True: compile a single graph that handles any batch size via
        # symbolic shapes. Avoids recompilation thrashing during league selfplay
        # where 24+ opponents each get different-sized subsets per turn.
        compiled = torch.compile(
            self._forward_impl,
            mode=self._compile_mode,
            dynamic=True,
            fullgraph=False,
        )
        compiled_value = torch.compile(
            self._forward_value_impl,
            mode=self._compile_mode,
            dynamic=True,
            fullgraph=False,
        )
        compiled_raw = torch.compile(
            self._forward_raw_impl,
            mode=self._compile_mode,
            dynamic=True,
            fullgraph=False,
        )
        self._compiled_forward = compiled
        self._compiled_value_forward = compiled_value
        self._compiled_raw_forward = compiled_raw
        self._compiled = True

    def forward(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled:
            logits, value = self._compiled_forward(global_feat, card_feat, legal_mask)
            # Compiled paths may reuse output buffers across calls; clone before
            # returning so callers can safely hold these tensors past the next
            # forward.
            return logits.clone(), value.clone()
        return self._forward_impl(global_feat, card_feat, legal_mask)

    def forward_value(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> torch.Tensor:
        if self._compiled:
            value = self._compiled_value_forward(global_feat, card_feat)
            return value.clone()
        return self._forward_value_impl(global_feat, card_feat)

    def forward_raw(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled:
            logits, value = self._compiled_raw_forward(global_feat, card_feat)
            return logits.clone(), value.clone()
        return self._forward_raw_impl(global_feat, card_feat)

    def _combined_features(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> torch.Tensor:
        if self.arch == "attn":
            h_g = self.g_trunk(global_feat)
            # Add learned player-count conditioning.
            # PC one-hot is at global_feat[:, 10:13] (after gems=6, phase_scalar=1, phase_oh=3).
            pc_idx = global_feat[:, 10:13].argmax(dim=-1)  # (B,) in {0,1,2}
            h_g = h_g + self.pc_embed(pc_idx)
            h_c = self.c_embed(card_feat)
            query = h_g.unsqueeze(1)  # (B,1,H)
            attn_out, _ = self.attn(query, h_c, h_c, need_weights=False)
            h_a = self.post_attn(attn_out.squeeze(1))
            return torch.cat([h_g, h_a], dim=-1)
        flat_input = torch.cat(
            [global_feat, card_feat.reshape(card_feat.shape[0], -1)], dim=-1
        )
        h = self.flat_trunk(flat_input)
        # Add learned player-count conditioning for flat arch too.
        pc_idx = global_feat[:, 10:13].argmax(dim=-1)
        # For flat arch, hidden*2 output; add pc_embed to first half.
        pc_vec = self.pc_embed(pc_idx)
        h[:, :self.hidden] = h[:, :self.hidden] + pc_vec
        return h

    def _pc_idx_from_global(self, global_feat: torch.Tensor) -> torch.Tensor:
        """Extract player-count index (0=2p, 1=3p, 2=4p) from global_feat."""
        return global_feat[:, 10:13].argmax(dim=-1)  # (B,)

    def _forward_impl(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits (B, NUM_ACTIONS), value (B, MAX_PLAYERS))."""
        logits, value = self._forward_raw_impl(global_feat, card_feat)
        mask_fill = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~legal_mask, mask_fill)
        return masked_logits, value

    def _forward_raw_impl(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = self._combined_features(global_feat, card_feat)
        pc_idx = self._pc_idx_from_global(global_feat)

        # Route through PC-specific heads. In a batch all samples have the
        # same PC (selfplay runs one PC per iteration), so we fast-path that
        # common case. Mixed-PC batches (e.g. replay buffer) use scatter.
        unique_pcs = pc_idx.unique()
        if unique_pcs.numel() == 1:
            pc = str(int(unique_pcs.item()))
            logits = self.policy_heads[pc](combined)
            value = torch.tanh(self.value_heads[pc](combined))
        else:
            B = combined.shape[0]
            # Run all heads and gather — avoids dtype issues with AMP and
            # graph breaks from indexed assignment.
            all_logits = torch.stack([self.policy_heads[str(i)](combined) for i in range(3)], dim=0)  # (3, B, A)
            all_values = torch.stack([torch.tanh(self.value_heads[str(i)](combined)) for i in range(3)], dim=0)  # (3, B, P)
            # Gather by pc_idx: (B,) -> index into dim 0
            idx_p = pc_idx.unsqueeze(-1).expand(-1, all_logits.shape[-1]).unsqueeze(0)  # (1, B, A)
            logits = all_logits.gather(0, idx_p).squeeze(0)  # (B, A)
            idx_v = pc_idx.unsqueeze(-1).expand(-1, all_values.shape[-1]).unsqueeze(0)  # (1, B, P)
            value = all_values.gather(0, idx_v).squeeze(0)  # (B, P)
        return logits, value

    def _forward_value_impl(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> torch.Tensor:
        combined = self._combined_features(global_feat, card_feat)
        pc_idx = self._pc_idx_from_global(global_feat)

        unique_pcs = pc_idx.unique()
        if unique_pcs.numel() == 1:
            pc = str(int(unique_pcs.item()))
            return torch.tanh(self.value_heads[pc](combined))
        else:
            all_values = torch.stack([torch.tanh(self.value_heads[str(i)](combined)) for i in range(3)], dim=0)
            idx_v = pc_idx.unsqueeze(-1).expand(-1, all_values.shape[-1]).unsqueeze(0)
            return all_values.gather(0, idx_v).squeeze(0)

    def net_device(self) -> torch.device:
        return next(self.parameters()).device

    def inference_from_encoded(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run policy+value inference from pre-encoded tensors."""
        return self.forward(global_feat, card_feat, legal_mask)

    def value_from_encoded(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Run value-only inference from pre-encoded tensors."""
        return self.forward_value(global_feat, card_feat)

    def raw_from_encoded(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run unmasked policy+value inference from pre-encoded tensors."""
        return self.forward_raw(global_feat, card_feat)

    @torch.no_grad()
    def inference(
        self,
        engine: BE.BatchedEngine,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the engine state and run the network on the engine device."""
        g, c, _, legal = ENC.encode_state_with_legal(engine)
        if engine.device != self.net_device():
            raise ValueError(
                f"net and engine must be on the same device, got net={self.net_device()} "
                f"engine={engine.device}"
            )
        logits, value = self.inference_from_encoded(g, c, legal)
        return logits, value, legal

    @torch.no_grad()
    def value_inference(self, engine: BE.BatchedEngine) -> torch.Tensor:
        """Encode the engine state and run only the value head."""
        g, c, _ = ENC.encode_state(engine)
        if engine.device != self.net_device():
            raise ValueError(
                f"net and engine must be on the same device, got net={self.net_device()} "
                f"engine={engine.device}"
            )
        return self.value_from_encoded(g, c)
