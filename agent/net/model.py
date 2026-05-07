"""Policy + value network.

Architecture (V1):
- MLP trunk over `global_feat` -> hidden vector h_g (dim H)
- Per-card MLP embedding of `card_feat` -> h_c (B, N_CARDS, H)
- Cross attention: query = h_g projected, keys/values = h_c. Output: h_attn (B, H)
- Policy head: MLP over (h_g + h_attn) producing `NUM_ACTIONS` logits. Masked
  before softmax.
- Value head: MLP over (h_g + h_attn) producing `MAX_PLAYERS` scalars representing
  each seat's predicted final placement score (seat 0 = current player, seats
  rotated as in encoder). Inactive seats' targets are masked at loss time.
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
        self.policy_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, BE.MAX_PLAYERS),
        )

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
        compiled = torch.compile(
            self._forward_impl,
            mode=self._compile_mode,
            dynamic=False,
            fullgraph=False,
        )
        compiled_value = torch.compile(
            self._forward_value_impl,
            mode=self._compile_mode,
            dynamic=False,
            fullgraph=False,
        )
        compiled_raw = torch.compile(
            self._forward_raw_impl,
            mode=self._compile_mode,
            dynamic=False,
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
            h_c = self.c_embed(card_feat)
            query = h_g.unsqueeze(1)  # (B,1,H)
            attn_out, _ = self.attn(query, h_c, h_c, need_weights=False)
            h_a = self.post_attn(attn_out.squeeze(1))
            return torch.cat([h_g, h_a], dim=-1)
        flat_input = torch.cat(
            [global_feat, card_feat.reshape(card_feat.shape[0], -1)], dim=-1
        )
        return self.flat_trunk(flat_input)

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
        logits = self.policy_head(combined)
        value = torch.tanh(self.value_head(combined))
        return logits, value

    def _forward_value_impl(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
    ) -> torch.Tensor:
        combined = self._combined_features(global_feat, card_feat)
        return torch.tanh(self.value_head(combined))

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
