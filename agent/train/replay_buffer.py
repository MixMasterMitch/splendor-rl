"""Fixed-capacity circular replay buffer for (state, policy, value) triples.

The buffer stores encoded features rather than raw engine state to keep memory
predictable:
- global_feat (N, D_GLOBAL) float32 on CPU, float16 otherwise
- card_feat   (N, N_CARDS, D_CARD) float32 on CPU, float16 otherwise
- legal_mask  (N, NUM_ACTIONS) bool
- policy      (N, NUM_ACTIONS) float32 on CPU, float16 otherwise
- value       (N, MAX_PLAYERS) float32 on CPU, float16 otherwise

Writes append in circular fashion; reads sample without replacement within a
batch. The buffer is independent of torch device.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device | str = "cpu",
        storage_dtype: torch.dtype | None = None,
    ):
        self.capacity = capacity
        self.device = torch.device(device)
        self.storage_dtype = (
            storage_dtype
            if storage_dtype is not None
            else (torch.float32 if self.device.type == "cpu" else torch.float16)
        )
        self.size = 0
        self.next_idx = 0
        self.global_feat = torch.zeros(
            (capacity, ENC.D_GLOBAL), dtype=self.storage_dtype, device=self.device
        )
        self.card_feat = torch.zeros(
            (capacity, ENC.N_CARDS, ENC.D_CARD), dtype=self.storage_dtype, device=self.device
        )
        self.legal_mask = torch.zeros(
            (capacity, A.NUM_ACTIONS), dtype=torch.bool, device=self.device
        )
        self.policy = torch.zeros(
            (capacity, A.NUM_ACTIONS), dtype=self.storage_dtype, device=self.device
        )
        self.value = torch.zeros(
            (capacity, BE.MAX_PLAYERS), dtype=self.storage_dtype, device=self.device
        )

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        global_feat: torch.Tensor,
        card_feat: torch.Tensor,
        legal_mask: torch.Tensor,
        policy: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        n = global_feat.shape[0]
        gf = global_feat.to(device=self.device, dtype=self.storage_dtype)
        cf = card_feat.to(device=self.device, dtype=self.storage_dtype)
        lm = legal_mask.to(self.device)
        pl = policy.to(device=self.device, dtype=self.storage_dtype)
        vl = value.to(device=self.device, dtype=self.storage_dtype)

        end = self.next_idx + n
        if end <= self.capacity:
            self.global_feat[self.next_idx : end] = gf
            self.card_feat[self.next_idx : end] = cf
            self.legal_mask[self.next_idx : end] = lm
            self.policy[self.next_idx : end] = pl
            self.value[self.next_idx : end] = vl
        else:
            first = self.capacity - self.next_idx
            self.global_feat[self.next_idx :] = gf[:first]
            self.card_feat[self.next_idx :] = cf[:first]
            self.legal_mask[self.next_idx :] = lm[:first]
            self.policy[self.next_idx :] = pl[:first]
            self.value[self.next_idx :] = vl[:first]
            rest = n - first
            self.global_feat[:rest] = gf[first:]
            self.card_feat[:rest] = cf[first:]
            self.legal_mask[:rest] = lm[first:]
            self.policy[:rest] = pl[first:]
            self.value[:rest] = vl[first:]
        self.next_idx = (self.next_idx + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None) -> dict:
        if self.size == 0:
            raise RuntimeError("empty replay buffer")
        idx = torch.randint(
            0, self.size, (batch_size,), device=self.device, generator=generator
        )
        global_feat = self.global_feat[idx]
        card_feat = self.card_feat[idx]
        policy = self.policy[idx]
        value = self.value[idx]
        if self.storage_dtype != torch.float32:
            global_feat = global_feat.to(torch.float32)
            card_feat = card_feat.to(torch.float32)
            policy = policy.to(torch.float32)
            value = value.to(torch.float32)
        return {
            "global_feat": global_feat,
            "card_feat": card_feat,
            "legal_mask": self.legal_mask[idx],
            "policy": policy,
            "value": value,
        }

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "next_idx": self.next_idx,
            "global_feat": self.global_feat,
            "card_feat": self.card_feat,
            "legal_mask": self.legal_mask,
            "policy": self.policy,
            "value": self.value,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.size = int(sd["size"])
        self.next_idx = int(sd["next_idx"])
        self.global_feat = sd["global_feat"].to(device=self.device, dtype=self.storage_dtype)
        self.card_feat = sd["card_feat"].to(device=self.device, dtype=self.storage_dtype)
        self.legal_mask = sd["legal_mask"].to(self.device)
        self.policy = sd["policy"].to(device=self.device, dtype=self.storage_dtype)
        self.value = sd["value"].to(device=self.device, dtype=self.storage_dtype)
