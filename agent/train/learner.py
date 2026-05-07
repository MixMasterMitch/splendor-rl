"""Learner: samples minibatches from the replay buffer and updates the network
with a policy (KL) + value (MSE) loss.

Returns per-step metrics useful for the iterative training loop.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..env import batched_engine as BE
from ..net import model as M
from .replay_buffer import ReplayBuffer


def make_optimizer(net: M.SplendorNet, lr: float = 3e-4, weight_decay: float = 1e-4):
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)


def step(
    net: M.SplendorNet,
    buffer: ReplayBuffer,
    optim: torch.optim.Optimizer,
    batch_size: int = 256,
    value_loss_weight: float = 1.0,
    policy_loss_weight: float = 1.0,
    entropy_bonus: float = 0.0,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
) -> dict:
    net.train()
    batch = buffer.sample(batch_size, generator=generator)
    g = batch["global_feat"].to(device)
    c = batch["card_feat"].to(device)
    mask = batch["legal_mask"].to(device)
    target_p = batch["policy"].to(device)
    target_v = batch["value"].to(device)

    # Normalize target policy (training collects may be unnormalized)
    target_p = target_p.clamp_min(0)
    sums = target_p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_p = target_p / sums

    use_amp = grad_scaler is not None

    # Forward pass — optionally under AMP autocast for mixed precision.
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits, value = net(g, c, mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(target_p * log_probs).sum(dim=-1).mean()

        # Only supervise active seats in value loss; active seats are those with
        # abs(target_v) > 1e-6
        active = target_v.abs() > 1e-6
        if active.any():
            value_loss = F.mse_loss(value[active], target_v[active])
        else:
            value_loss = torch.zeros((), device=device)

        loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss

    with torch.no_grad():
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1).mean()
    if entropy_bonus > 0:
        loss = loss - entropy_bonus * ent

    optim.zero_grad(set_to_none=True)

    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        grad_scaler.step(optim)
        grad_scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optim.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(ent.item()),
        "grad_norm": float(grad_norm),
    }
