"""Simplified Gumbel-root MCTS over the batched engine.

Design notes:
- Full MuZero-style batched MCTS is complex to implement from scratch for a
  game with stochasticity (deck reveals) and variable branching.
- For our V1, we use a pragmatic "Gumbel root + 1-ply learned value" scheme:
  for the top-K actions at the root (selected via Gumbel sampling on prior
  logits), we expand all B x K children in one batched engine step, score
  them with a single value-network pass, and combine those Q estimates with
  the prior to pick a final action. This is enough to bootstrap AlphaZero-
  style training and can be upgraded to full Gumbel-AlphaZero MCTS later.
- Tree search budget per move: `num_sims` root-child simulations (default 8).

The Gumbel-root formulation from Danihelka et al. (2022) is used to both
explore at the root and produce a well-behaved improved policy target for
policy distillation.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import model as M


def _sample_gumbel(shape, device):
    u = torch.rand(shape, device=device).clamp_min(1e-9)
    return -torch.log(-torch.log(u))


def gumbel_root_act(
    engine: BE.BatchedEngine,
    net: M.SplendorNet,
    num_sims: int = 8,
    temperature: float = 1.0,
    root_noise_scale: float = 1.0,
    dirichlet_alpha: float = 0.0,
    dirichlet_mix: float = 0.0,
    q_scale: float = 10.0,
    precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select actions for each game in the batch."""
    return gumbel_root_act_python(
        engine,
        net,
        num_sims=num_sims,
        temperature=temperature,
        root_noise_scale=root_noise_scale,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_mix=dirichlet_mix,
        q_scale=q_scale,
        precomputed=precomputed,
    )


def gumbel_root_act_python(
    engine: BE.BatchedEngine,
    net: M.SplendorNet,
    num_sims: int = 8,
    temperature: float = 1.0,
    root_noise_scale: float = 1.0,
    dirichlet_alpha: float = 0.0,
    dirichlet_mix: float = 0.0,
    q_scale: float = 10.0,
    precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select an action for each game in the batch.

    Returns (actions (B,) int64, improved_policy (B, NUM_ACTIONS) float).

    The improved policy is the softmax over logits + gumbel + completed
    Q-values for the simulated top-K actions (Danihelka et al. "Gumbel
    AlphaZero"), and is what we train the policy toward.

    When `dirichlet_alpha > 0` we additionally mix Dirichlet noise into the
    prior over LEGAL actions (AlphaZero's standard exploration trick). This
    encourages occasional exploration of under-weighted actions like BUY even
    when the current policy is concentrated on TAKE3.
    """
    device = engine.device
    B = engine.batch_size
    NA = A.NUM_ACTIONS

    if precomputed is None:
        logits, _, legal = net.inference(engine)
    else:
        g, c, legal = precomputed
        logits, _ = net.inference_from_encoded(g, c, legal)

    if dirichlet_alpha > 0.0:
        # Sample Dirichlet noise over the legal support of each game.
        legal_f = legal.to(logits.dtype)
        # Sample gamma(alpha) for all NA, then zero out illegal entries and
        # renormalize to produce per-game Dirichlet draws over the legal set.
        gamma = torch._standard_gamma(
            torch.full_like(legal_f, dirichlet_alpha)
        )
        gamma = gamma * legal_f
        gamma = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Mix into policy space via log-prob perturbation.
        prior = torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)
        mixed = (1.0 - dirichlet_mix) * prior + dirichlet_mix * gamma
        logits = torch.log(mixed.clamp_min(1e-9))

    # Sample Gumbel perturbation at the root
    gumbel = _sample_gumbel((B, NA), device) * root_noise_scale
    masked_score = (logits + gumbel).masked_fill(~legal, -1e9)

    # Top-K actions to "search"
    k = min(num_sims, NA)
    if k <= 0:
        probs = torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)
        action = probs.argmax(dim=-1)
        return action, probs
    # Keep the root expansion width fixed so compiled CPU graphs do not
    # re-specialize whenever the legal-action count changes.
    _, topk_idx = torch.topk(masked_score, k=k, dim=-1)  # (B,k)

    # Expand all top-k root children in one B x K batch, then score them with
    # a single value-head call from the successor state's rotated perspective.
    q_values = _evaluate_root_children_batched(engine, net, topk_idx, legal)

    # Improved policy target: following Danihelka et al. (2022), we combine
    # prior and "completed Q-values". For each game, unvisited actions get a
    # neutral baseline v_mix = prior-weighted mean of the searched Q-values
    # (so unvisited actions don't inherit an overly optimistic or pessimistic
    # implicit Q of 0). Visited actions use their searched Q. Then:
    #   improved_logits = logits + sigma(completed_q) - (no root gumbel here)
    # and we softmax over legal actions.
    searched = torch.isfinite(q_values)
    prior_probs = torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)
    # Prior-weighted sum/mass over searched actions per game.
    weight = prior_probs * searched.to(logits.dtype)
    mass = weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    q_safe = torch.where(searched, q_values, torch.zeros_like(q_values))
    v_mix = (weight * q_safe).sum(dim=-1, keepdim=True) / mass  # (B, 1)
    completed_q = torch.where(searched, q_values, v_mix.expand_as(q_values))
    sigma_q = q_scale * completed_q

    # Training target: no gumbel here; gumbel is only used to randomize the
    # top-K selection for sampling during self-play. The policy we regress to
    # should be the clean search-improved posterior.
    improved_logits = (logits + sigma_q).masked_fill(~legal, -1e9)
    probs = torch.softmax(improved_logits / max(temperature, 1e-3), dim=-1)

    # Action selection: among the searched actions, pick the one with highest
    # `logits + sigma(q)` (deterministic). Exploration at the root comes from
    # (a) the Gumbel-driven top-K selection above and (b) the Dirichlet noise
    # mixed into `logits` before top-K was picked.
    deterministic = logits + sigma_q
    best_score = deterministic.masked_fill(~searched, -1e9)
    fallback_score = deterministic.masked_fill(~legal, -1e9)
    use_searched = searched.any(dim=-1, keepdim=True)
    score_pick = torch.where(use_searched, best_score, fallback_score)
    action = score_pick.argmax(dim=-1)
    return action, probs
def _safe_root_actions(topk_idx: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Remap any illegal top-k slot to a guaranteed legal fallback action."""
    any_legal = legal.to(torch.int64).argmax(dim=-1, keepdim=True)
    topk_legal = legal.gather(1, topk_idx)
    return torch.where(topk_legal, topk_idx, any_legal.expand_as(topk_idx))


def _root_value_index(
    parent_cp: torch.Tensor,
    child_cp: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Determine the correct value-head index for the root player in each child state.

    The value head outputs are rotated so seat 0 = the child's current player,
    using a rotation over ``MAX_PLAYERS`` (the encoder rotates by MAX_PLAYERS,
    not ``num_players``, to keep tensor shapes constant). In the child's
    rotated view, the seat holding the root player (parent's current player)
    sits at index ``(parent_cp - child_cp) % MAX_PLAYERS``. This works
    uniformly for the same-player case (discard/noble-pick, parent_cp ==
    child_cp → 0), normal advance (e.g. parent_cp=0, child_cp=1 → MAX_PLAYERS-1),
    and wrap-around (parent_cp=nP-1, child_cp=0 → nP-1).

    Historical note: previous versions returned ``num_players - 1`` for the
    "advanced" case, which coincidentally works at 4p (= MAX_PLAYERS) but
    systematically reads either an empty seat (zero-supervised) or, worse,
    the opponent's predicted value at 2p/3p — maximizing the opponent's
    position instead of the root's. This manifested as a persistent plateau
    in 3p strength across thousands of training iterations.

    Returns a (N,) long tensor of column indices to gather from value output.
    """
    from ..env import batched_engine as _BE

    diff = parent_cp.to(torch.long) - child_cp.to(torch.long)
    return diff.remainder(_BE.MAX_PLAYERS)


def _evaluate_root_children_sequential(
    engine: BE.BatchedEngine,
    net: M.SplendorNet,
    topk_idx: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    """Reference root-child evaluation path used for parity checks."""
    B = engine.batch_size
    q_values = torch.full((B, A.NUM_ACTIONS), float("-inf"), device=engine.device)
    safe_topk = _safe_root_actions(topk_idx, legal)
    parent_cp = engine.current_player.to(torch.long)
    for ki in range(safe_topk.shape[1]):
        clone = engine.clone()
        action_k = safe_topk[:, ki]
        clone.apply(action_k)
        v2 = net.value_inference(clone)
        child_cp = clone.current_player.to(torch.long)
        val_idx = _root_value_index(parent_cp, child_cp, engine.num_players)
        q_for_root = v2.gather(1, val_idx.unsqueeze(-1)).squeeze(-1)
        q_values.scatter_(1, action_k.unsqueeze(-1), q_for_root.unsqueeze(-1))
    return q_values


def _evaluate_root_children_batched(
    engine: BE.BatchedEngine,
    net: M.SplendorNet,
    topk_idx: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    """Expand all root children in one B x K batched engine/value pass."""
    B = engine.batch_size
    K = topk_idx.shape[1]
    q_values = torch.full((B, A.NUM_ACTIONS), float("-inf"), device=engine.device)
    if K == 0:
        return q_values
    safe_topk = _safe_root_actions(topk_idx, legal)
    # Track parent's current_player before applying child actions.
    parent_cp = engine.current_player.to(torch.long).repeat_interleave(K)  # (B*K,)
    child_engine = engine.repeat_interleave(K)
    child_engine.apply(safe_topk.reshape(-1))
    child_cp = child_engine.current_player.to(torch.long)  # (B*K,)
    val_idx = _root_value_index(parent_cp, child_cp, engine.num_players)  # (B*K,)
    child_values_all = net.value_inference(child_engine)  # (B*K, MAX_PLAYERS)
    child_values = child_values_all.gather(1, val_idx.unsqueeze(-1)).squeeze(-1).reshape(B, K)
    q_values.scatter_(1, safe_topk, child_values)
    return q_values
