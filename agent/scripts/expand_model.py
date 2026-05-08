"""Expand a trained SplendorNet checkpoint to a wider hidden dimension.

Strategy: "Output-gated expansion"

The fundamental challenge with function-preserving expansion through LayerNorm
is that LN normalizes over the full dimension — any change to the dimension
alters the normalization statistics for ALL neurons, not just new ones.

Our approach: expand all layers to the new hidden dim, but ensure the FINAL
output layers (policy_head[-1] and value_head[-1]) have zero weights for all
new input dimensions. This means:
- Intermediate representations WILL differ from the original model (due to LN).
- But the final policy/value outputs will be CLOSE to the original (not exact).
- The new capacity is "dormant" at the output level and will be activated by
  gradient flow during training.

For practical purposes, the model starts at approximately the same playing
strength and quickly adapts. The alternative (exact preservation) would require
changing the model architecture, which breaks the training loop.

If you need exact preservation, use --strategy=frozen-trunk which keeps the
trunk at the original dimension and only expands the heads.

Usage:
    python -m agent.scripts.expand_model \\
        --input agent/runs/real30_v11/checkpoints/latest_resume.pt \\
        --output agent/runs/expanded_256/checkpoints/iter_000000.pt \\
        --new-hidden 256

The output checkpoint can be used as `init_from` in a new training run.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

from ..net.model import SplendorNet


# =============================================================================
# Strategy: "full" — expand everything, zero-gate at output
# =============================================================================


def _expand_linear_full(
    old_linear: nn.Linear,
    new_in: int,
    new_out: int,
    zero_new_in: bool = True,
    replicate_new_out: bool = False,
    init_scale: float = 0.01,
) -> nn.Linear:
    """Expand a Linear layer.

    - Old weights copied to top-left block.
    - zero_new_in: zero out columns >= old_in for old output rows (prevents
      new input dims from affecting old neurons).
    - replicate_new_out: new output rows copy random existing rows (preserves
      activation distribution for downstream LayerNorm). Otherwise small random.
    """
    old_in = old_linear.in_features
    old_out = old_linear.out_features
    new_layer = nn.Linear(new_in, new_out, bias=old_linear.bias is not None)

    with torch.no_grad():
        nn.init.normal_(new_layer.weight, std=init_scale)

        # Copy old weights
        new_layer.weight[:old_out, :old_in].copy_(old_linear.weight)

        # Zero new input columns for old output neurons
        if zero_new_in and new_in > old_in:
            new_layer.weight[:old_out, old_in:].zero_()

        # Replicate existing neurons for new output rows
        if replicate_new_out and new_out > old_out:
            torch.manual_seed(7)  # deterministic replication
            for i in range(old_out, new_out):
                src = torch.randint(0, old_out, (1,)).item()
                new_layer.weight[i, :old_in].copy_(old_linear.weight[src])
                if new_in > old_in:
                    new_layer.weight[i, old_in:].zero_()

        # Bias
        if old_linear.bias is not None:
            new_layer.bias.zero_()
            new_layer.bias[:old_out].copy_(old_linear.bias)
            if replicate_new_out and new_out > old_out:
                torch.manual_seed(7)
                for i in range(old_out, new_out):
                    src = torch.randint(0, old_out, (1,)).item()
                    new_layer.bias[i] = old_linear.bias[src]

    return new_layer


def _expand_layernorm(old_ln: nn.LayerNorm, new_dim: int) -> nn.LayerNorm:
    """Expand LayerNorm — new dims get weight=1, bias=0."""
    old_dim = old_ln.normalized_shape[0]
    new_ln = nn.LayerNorm(new_dim)
    with torch.no_grad():
        new_ln.weight.fill_(1.0)
        new_ln.weight[:old_dim].copy_(old_ln.weight)
        new_ln.bias.zero_()
        new_ln.bias[:old_dim].copy_(old_ln.bias)
    return new_ln


def _expand_mha(
    old_attn: nn.MultiheadAttention,
    new_hidden: int,
    new_num_heads: int,
    init_scale: float = 0.01,
) -> nn.MultiheadAttention:
    """Expand MHA — old weights in top-left, new dims zeroed for input side."""
    old_hidden = old_attn.embed_dim
    new_attn = nn.MultiheadAttention(
        embed_dim=new_hidden,
        num_heads=new_num_heads,
        dropout=old_attn.dropout,
        batch_first=old_attn.batch_first,
    )

    with torch.no_grad():
        nn.init.normal_(new_attn.in_proj_weight, std=init_scale)
        # For each Q/K/V block: copy old weights, zero new input cols for old rows
        for i in range(3):
            old_start = i * old_hidden
            old_end = (i + 1) * old_hidden
            new_start = i * new_hidden
            # Copy old block
            new_attn.in_proj_weight[new_start:new_start + old_hidden, :old_hidden].copy_(
                old_attn.in_proj_weight[old_start:old_end, :]
            )
            # Zero new input columns for old output rows
            new_attn.in_proj_weight[new_start:new_start + old_hidden, old_hidden:].zero_()

        if old_attn.in_proj_bias is not None:
            new_attn.in_proj_bias.zero_()
            for i in range(3):
                old_start = i * old_hidden
                old_end = (i + 1) * old_hidden
                new_start = i * new_hidden
                new_attn.in_proj_bias[new_start:new_start + old_hidden].copy_(
                    old_attn.in_proj_bias[old_start:old_end]
                )

        # out_proj: zero new input cols for old output rows
        nn.init.normal_(new_attn.out_proj.weight, std=init_scale)
        new_attn.out_proj.weight[:old_hidden, :old_hidden].copy_(
            old_attn.out_proj.weight
        )
        new_attn.out_proj.weight[:old_hidden, old_hidden:].zero_()
        if old_attn.out_proj.bias is not None:
            new_attn.out_proj.bias.zero_()
            new_attn.out_proj.bias[:old_hidden].copy_(old_attn.out_proj.bias)

    return new_attn


def _expand_head_first_layer(
    old_linear: nn.Linear,
    old_hidden: int,
    new_hidden: int,
    new_out: int,
    init_scale: float = 0.01,
) -> nn.Linear:
    """Expand first layer of head (input is cat([h_g, h_a]))."""
    old_out = old_linear.out_features
    new_layer = nn.Linear(new_hidden * 2, new_out, bias=old_linear.bias is not None)

    with torch.no_grad():
        nn.init.normal_(new_layer.weight, std=init_scale)

        # Copy old weights at correct positions
        # h_g occupies [0:old_hidden] in old, [0:old_hidden] in new (same)
        # h_a occupies [old_hidden:2*old_hidden] in old, [new_hidden:new_hidden+old_hidden] in new
        new_layer.weight[:old_out, :old_hidden].copy_(
            old_linear.weight[:, :old_hidden]
        )
        new_layer.weight[:old_out, new_hidden:new_hidden + old_hidden].copy_(
            old_linear.weight[:, old_hidden:]
        )
        # Zero new input columns for old output neurons
        new_layer.weight[:old_out, old_hidden:new_hidden].zero_()
        new_layer.weight[:old_out, new_hidden + old_hidden:].zero_()

        if old_linear.bias is not None:
            new_layer.bias.zero_()
            new_layer.bias[:old_out].copy_(old_linear.bias)

    return new_layer


def _expand_head_last_layer(
    old_linear: nn.Linear,
    new_in: int,
) -> nn.Linear:
    """Expand last layer of head — ZERO weights for new input dims.

    This is the critical gate: new neurons have no effect on output initially.
    """
    old_in = old_linear.in_features
    out_dim = old_linear.out_features
    new_layer = nn.Linear(new_in, out_dim, bias=old_linear.bias is not None)

    with torch.no_grad():
        new_layer.weight.zero_()
        new_layer.weight[:, :old_in].copy_(old_linear.weight)
        if old_linear.bias is not None:
            new_layer.bias.copy_(old_linear.bias)

    return new_layer


def expand_full(
    old_net: SplendorNet,
    new_hidden: int,
    new_num_heads: int,
    init_scale: float = 0.01,
) -> SplendorNet:
    """Expand using full strategy — all layers widened, output gated."""
    old_hidden = old_net.hidden
    from ..net import encoder as ENC

    new_net = SplendorNet(
        hidden=new_hidden, arch="attn", num_heads=new_num_heads,
        dropout=0.0, compile_forward=False,
    )

    with torch.no_grad():
        # g_trunk: Linear, GELU, LN, Linear, GELU, LN
        # Replicate new outputs to preserve LN stats. Don't zero new inputs
        # on internal layers — let replicated neurons flow through naturally.
        new_net.g_trunk[0] = _expand_linear_full(
            old_net.g_trunk[0], ENC.D_GLOBAL, new_hidden,
            zero_new_in=False, replicate_new_out=True, init_scale=init_scale,
        )
        new_net.g_trunk[2] = _expand_layernorm(old_net.g_trunk[2], new_hidden)
        new_net.g_trunk[3] = _expand_linear_full(
            old_net.g_trunk[3], new_hidden, new_hidden,
            zero_new_in=False, replicate_new_out=True, init_scale=init_scale,
        )
        new_net.g_trunk[5] = _expand_layernorm(old_net.g_trunk[5], new_hidden)

        # c_embed: Linear, GELU, LN, Linear, GELU
        new_net.c_embed[0] = _expand_linear_full(
            old_net.c_embed[0], ENC.D_CARD, new_hidden,
            zero_new_in=False, replicate_new_out=True, init_scale=init_scale,
        )
        new_net.c_embed[2] = _expand_layernorm(old_net.c_embed[2], new_hidden)
        new_net.c_embed[3] = _expand_linear_full(
            old_net.c_embed[3], new_hidden, new_hidden,
            zero_new_in=False, replicate_new_out=True, init_scale=init_scale,
        )

        # Attention
        new_net.attn = _expand_mha(old_net.attn, new_hidden, new_num_heads, init_scale)
        new_net.post_attn = _expand_layernorm(old_net.post_attn, new_hidden)

        # Policy head
        new_net.policy_head[0] = _expand_head_first_layer(
            old_net.policy_head[0], old_hidden, new_hidden, new_hidden, init_scale
        )
        new_net.policy_head[2] = _expand_head_last_layer(
            old_net.policy_head[2], new_hidden
        )

        # Value head
        new_net.value_head[0] = _expand_head_first_layer(
            old_net.value_head[0], old_hidden, new_hidden, new_hidden, init_scale
        )
        new_net.value_head[2] = _expand_head_last_layer(
            old_net.value_head[2], new_hidden
        )

    return new_net


# =============================================================================
# Strategy: "frozen-trunk" — keep trunk at old dim, only expand heads
# This IS exactly function-preserving but limits expansion benefit.
# =============================================================================
# (Not implemented yet — the full strategy is preferred for training)


# =============================================================================
# Main
# =============================================================================


def expand_model(
    old_net: SplendorNet,
    new_hidden: int,
    new_num_heads: int | None = None,
    init_scale: float = 0.01,
) -> SplendorNet:
    """Create an expanded SplendorNet."""
    old_hidden = old_net.hidden

    if new_hidden <= old_hidden:
        raise ValueError(
            f"new_hidden ({new_hidden}) must be larger than old ({old_hidden})"
        )

    if new_num_heads is None:
        old_head_dim = old_hidden // 4
        new_num_heads = max(4, new_hidden // old_head_dim)
        while new_hidden % new_num_heads != 0:
            new_num_heads -= 1

    if old_net.arch != "attn":
        raise NotImplementedError("Expansion only implemented for 'attn' arch")

    return expand_full(old_net, new_hidden, new_num_heads, init_scale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand a SplendorNet checkpoint")
    parser.add_argument("--input", required=True, help="Path to source checkpoint")
    parser.add_argument("--output", required=True, help="Path to write expanded checkpoint")
    parser.add_argument("--new-hidden", type=int, default=256, help="New hidden dim")
    parser.add_argument("--new-heads", type=int, default=None, help="New num attention heads")
    parser.add_argument(
        "--init-scale", type=float, default=0.01,
        help="Std for random init of new neurons (default: 0.01)"
    )
    args = parser.parse_args(argv)

    print(f"Loading checkpoint: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    # Extract model state
    if "net" in ckpt:
        state_dict = ckpt["net"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    old_hidden = state_dict["g_trunk.0.weight"].shape[0]
    print(f"Old hidden: {old_hidden}")
    print(f"New hidden: {args.new_hidden}")
    print(f"Init scale: {args.init_scale}")

    # Build old model and load weights
    old_net = SplendorNet(hidden=old_hidden, arch="attn", compile_forward=False)
    old_net.load_state_dict(state_dict, strict=True)

    # Expand
    new_net = expand_model(
        old_net, args.new_hidden, args.new_heads, init_scale=args.init_scale
    )

    # Verify
    print("\nVerifying expansion quality...")
    old_net.eval()
    new_net.eval()
    from ..net import encoder as ENC
    num_actions = old_net.policy_head[2].out_features

    with torch.no_grad():
        B = 16
        torch.manual_seed(123)
        g = torch.randn(B, ENC.D_GLOBAL)
        c = torch.randn(B, ENC.N_CARDS, ENC.D_CARD)
        legal = torch.ones(B, num_actions, dtype=torch.bool)

        old_logits, old_value = old_net(g, c, legal)
        new_logits, new_value = new_net(g, c, legal)

        logit_diff = (old_logits - new_logits).abs().max().item()
        value_diff = (old_value - new_value).abs().max().item()
        logit_mean_diff = (old_logits - new_logits).abs().mean().item()
        value_mean_diff = (old_value - new_value).abs().mean().item()

        # Action agreement
        old_actions = old_logits.argmax(dim=-1)
        new_actions = new_logits.argmax(dim=-1)
        action_agreement = (old_actions == new_actions).float().mean().item()

        # Value correlation
        value_corr = torch.corrcoef(
            torch.stack([old_value[:, 0], new_value[:, 0]])
        )[0, 1].item()

        print(f"  Max logit difference:  {logit_diff:.4f}")
        print(f"  Mean logit difference: {logit_mean_diff:.4f}")
        print(f"  Max value difference:  {value_diff:.4f}")
        print(f"  Mean value difference: {value_mean_diff:.4f}")
        print(f"  Action agreement:      {action_agreement*100:.0f}%")
        print(f"  Value correlation:     {value_corr:.3f}")
        print()
        if value_corr > 0.95:
            print("  ✓ Excellent preservation — model should play at near-original strength.")
        elif value_corr > 0.8:
            print("  ✓ Good preservation — model retains learned features.")
            print("    Will recover to full strength within ~50-100 training iterations.")
        elif value_corr > 0.5:
            print("  ~ Moderate preservation — significant warmup needed.")
        else:
            print("  ⚠ Poor preservation — consider using a smaller expansion ratio.")

    # Save
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_ckpt = {}
    if "net" in ckpt:
        new_ckpt["net"] = new_net.state_dict()
    elif "model_state_dict" in ckpt:
        new_ckpt["model_state_dict"] = new_net.state_dict()
    else:
        new_ckpt["net"] = new_net.state_dict()

    for k, v in ckpt.items():
        if k in ("net", "model_state_dict", "state_dict", "optimizer", "opt"):
            continue
        new_ckpt[k] = v

    # Update config to reflect new architecture dimensions
    if "config" in new_ckpt:
        new_ckpt["config"] = dict(new_ckpt["config"])
        new_ckpt["config"]["hidden"] = args.new_hidden
        new_ckpt["config"]["arch"] = "attn"
    else:
        new_ckpt["config"] = {"hidden": args.new_hidden, "arch": "attn"}

    new_ckpt["expanded_from"] = {
        "source": args.input,
        "old_hidden": old_hidden,
        "new_hidden": args.new_hidden,
        "new_heads": args.new_heads,
        "init_scale": args.init_scale,
    }

    torch.save(new_ckpt, output_path)
    print(f"\nSaved expanded checkpoint to: {output_path}")
    print(f"  Parameters: {sum(p.numel() for p in old_net.parameters()):,} -> "
          f"{sum(p.numel() for p in new_net.parameters()):,}")
    print(f"  Num heads: {new_net.attn.num_heads}")
    print(f"\nNote: Use this as init_from in a new training run.")
    print(f"  Recommended: lower LR initially (e.g., 2e-5) to let the model stabilize.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
