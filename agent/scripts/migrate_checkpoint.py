"""Migrate a checkpoint from an older model to the current architecture.

Handles:
1. g_trunk.0.weight expansion (scalar num_players -> 3-dim one-hot, +2 cols)
2. pc_embed.weight addition (initialized to zeros)
3. Shared policy_head/value_head -> per-PC policy_heads/value_heads (replicated)

Usage:
    python -m agent.scripts.migrate_checkpoint \
        --input agent/runs/league/ckpt_02699_i1200.pt \
        --output agent/runs/attn256_v4/checkpoints/iter_000000.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

from agent.net import model as M
from agent.train.checkpointing import (
    checkpoint_net_spec,
    checkpoint_net_state_dict,
    load_checkpoint_payload,
)


def migrate_state_dict(
    old_sd: dict[str, torch.Tensor],
    hidden: int,
    arch: str,
) -> dict[str, torch.Tensor]:
    """Adapt an old state_dict to the current SplendorNet architecture."""
    new_sd = dict(old_sd)

    # 1. Expand g_trunk.0.weight if from pre-one-hot era (103 -> 105 input dims)
    trunk_key = "g_trunk.0.weight"
    if trunk_key in new_sd and new_sd[trunk_key].shape[1] == 103:
        w = new_sd[trunk_key]
        prefix = w[:, :11]
        suffix = w[:, 11:]
        pad = torch.zeros(w.shape[0], 2, dtype=w.dtype, device=w.device)
        new_sd[trunk_key] = torch.cat([prefix, pad, suffix], dim=1)
        print(f"  {trunk_key}: (256, 103) -> (256, 105)")

    # flat arch equivalent
    flat_key = "flat_trunk.0.weight"
    if flat_key in new_sd:
        w = new_sd[flat_key]
        # Detect old size (should be 2 less than current)
        net_tmp = M.SplendorNet(hidden=hidden, arch="flat")
        expected = net_tmp.state_dict()[flat_key].shape[1]
        del net_tmp
        if w.shape[1] == expected - 2:
            prefix = w[:, :11]
            suffix = w[:, 11:]
            pad = torch.zeros(w.shape[0], 2, dtype=w.dtype, device=w.device)
            new_sd[flat_key] = torch.cat([prefix, pad, suffix], dim=1)
            print(f"  {flat_key}: expanded by 2 input dims")

    # 2. Add pc_embed.weight as zeros if missing
    if "pc_embed.weight" not in new_sd:
        new_sd["pc_embed.weight"] = torch.zeros(3, hidden)
        print(f"  pc_embed.weight: added (3, {hidden}) zeros")

    # 3. Convert shared policy_head/value_head -> per-PC heads
    for head_name in ("policy_head", "value_head"):
        heads_name = head_name.replace("_head", "_heads")
        old_keys = [k for k in new_sd if k.startswith(f"{head_name}.")]
        new_keys = [k for k in new_sd if k.startswith(f"{heads_name}.")]
        if old_keys and not new_keys:
            print(f"  {head_name} -> {heads_name}: replicating to 3 PC-specific heads")
            for old_key in list(old_keys):
                suffix = old_key[len(f"{head_name}."):]  # e.g. "0.weight"
                for pc in ("0", "1", "2"):
                    new_key = f"{heads_name}.{pc}.{suffix}"
                    new_sd[new_key] = new_sd[old_key].clone()
                del new_sd[old_key]

    return new_sd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Migrate a checkpoint to the current model architecture."
    )
    p.add_argument("--input", required=True, help="Source checkpoint path.")
    p.add_argument("--output", required=True, help="Output checkpoint path.")
    p.add_argument("--verify", action="store_true", default=True,
                   help="Verify migrated state_dict loads (default: True).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading: {input_path}")
    payload = load_checkpoint_payload(input_path, map_location="cpu")
    spec = checkpoint_net_spec(payload)
    print(f"  arch={spec.arch}, hidden={spec.hidden}")

    old_sd = checkpoint_net_state_dict(payload)
    print(f"  keys: {len(old_sd)}")

    print("Migrating...")
    new_sd = migrate_state_dict(old_sd, hidden=spec.hidden, arch=spec.arch)

    if args.verify:
        print("Verifying strict load...")
        net = M.SplendorNet(hidden=spec.hidden, arch=spec.arch)
        net.load_state_dict(new_sd, strict=True)
        print("  OK")

    out_payload = {
        "net": new_sd,
        "iteration": 0,
        "config": {"hidden": spec.hidden, "arch": spec.arch},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output_path)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
