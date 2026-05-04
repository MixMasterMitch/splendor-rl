"""Probe what a trained net actually does.

Loads a checkpoint, plays a single batched game to completion vs a random bot,
and prints per-turn:
- top-5 actions the net wants (by policy logit over legal mask)
- top-5 actions the net's value predicts for each seat
- whether the chosen action is a PASS, gem-take, reserve, or buy

Usage:
    bazel run //experimental/mloeppky/splendor/agent/scripts:diagnose_policy -- \
        --ckpt runs/real10/checkpoints/iter_000010.pt --turns 60
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.net import encoder as ENC
from agent.net import model as M
from agent.train import checkpointing as CK


def _action_class(action_id: int) -> str:
    if action_id == A.PASS_ACTION:
        return "PASS"
    if A.TAKE3_BASE <= action_id < A.TAKE3_BASE + A.TAKE3_COUNT:
        return "TAKE3"
    if A.TAKE2_BASE <= action_id < A.TAKE2_BASE + A.TAKE2_COUNT:
        return "TAKE2"
    if A.RESERVE_GRID_BASE <= action_id < A.RESERVE_GRID_BASE + A.RESERVE_GRID_COUNT:
        return "RESERVE_GRID"
    if A.RESERVE_BLIND_BASE <= action_id < A.RESERVE_BLIND_BASE + A.RESERVE_BLIND_COUNT:
        return "RESERVE_BLIND"
    if A.BUY_GRID_BASE <= action_id < A.BUY_GRID_BASE + A.BUY_GRID_COUNT:
        return "BUY_GRID"
    if A.BUY_RESERVED_BASE <= action_id < A.BUY_RESERVED_BASE + A.BUY_RESERVED_COUNT:
        return "BUY_RESERVED"
    if A.DISCARD_BASE <= action_id < A.DISCARD_BASE + A.DISCARD_COUNT:
        return "DISCARD"
    if A.PICK_NOBLE_BASE <= action_id < A.PICK_NOBLE_BASE + A.PICK_NOBLE_COUNT:
        return "PICK_NOBLE"
    return f"?act{action_id}"


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--arch", choices=["attn", "flat"], default="")
    p.add_argument("--turns", type=int, default=60)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--batch", type=int, default=1, help="Number of games to simulate")
    args = p.parse_args(argv)

    device = "cpu"
    payload = CK.load_checkpoint_payload(args.ckpt, map_location=device)
    spec = CK.checkpoint_net_spec(payload)
    hidden = spec.hidden
    arch = args.arch or spec.arch
    net = M.SplendorNet(hidden=hidden, arch=arch).to(device)
    net.load_state_dict(CK.checkpoint_net_state_dict(payload))
    net.eval()

    engine = BE.BatchedEngine(
        batch_size=args.batch, num_players=2, device=device, seed=42
    )

    pass_count = 0
    buy_count = 0
    take_count = 0
    reserve_count = 0

    print(f"== Playing {args.batch} batched games with ckpt {args.ckpt} ==")
    for turn in range(args.turns):
        if engine.ended.all().item():
            print(f"all games ended at turn {turn}")
            break
        g, c, _, legal = ENC.encode_state_with_legal(engine)
        with torch.no_grad():
            logits, value = net(g, c, legal)
        # For game 0 only: print top-k
        if args.batch == 1 or turn < 5:
            b = 0
            if not engine.ended[b].item():
                lmask = legal[b]
                lmasked = torch.where(
                    lmask, logits[b], torch.full_like(logits[b], -1e9)
                )
                probs = torch.softmax(lmasked, dim=-1)
                topk = torch.topk(probs, k=min(args.top, int(lmask.sum().item())))
                cp = int(engine.current_player[b].item())
                pts = [int(x) for x in engine.points[b].tolist()]
                print(
                    f"t={turn} cp={cp} pts={pts} value={value[b].tolist()}"
                )
                for pr, act in zip(topk.values.tolist(), topk.indices.tolist()):
                    print(
                        f"   {pr:6.3f}  act={act:>3}  {_action_class(int(act))}  "
                        f"{A.action_name(int(act))}"
                    )

        # Greedy act per game (argmax over legal)
        lmasked_b = torch.where(legal, logits, torch.full_like(logits, -1e9))
        actions = lmasked_b.argmax(dim=-1)
        for b in range(args.batch):
            if engine.ended[b].item():
                continue
            a = int(actions[b].item())
            klass = _action_class(a)
            if klass == "PASS":
                pass_count += 1
            elif klass.startswith("BUY"):
                buy_count += 1
            elif klass.startswith("TAKE"):
                take_count += 1
            elif klass.startswith("RESERVE"):
                reserve_count += 1
        engine.apply(actions)

    total = pass_count + buy_count + take_count + reserve_count
    print("\n== Action class distribution (greedy) ==")
    for name, v in [("PASS", pass_count), ("BUY", buy_count), ("TAKE", take_count), ("RESERVE", reserve_count)]:
        pct = 100.0 * v / max(total, 1)
        print(f"  {name:<8} {v:>5}  {pct:5.1f}%")
    print(
        f"\nfinal points: {engine.points.tolist()}  ended: {engine.ended.tolist()}  winners pts: {engine.points.max(dim=-1).values.tolist()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
