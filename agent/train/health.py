"""Decision rules for the iterative training loop.

After each short burst the implementing agent calls `decide_next_action` with
the latest metric rows; it returns a string in:
- "continue": keep training with the current config.
- "checkpoint_and_eval": dump a checkpoint and run the eval ladder.
- "reduce_lr": lower the learning rate by 10x next burst.
- "stop_regressing": evals are regressing beyond noise; roll back to last good
  checkpoint and try a smaller update schedule.
- "stop_converged": evals are flat at strong level for several evals; training
  considered done for this milestone.

The rules are intentionally simple and conservative so the agent can reason
about them from the journal.
"""

from __future__ import annotations

from typing import List, Optional


def _mean(xs):
    return sum(xs) / max(len(xs), 1)


def _trend(xs):
    if len(xs) < 2:
        return 0.0
    return xs[-1] - xs[0]


def _rows_since_last_lr_change(metrics_history: List[dict]) -> int:
    ladder = [m for m in metrics_history if "winrate_vs_heuristic" in m]
    if not ladder:
        return 10**9
    current_lr = ladder[-1].get("lr")
    if current_lr is None:
        return 10**9
    current = float(current_lr)
    same = 0
    for row in reversed(ladder):
        lr = row.get("lr")
        if lr is None:
            break
        if abs(float(lr) - current) > 1e-12:
            break
        same += 1
    return max(same - 1, 0)


def decide_next_action(
    metrics_history: List[dict],
    recent_k: int = 5,
    min_rows_for_decision: int = 8,
    strong_winrate: float = 0.9,
    regress_epsilon: float = -0.10,
    flat_epsilon: float = 0.02,
    stuck_winrate_threshold: float = 0.3,
    rand_win_threshold: float = 0.55,
    loss_plateau_epsilon: float = 0.02,
    reduce_lr_cooldown_rows: int = 5,
    late_stage_winrate_threshold: float = 0.5,
    late_stage_regress_epsilon: float = -0.04,
    late_stage_cooldown_rows: int = 8,
    min_lr: float = 3e-6,
) -> str:
    """`metrics_history` is the list of JSON rows written to metrics.jsonl in
    order. Only metric rows with `winrate_vs_heuristic` are consulted here.

    Conservative rule set:
    - Always "continue" until we have at least `min_rows_for_decision` eval rows.
    - "stop_converged": `strong_winrate` sustained across the whole recent
      window AND a near-zero trend.
    - "stop_regressing": large, sustained negative trend across `recent_k`.
    - "reduce_lr": all of the following, else "continue":
        1. recent window shows a win-rate plateau at weak play (vs both random
           and heuristic);
        2. training loss is also plateauing (recent loss not appreciably lower
           than at the start of the window) — if loss is still dropping, the
           network is still learning and we should NOT touch the learning rate;
        3. no `reduce_lr` event has fired in the last `reduce_lr_cooldown_rows`
           eval rows. This prevents the old runaway where LR compounded down
           by 10x every iteration, eventually reaching numerical zero.
    """
    ladder = [m for m in metrics_history if "winrate_vs_heuristic" in m]
    if len(ladder) < min_rows_for_decision:
        return "continue"

    recent = ladder[-recent_k:]
    wrs = [float(m["winrate_vs_heuristic"]) for m in recent]
    rand_wrs = [float(m.get("winrate_vs_random", 0.0)) for m in recent]
    losses = [float(m.get("loss", 0.0)) for m in recent]

    latest = wrs[-1]
    trend = _trend(wrs)
    mean_recent = _mean(wrs)
    mean_rand = _mean(rand_wrs)
    loss_drop = losses[0] - losses[-1] if len(losses) >= 2 else 0.0
    rows_since_lr_change = _rows_since_last_lr_change(metrics_history)
    latest_lr = float(ladder[-1].get("lr", 1.0))

    # Stop only if the agent has clearly converged at strong play.
    if (
        latest >= strong_winrate
        and mean_recent >= strong_winrate
        and abs(trend) <= flat_epsilon
    ):
        return "stop_converged"

    # Late-stage annealing: once the agent is clearly competent, prefer to
    # lower LR on flat / mildly regressing windows instead of stopping.
    in_late_stage = latest >= late_stage_winrate_threshold or mean_recent >= late_stage_winrate_threshold
    if (
        in_late_stage
        and latest_lr > min_lr
        and rows_since_lr_change >= late_stage_cooldown_rows
        and (abs(trend) <= flat_epsilon or trend <= late_stage_regress_epsilon)
    ):
        return "reduce_lr"

    # Stop on a clear, sustained regression across the whole recent window.
    if trend <= regress_epsilon and len(wrs) >= recent_k and not in_late_stage:
        return "stop_regressing"

    # Reduce LR only when learning has visibly stalled.
    plateau_wins = (
        len(wrs) >= recent_k
        and mean_recent < stuck_winrate_threshold
        and mean_rand < rand_win_threshold
        and abs(trend) <= flat_epsilon
    )
    plateau_loss = len(losses) >= recent_k and loss_drop <= loss_plateau_epsilon
    if plateau_wins and plateau_loss and latest_lr > min_lr:
        if rows_since_lr_change >= reduce_lr_cooldown_rows:
            return "reduce_lr"

    return "continue"
