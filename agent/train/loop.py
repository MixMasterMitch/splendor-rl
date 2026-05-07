"""The iterative train-evaluate-improve loop.

Given a Run, config, and optional checkpoint, performs:
1. A bounded training burst (`--max-iters` iterations or `--max-wall-minutes`
   of wall clock). Each iteration: self-play burst -> buffer write -> N learner
   steps.
2. Every `eval_every` iterations, runs the eval ladder, writes a metric row,
   and consults `health.decide_next_action` to decide whether to continue,
   reduce LR, stop, etc.
3. Periodically checkpoints (atomic write) and updates `state.json`.
4. Emits heartbeat writes on a background cadence so watchers can detect
   hangs.

This loop is the main entry point for `scripts/train.py` and is designed to
survive process death: re-running with the same run-id picks up from the
latest `state.json` + checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Optional

import torch

from ..eval import ladder as LAD
from ..net import model as M
from ..obs import journal as J
from ..obs.run import Run
from replay import players as P
from .async_eval import AsyncEvalConfig, AsyncEvalHandle
from .checkpointing import (
    checkpoint_net_spec,
    checkpoint_net_state_dict,
    load_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
)
from .device import (
    configure_device,
    resolve_device,
)
from .health import decide_next_action
from .learner import make_optimizer, step as learner_step
from .league import League
from .league_selfplay import run_league_selfplay
from .replay_buffer import ReplayBuffer
from .selfplay import run_selfplay


@dataclasses.dataclass
class LoopConfig:
    num_players: int = 2
    device: str = "cpu"
    compile_net: bool = False
    hidden: int = 192
    arch: str = "flat"
    selfplay_games: int = 512
    selfplay_sims: int = 8
    selfplay_max_turns: int = 160
    replay_capacity: int = 600_000
    learner_batch: int = 256
    learner_steps_per_iter: int = 192
    entropy_bonus: float = 0.015
    # eval_games split evenly by seat; each seat rotation runs B = eval_games //
    # num_players. Keep per_seat >= 128 so CPU batch-throughput is near optimum.
    eval_every: int = 2
    eval_games: int = 256
    eval_sims: int = 4
    eval_max_turns: int = 200
    rank_eval_games: int = 512
    rank_eval_sims: int = 16
    rank_eval_max_turns: int = 200
    checkpoint_every: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_iters: int = 20
    max_wall_minutes: float = 30.0
    # Absolute or workspace-relative path to a checkpoint file whose net weights
    # should be used to warm-start a fresh run (iteration counter still starts
    # at 0, no optimizer/replay-buffer carry-over). Only consulted when the run
    # has no existing checkpoints in runs/<run_id>/checkpoints. Cleanly layers
    # on top of the normal resume behavior.
    init_from: str = ""
    league_ckpt_every: int = 5
    league_selfplay_every: int = 3  # every N iters after league has entries, use league opponents
    league_opponent_prob: float = 0.5
    league_max_entries: int = 24
    league_keep_recent: int = 8
    league_rating_games: int = 64
    league_rating_sims: int = 8
    league_rating_matches: int = 4
    rating_random_anchor: float = 1000.0
    rating_heuristic_anchor: float = 2500.0
    # Exploration hyperparameters for self-play root MCTS.
    dirichlet_alpha: float = 0.15  # AlphaZero-style root prior noise; tuned via Optuna
    dirichlet_mix: float = 0.40  # fraction of prior replaced by Dirichlet at root
    # Value-target shaping: per-turn discount of the terminal reward. Values
    # closer to 1 mean less pressure to end the game quickly.
    time_discount: float = 0.995
    # MCTS root Q-value scaling. Following Danihelka et al. (2022), the final
    # improved policy target is `softmax(logits + gumbel + q_scale * q)`; a
    # large q_scale makes the search results dominate the prior once the game
    # has produced signal. Tuned via Optuna: top trials converged on 19-25.
    q_scale: float = 22.0
    use_amp: bool = False
    async_eval: bool = False
    # Wall-clock-based eval interval.  When > 0, eval triggers every N minutes
    # of elapsed wall time instead of every ``eval_every`` iterations.
    eval_every_mins: float = 0.0
    # Number of recent archive checkpoints (iter_*.pt) to keep.  Older ones
    # are deleted after each new checkpoint save.  League checkpoints in the
    # league/ subdirectory are never touched.  0 = keep all (no cleanup).
    keep_recent_checkpoints: int = 3
    # Mixed player-count training.  When non-empty, selfplay rotates through
    # these player counts each iteration (e.g. [2, 3, 4] trains all three).
    # When empty, uses ``num_players`` for every iteration.
    mixed_players: list[int] = dataclasses.field(default_factory=list)


# GPU-optimized defaults applied when the resolved device is CUDA.
# Tuned via Optuna cold-phase search (study gpu-tune-cold, 56 trials).
_GPU_DEFAULTS: dict[str, object] = {
    "selfplay_games": 4096,
    "selfplay_sims": 32,
    "learner_batch": 4096,
    "replay_capacity": 820_000,
    "learner_steps_per_iter": 64,
    "async_eval": True,
}


def apply_device_defaults(cfg: LoopConfig, device: str) -> LoopConfig:
    """Return a new LoopConfig with device-conditional defaults applied.

    Only overrides fields that still hold their LoopConfig dataclass defaults.
    CLI-provided values (which differ from defaults) are preserved.

    GPU defaults (when device starts with "cuda"):
        selfplay_games=4096, selfplay_sims=32, learner_batch=16384,
        replay_capacity=800_000, learner_steps_per_iter=48,
        async_eval=True

    CPU defaults remain unchanged from LoopConfig.__init__.
    """
    if not device.startswith("cuda"):
        # CPU path: return an identical copy without any changes.
        return dataclasses.replace(cfg)

    # Build a dict of overrides: only apply GPU defaults for fields where the
    # user hasn't provided an explicit CLI value (i.e. the field still matches
    # the LoopConfig() default).
    factory_defaults = LoopConfig()
    overrides: dict[str, object] = {}
    for field_name, gpu_value in _GPU_DEFAULTS.items():
        current_value = getattr(cfg, field_name)
        default_value = getattr(factory_defaults, field_name)
        if current_value == default_value:
            overrides[field_name] = gpu_value

    return dataclasses.replace(cfg, **overrides)


def _validate_buffer_capacity(cfg: LoopConfig, run: Run) -> None:
    """Log a warning if replay_capacity < selfplay_games * 200."""
    estimated_max_samples = cfg.selfplay_games * 200
    if cfg.replay_capacity < estimated_max_samples:
        run.event("buffer_capacity_warning", {
            "replay_capacity": cfg.replay_capacity,
            "selfplay_games": cfg.selfplay_games,
            "estimated_max_samples": estimated_max_samples,
            "recommendation": f"Consider replay_capacity >= {estimated_max_samples}",
        }, level="WARN")


def _latest_ckpt(ckpt_dir: pathlib.Path) -> Optional[pathlib.Path]:
    resume_ckpt = ckpt_dir / "latest_resume.pt"
    if resume_ckpt.exists():
        return resume_ckpt
    ckpts = sorted(ckpt_dir.glob("iter_*.pt"))
    return ckpts[-1] if ckpts else None


def _cleanup_old_checkpoints(ckpt_dir: pathlib.Path, keep: int, run: Run) -> None:
    """Delete old archive checkpoints, keeping the most recent *keep*.

    Only removes ``iter_*.pt`` files in the top-level checkpoint directory.
    ``latest_resume.pt`` and everything under ``league/`` are untouched.
    """
    if keep <= 0:
        return
    ckpts = sorted(ckpt_dir.glob("iter_*.pt"))
    to_remove = ckpts[:-keep] if len(ckpts) > keep else []
    for path in to_remove:
        try:
            path.unlink()
            run.event("checkpoint_cleaned", {"path": str(path)})
        except OSError:
            pass


def _eval_priority(row: dict) -> tuple[float, float, float, float, float]:
    heuristic_score = float(
        row.get("rank_winrate_vs_heuristic", row.get("winrate_vs_heuristic", 0.0))
    ) + 0.5 * float(
        row.get("rank_ties_vs_heuristic", row.get("ties_vs_heuristic", 0.0))
    )
    random_score = float(
        row.get("rank_winrate_vs_random", row.get("winrate_vs_random", 0.0))
    ) + 0.5 * float(
        row.get("rank_ties_vs_random", row.get("ties_vs_random", 0.0))
    )
    finished_score = float(
        row.get("rank_finished_vs_heuristic", row.get("finished_vs_heuristic", 0.0))
    )
    avg_finished_step = -float(
        row.get(
            "rank_avg_finished_step_vs_heuristic",
            row.get(
                "avg_finished_step_vs_heuristic",
                row.get("avg_turns_vs_heuristic", float("inf")),
            ),
        )
    )
    max_finished_step = -float(
        row.get(
            "rank_max_finished_step_vs_heuristic",
            row.get(
                "max_finished_step_vs_heuristic",
                row.get("avg_turns_vs_heuristic", float("inf")),
            ),
        )
    )
    return (
        heuristic_score,
        random_score,
        finished_score,
        avg_finished_step,
        max_finished_step,
    )


def _checkpoint_metadata(row: dict | None) -> dict:
    if row is None:
        return {}
    keys = [
        "winrate_vs_random",
        "ties_vs_random",
        "finished_vs_random",
        "avg_turns_vs_random",
        "winrate_vs_heuristic",
        "ties_vs_heuristic",
        "finished_vs_heuristic",
        "avg_turns_vs_heuristic",
        "avg_finished_step_vs_heuristic",
        "max_finished_step_vs_heuristic",
        "winrate_vs_heuristic_opus",
        "ties_vs_heuristic_opus",
        "finished_vs_heuristic_opus",
        "avg_turns_vs_heuristic_opus",
        "rank_winrate_vs_random",
        "rank_ties_vs_random",
        "rank_finished_vs_random",
        "rank_avg_turns_vs_random",
        "rank_winrate_vs_heuristic",
        "rank_ties_vs_heuristic",
        "rank_finished_vs_heuristic",
        "rank_avg_turns_vs_heuristic",
        "rank_avg_finished_step_vs_heuristic",
        "rank_max_finished_step_vs_heuristic",
        "rank_winrate_vs_heuristic_opus",
        "rank_ties_vs_heuristic_opus",
        "rank_finished_vs_heuristic_opus",
        "rank_avg_turns_vs_heuristic_opus",
    ]
    out = {key: row[key] for key in keys if key in row}
    out["score_hint"] = _eval_priority(row)[0]
    return out


def _run_rank_eval(
    net: M.SplendorNet,
    cfg: LoopConfig,
    device: str,
    seed: int,
) -> dict:
    metrics = LAD.evaluate(
        net,
        num_players=cfg.num_players,
        num_games=cfg.rank_eval_games,
        device=device,
        num_sims=cfg.rank_eval_sims,
        max_turns=cfg.rank_eval_max_turns,
        seed=seed,
    )
    return {f"rank_{key}": value for key, value in metrics.items()}


def _rate_new_league_entry(
    league: League,
    entry: dict | None,
    net: M.SplendorNet,
    cfg: LoopConfig,
    device: str,
    seed: int,
) -> dict | None:
    if entry is None:
        return None
    entry_idx = int(entry["idx"])
    entry_entity = f"ckpt:{entry_idx}"
    matches: list[dict] = []
    if cfg.league_rating_games > 0 and cfg.league_rating_matches > 0:
        candidates = league.rating_candidates(
            exclude_idx=entry_idx,
            limit=cfg.league_rating_matches,
        )
        for offset, opp in enumerate(candidates):
            opp_idx = int(opp["idx"])
            opp_name = f"league_{opp_idx}"
            resolved_path = str(league._resolve_path(opp["path"]))
            opponents = {
                opp_name: (
                    lambda path=resolved_path: P.NetPolicy(
                        path,
                        num_sims=cfg.league_rating_sims,
                        device=device,
                    ).choose
                )
            }
            metrics = LAD.evaluate(
                net,
                num_players=cfg.num_players,
                num_games=cfg.league_rating_games,
                opponents=opponents,
                device=device,
                num_sims=cfg.league_rating_sims,
                max_turns=cfg.rank_eval_max_turns,
                seed=seed + offset,
            )
            winrate = float(metrics.get(f"winrate_vs_{opp_name}", 0.0))
            tie_rate = float(metrics.get(f"ties_vs_{opp_name}", 0.0))
            wins = int(round(cfg.league_rating_games * winrate))
            ties = int(round(cfg.league_rating_games * tie_rate))
            losses = max(int(cfg.league_rating_games) - wins - ties, 0)
            league.record_result(
                entry_entity,
                f"ckpt:{opp_idx}",
                float(wins),
                float(losses),
                float(ties),
            )
            matches.append(
                {
                    "opponent_idx": opp_idx,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "score": winrate + 0.5 * tie_rate,
                }
            )
    ratings = league.recompute_ratings()
    updated = league.entry_by_idx(entry_idx)
    if updated is None:
        return None
    return {
        "idx": entry_idx,
        "rating": float(updated.get("rating", ratings.get(entry_entity, 0.0))),
        "games": int(updated.get("games", 0)),
        "matches": matches,
    }


def _is_new_best_eval(metrics_history: list[dict], row: dict) -> bool:
    if "winrate_vs_heuristic" not in row:
        return False
    current = _eval_priority(row)
    best_prior: tuple[float, float, float, float, float] | None = None
    for prev in metrics_history:
        if "winrate_vs_heuristic" not in prev:
            continue
        score = _eval_priority(prev)
        if best_prior is None or score > best_prior:
            best_prior = score
    return best_prior is None or current > best_prior


def run_loop(run: Run, cfg: LoopConfig) -> dict:
    run.write_config_if_missing(dataclasses.asdict(cfg))
    run.event("loop_start", {"config": dataclasses.asdict(cfg)})

    device = resolve_device(cfg.device)
    dev_info = configure_device(device)
    run.event(
        "device_selected",
        {
            "requested": cfg.device,
            "device": device,
            "compile_net": cfg.compile_net,
            **dev_info,
        },
    )

    # Apply device-conditional defaults and validate buffer capacity.
    cfg = apply_device_defaults(cfg, device)
    _validate_buffer_capacity(cfg, run)
    run.event("effective_config", dataclasses.asdict(cfg))

    net = M.SplendorNet(hidden=cfg.hidden, arch=cfg.arch).to(device)
    if cfg.compile_net:
        net.enable_compile()
    optim = make_optimizer(net, lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, device=device)

    # Inject human-flagged training replays into the buffer at startup.
    from .replay_injection import inject_replays_into_buffer
    n_injected = inject_replays_into_buffer(buffer, time_discount=cfg.time_discount)
    if n_injected > 0:
        run.event("replay_injection", {"samples_injected": n_injected})

    # AMP mixed-precision: create GradScaler only when use_amp is True and
    # the resolved device is CUDA.  On CPU, log a warning and skip.
    grad_scaler: Optional[torch.amp.GradScaler] = None
    if cfg.use_amp:
        if device.startswith("cuda"):
            grad_scaler = torch.amp.GradScaler("cuda")
            run.event("amp_enabled", {"device": device})
        else:
            run.event("amp_skipped_cpu", {
                "device": device,
                "reason": "use_amp=True but device is CPU; skipping GradScaler",
            })

    start_iter = 0
    ckpt = _latest_ckpt(run.ckpt_dir)
    if ckpt is not None:
        payload = load_checkpoint_payload(ckpt, map_location=device)
        src_spec = checkpoint_net_spec(payload)
        if src_spec.arch != cfg.arch:
            raise ValueError(
                f"checkpoint arch mismatch: run expects arch={cfg.arch}, checkpoint has arch={src_spec.arch}"
            )
        if src_spec.hidden != cfg.hidden:
            raise ValueError(
                f"checkpoint hidden mismatch: run expects hidden={cfg.hidden}, "
                f"checkpoint has hidden={src_spec.hidden}"
            )
        payload = load_checkpoint(ckpt, net, optim, buffer, map_location=device)
        start_iter = int(payload.get("iteration", 0))
        run.event("loop_resumed", {"from": str(ckpt), "iter": start_iter})
    elif cfg.init_from:
        init_path = pathlib.Path(cfg.init_from)
        if not init_path.is_absolute():
            # Resolve relative to BUILD_WORKING_DIRECTORY (when launched via
            # `bazel run`) or the current working directory.
            import os as _os

            bwd = _os.environ.get("BUILD_WORKING_DIRECTORY")
            if bwd and not init_path.exists():
                init_path = pathlib.Path(bwd) / cfg.init_from
        payload = load_checkpoint_payload(init_path, map_location=device)
        src_spec = checkpoint_net_spec(payload)
        if src_spec.arch != cfg.arch:
            raise ValueError(
                f"init_from arch mismatch: run expects arch={cfg.arch}, checkpoint has arch={src_spec.arch}"
            )
        if src_spec.hidden != cfg.hidden:
            raise ValueError(
                f"init_from hidden mismatch: run expects hidden={cfg.hidden}, "
                f"checkpoint has hidden={src_spec.hidden}"
            )
        # Load weights only; optimizer state and replay buffer stay fresh so
        # this is a pure warm-start of the policy/value function.
        net.load_state_dict(checkpoint_net_state_dict(payload))
        run.event(
            "loop_init_from",
            {
                "from": str(init_path),
                "src_iter": payload.get("iteration"),
                "src_hidden": src_spec.hidden,
                "src_arch": src_spec.arch,
            },
        )

    league = League(
        run.root.parent / "league",
        max_entries=cfg.league_max_entries,
        keep_recent=cfg.league_keep_recent,
        anchors={
            "random": cfg.rating_random_anchor,
            "heuristic": cfg.rating_heuristic_anchor,
        },
    )

    t_start = time.monotonic()
    metrics_history: list[dict] = []
    if run.metrics_path.exists():
        with open(run.metrics_path) as f:
            for line in f:
                try:
                    metrics_history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    decision = "continue"
    cur_iter = start_iter
    last_eval_row: dict | None = None

    # Async eval handle: created when cfg.async_eval is True so that eval
    # runs in a CPU subprocess while the GPU continues selfplay + learning.
    async_eval_handle: Optional[AsyncEvalHandle] = None
    if cfg.async_eval:
        async_eval_handle = AsyncEvalHandle(
            AsyncEvalConfig(
                eval_games=cfg.eval_games,
                eval_sims=cfg.eval_sims,
                eval_max_turns=cfg.eval_max_turns,
                num_players=cfg.num_players,
            )
        )
        run.event("async_eval_enabled", {"config": dataclasses.asdict(cfg)})

    # Wall-clock eval tracking: when eval_every_mins > 0, eval triggers based
    # on elapsed wall time rather than iteration count.
    last_eval_wall_min: float = -999.0  # force initial eval

    # Initial eval on the starting net (baseline for learning curves).
    if cfg.eval_every_mins > 0 and cfg.eval_games > 0:
        if async_eval_handle is not None:
            snapshot = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            async_eval_handle.launch(snapshot, cur_iter, seed=0)
            run.event("async_eval_launched", {"iteration": cur_iter, "reason": "initial_baseline"})
        else:
            eval_metrics = LAD.evaluate(
                net,
                num_players=cfg.num_players,
                num_games=cfg.eval_games,
                device=device,
                num_sims=cfg.eval_sims,
                max_turns=cfg.eval_max_turns,
                seed=0,
            )
            row = {
                "iter": cur_iter,
                "elapsed_min": 0.0,
                **eval_metrics,
            }
            run.metric(row)
            metrics_history.append(row)
            run.event("initial_eval_done", {"iter": cur_iter, **eval_metrics})
        last_eval_wall_min = 0.0

    while True:
        elapsed_min = (time.monotonic() - t_start) / 60.0
        iters_done = cur_iter - start_iter

        # -- Collect completed async eval results at each iteration boundary --
        if async_eval_handle is not None:
            result = async_eval_handle.try_collect()
            if result is not None:
                iter_tag, eval_results = result
                if "error" in eval_results:
                    run.event("async_eval_failed", {
                        "iteration": iter_tag,
                        "error": eval_results["error"],
                    })
                else:
                    run.event("async_eval_results", {
                        "iteration": iter_tag,
                        **eval_results,
                    })
                    run.metric({
                        "iter": iter_tag,
                        "async_eval": True,
                        "elapsed_min": elapsed_min,
                        **eval_results,
                    })

        if iters_done >= cfg.max_iters:
            run.event("loop_max_iters_reached", {"iter": cur_iter})
            break
        if elapsed_min >= cfg.max_wall_minutes:
            run.event("loop_max_wall_reached", {"iter": cur_iter, "elapsed_min": elapsed_min})
            break

        cur_iter += 1

        run.write_heartbeat(
            {
                "iter": cur_iter,
                "phase": "selfplay",
                "elapsed_min": elapsed_min,
                "buffer_size": len(buffer),
            }
        )

        # Determine player count for this iteration.
        if cfg.mixed_players:
            iter_num_players = cfg.mixed_players[(cur_iter - 1) % len(cfg.mixed_players)]
        else:
            iter_num_players = cfg.num_players

        use_league = (
            cfg.league_selfplay_every > 0
            and cur_iter % cfg.league_selfplay_every == 0
            and len(league.list_entries()) > 0
        )
        if use_league:
            sp_metrics = run_league_selfplay(
                net,
                buffer,
                league,
                num_players=iter_num_players,
                num_games=cfg.selfplay_games,
                device=device,
                max_turns=cfg.selfplay_max_turns,
                num_sims=cfg.selfplay_sims,
                seed=cur_iter,
                league_prob=cfg.league_opponent_prob,
                time_discount=cfg.time_discount,
            )
            run.event("league_selfplay_done", {"iter": cur_iter, "num_players": iter_num_players, **sp_metrics})
        else:
            sp_metrics = run_selfplay(
                net,
                buffer,
                num_players=iter_num_players,
                num_games=cfg.selfplay_games,
                device=device,
                max_turns=cfg.selfplay_max_turns,
                num_sims=cfg.selfplay_sims,
                seed=cur_iter,
                time_discount=cfg.time_discount,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_mix=cfg.dirichlet_mix,
                q_scale=cfg.q_scale,
            )
            run.event("selfplay_done", {"iter": cur_iter, "num_players": iter_num_players, **sp_metrics})

        # Train
        train_metrics = {}
        if len(buffer) >= cfg.learner_batch:
            run.write_heartbeat({"iter": cur_iter, "phase": "learner"})
            accum = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            steps = 0
            for _ in range(cfg.learner_steps_per_iter):
                m = learner_step(
                    net, buffer, optim,
                    batch_size=cfg.learner_batch,
                    entropy_bonus=cfg.entropy_bonus,
                    device=device,
                    grad_scaler=grad_scaler,
                )
                for k, v in m.items():
                    accum[k] = accum.get(k, 0.0) + v
                steps += 1
            for k in accum:
                accum[k] /= max(steps, 1)
            train_metrics = accum
            run.event("learner_done", {"iter": cur_iter, **train_metrics})
        else:
            run.event("learner_skipped_empty_buffer", {"iter": cur_iter, "buffer": len(buffer)})

        # Eval
        checkpoint_reasons: list[str] = []
        # Determine whether eval is due: wall-clock mode or iteration mode.
        eval_due = False
        if cfg.eval_every_mins > 0:
            eval_due = (elapsed_min - last_eval_wall_min) >= cfg.eval_every_mins
        else:
            eval_due = cur_iter % cfg.eval_every == 0
        if eval_due:
            last_eval_wall_min = elapsed_min
            if async_eval_handle is not None:
                # -- Async eval path: launch CPU subprocess, don't block GPU --
                run.write_heartbeat({"iter": cur_iter, "phase": "async_eval_launch"})
                snapshot = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                if not async_eval_handle.launch(snapshot, cur_iter, seed=cur_iter * 7):
                    run.event("async_eval_skipped", {
                        "iteration": cur_iter,
                        "reason": "previous eval still active",
                    }, level="WARN")
                else:
                    run.event("async_eval_launched", {"iteration": cur_iter})
            else:
                # -- Synchronous eval path (original behavior) --
                run.write_heartbeat({"iter": cur_iter, "phase": "eval"})
                eval_metrics = LAD.evaluate(
                    net,
                    num_players=cfg.num_players,
                    num_games=cfg.eval_games,
                    device=device,
                    num_sims=cfg.eval_sims,
                    max_turns=cfg.eval_max_turns,
                    seed=cur_iter * 7,
                )
                row = {
                    "iter": cur_iter,
                    "elapsed_min": (time.monotonic() - t_start) / 60.0,
                    "lr": optim.param_groups[0]["lr"],
                    **train_metrics,
                    **eval_metrics,
                    "buffer_size": len(buffer),
                }
                wants_rank_eval = (
                    cfg.rank_eval_games > 0
                    and (cur_iter % cfg.checkpoint_every == 0 or _is_new_best_eval(metrics_history, row))
                )
                if wants_rank_eval:
                    rank_metrics = _run_rank_eval(net, cfg, device, seed=cur_iter * 11)
                    row.update(rank_metrics)
                    run.event("rank_eval_done", {"iter": cur_iter, **rank_metrics})
                is_new_best_eval = _is_new_best_eval(metrics_history, row)
                run.metric(row)
                metrics_history.append(row)
                decision = decide_next_action(metrics_history)
                run.event("eval_done", {"iter": cur_iter, "decision": decision, **eval_metrics})
                last_eval_row = row
                if is_new_best_eval:
                    checkpoint_reasons.append("new_best_eval")
                J.append_entry(
                    run.journal_path,
                    f"iter {cur_iter} eval decision: {decision}",
                    body=f"metrics: {row}\nhistory_len: {len(metrics_history)}",
                )

                if decision == "reduce_lr":
                    for pg in optim.param_groups:
                        pg["lr"] = pg["lr"] * 0.1
                    run.event("lr_reduced", {"new_lr": optim.param_groups[0]["lr"]})
                elif decision in ("stop_regressing", "stop_converged"):
                    run.event("loop_stopping", {"decision": decision})
                    break

        # Checkpoint
        if cur_iter % cfg.checkpoint_every == 0:
            checkpoint_reasons.append("periodic")
        if checkpoint_reasons:
            path = run.ckpt_dir / f"iter_{cur_iter:06d}.pt"
            resume_path = run.ckpt_dir / "latest_resume.pt"
            save_checkpoint(path, net, optim, cur_iter, dataclasses.asdict(cfg), buffer=None)
            save_checkpoint(
                resume_path,
                net,
                optim,
                cur_iter,
                dataclasses.asdict(cfg),
                buffer,
            )
            run.event(
                "checkpoint_saved",
                {
                    "iter": cur_iter,
                    "path": str(path),
                    "resume_path": str(resume_path),
                    "reasons": checkpoint_reasons,
                },
            )
            run.write_state(
                {
                    "iter": cur_iter,
                    "last_checkpoint": str(resume_path),
                    "last_archive_checkpoint": str(path),
                    "decision": decision,
                }
            )

            # Clean up old archive checkpoints to save disk space.
            if cfg.keep_recent_checkpoints > 0:
                _cleanup_old_checkpoints(run.ckpt_dir, cfg.keep_recent_checkpoints, run)

        if cur_iter % cfg.league_ckpt_every == 0:
            entry_eval_row = (
                last_eval_row
                if last_eval_row is not None and int(last_eval_row.get("iter", -1)) == cur_iter
                else None
            )
            if entry_eval_row is None:
                if cfg.rank_eval_games > 0:
                    rank_metrics = _run_rank_eval(net, cfg, device, seed=cur_iter * 13 + 1)
                    entry_eval_row = {"iter": cur_iter, **rank_metrics}
                    run.event("league_rank_eval_done", {"iter": cur_iter, **rank_metrics})
                elif cfg.eval_games > 0:
                    eval_metrics = LAD.evaluate(
                        net,
                        num_players=cfg.num_players,
                        num_games=cfg.eval_games,
                        device=device,
                        num_sims=cfg.eval_sims,
                        max_turns=cfg.eval_max_turns,
                        seed=cur_iter * 13 + 1,
                    )
                    entry_eval_row = {"iter": cur_iter, **eval_metrics}
                    run.event("league_eval_done", {"iter": cur_iter, **eval_metrics})
            league.add_checkpoint(
                net,
                tag=f"i{cur_iter}",
                metadata=_checkpoint_metadata(entry_eval_row),
            )
            league_entry = league.latest_entry()
            if league_entry is not None and entry_eval_row is not None:
                league.record_checkpoint_baselines(
                    int(league_entry["idx"]),
                    entry_eval_row,
                    rank_games=cfg.rank_eval_games,
                    eval_games=cfg.eval_games,
                )
            run.event(
                "league_entry_added",
                {
                    "iter": cur_iter,
                    "league_size": len(league.list_entries()),
                    "league_idx": league_entry["idx"] if league_entry else None,
                },
            )
            rating = _rate_new_league_entry(
                league,
                league_entry,
                net,
                cfg,
                device,
                seed=cur_iter * 17,
            )
            if rating is not None:
                run.event("league_rating_done", {"iter": cur_iter, **rating})

    # -- Collect any remaining async eval results before final checkpoint --
    if async_eval_handle is not None:
        result = async_eval_handle.wait_and_collect()
        if result is not None:
            iter_tag, eval_results = result
            if "error" in eval_results:
                run.event("async_eval_failed", {
                    "iteration": iter_tag,
                    "error": eval_results["error"],
                })
            else:
                run.event("async_eval_results", {
                    "iteration": iter_tag,
                    **eval_results,
                })
                run.metric({
                    "iter": iter_tag,
                    "async_eval": True,
                    "elapsed_min": (time.monotonic() - t_start) / 60.0,
                    **eval_results,
                })
        async_eval_handle.cleanup()

    # -- Final eval at loop end (ensures a data point at the very end) --
    if cfg.eval_every_mins > 0 and cfg.eval_games > 0:
        final_elapsed = (time.monotonic() - t_start) / 60.0
        run.write_heartbeat({"iter": cur_iter, "phase": "final_eval"})
        eval_metrics = LAD.evaluate(
            net,
            num_players=cfg.num_players,
            num_games=cfg.eval_games,
            device=device,
            num_sims=cfg.eval_sims,
            max_turns=cfg.eval_max_turns,
            seed=cur_iter * 7 + 1,
        )
        row = {
            "iter": cur_iter,
            "elapsed_min": final_elapsed,
            "final_eval": True,
            **eval_metrics,
        }
        run.metric(row)
        metrics_history.append(row)
        run.event("final_eval_done", {"iter": cur_iter, **eval_metrics})

    # Final checkpoint + state
    final_path = run.ckpt_dir / f"iter_{cur_iter:06d}.pt"
    resume_path = run.ckpt_dir / "latest_resume.pt"
    save_checkpoint(final_path, net, optim, cur_iter, dataclasses.asdict(cfg), buffer=None)
    save_checkpoint(resume_path, net, optim, cur_iter, dataclasses.asdict(cfg), buffer)
    run.write_state(
        {
            "iter": cur_iter,
            "last_checkpoint": str(resume_path),
            "last_archive_checkpoint": str(final_path),
            "decision": decision,
            "finished_burst": True,
        }
    )
    run.event("loop_end", {"iter": cur_iter, "decision": decision})
    return {"iter": cur_iter, "decision": decision}
