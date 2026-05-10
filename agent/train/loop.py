"""The iterative train-evaluate-improve loop.

Given a Run, config, and optional checkpoint, performs:
1. A bounded training burst (`--max-iters` iterations or `--max-wall-minutes`
   of wall clock). Each iteration: self-play burst -> buffer write -> N learner
   steps.
2. Every `checkpoint_every` iterations, runs a unified eval (async on CPU)
   that plays 512 mixed-player-count games at 64 sims against a pool of
   opponents (random, heuristic, heuristic_opus, + 4 league checkpoints).
   All pairwise results feed into the rating system.
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
import random as stdlib_random
import time
from typing import Optional

import torch

from ..net import model as M
from ..obs.run import Run
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
from .learner import make_optimizer, step as learner_step
from .league import League
from .league_selfplay import run_league_selfplay
from .replay_buffer import ReplayBuffer
from .profiling import PhaseTimer
from .selfplay import run_selfplay
from .unified_eval import UnifiedEvalConfig, UnifiedEvalHandle


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
    # Per-player-count turn scaling: max_turns = turns_per_player * num_players.
    # When set > 0, overrides selfplay_max_turns with a per-PC computed value.
    selfplay_turns_per_player: int = 50
    replay_capacity: int = 600_000
    learner_batch: int = 256
    learner_steps_per_iter: int = 192
    entropy_bonus: float = 0.015
    checkpoint_every: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_iters: int = 20
    max_wall_minutes: float = 30.0
    init_from: str = ""
    league_selfplay_every: int = 3
    league_opponent_prob: float = 0.5
    league_opponent_sims: int = 4  # Opponents don't need full MCTS; saves ~5x on league iters
    league_distinct_opponents: int = 4  # Cap distinct nets per episode for GPU batching
    league_max_entries: int = 24
    league_keep_recent: int = 8
    rating_random_anchor: float = 1000.0
    # Exploration hyperparameters for self-play root MCTS.
    dirichlet_alpha: float = 0.15
    dirichlet_mix: float = 0.40
    time_discount: float = 1.0
    q_scale: float = 22.0
    use_amp: bool = False
    # Number of recent archive checkpoints to keep. 0 = keep all.
    keep_recent_checkpoints: int = 3
    # Mixed player-count training. Selfplay rotates through these counts.
    mixed_players: list[int] = dataclasses.field(default_factory=list)
    # --- Unified eval config ---
    eval_games: int = 512
    eval_sims: int = 64
    eval_max_turns: int = 200
    # Per-player-count turn scaling for eval: max_turns = eval_turns_per_player * num_players.
    # When set > 0, overrides eval_max_turns with a per-PC computed value.
    eval_turns_per_player: int = 60
    eval_weight_2p: float = 1.0
    eval_weight_3p: float = 1.0
    eval_weight_4p: float = 1.0
    eval_league_opponents: int = 4


# GPU-optimized defaults applied when the resolved device is CUDA.
# AMP and compile together give ~1.7x self-play throughput on attn/256.
# See agent/scripts/profile_selfplay.py for benchmarks.
_GPU_DEFAULTS: dict[str, object] = {
    "selfplay_games": 4096,
    "selfplay_sims": 32,
    "learner_batch": 4096,
    "replay_capacity": 820_000,
    "learner_steps_per_iter": 64,
    "use_amp": True,
    "compile_net": True,
}


def apply_device_defaults(cfg: LoopConfig, device: str, explicit_fields: set[str] | None = None) -> LoopConfig:
    """Return a new LoopConfig with device-conditional defaults applied.

    Fields listed in ``explicit_fields`` are never overridden, even if they
    match the factory default. This allows callers to distinguish "user didn't
    specify" from "user explicitly set to the default value".
    """
    if not device.startswith("cuda"):
        return dataclasses.replace(cfg)

    explicit_fields = explicit_fields or set()
    factory_defaults = LoopConfig()
    overrides: dict[str, object] = {}
    for field_name, gpu_value in _GPU_DEFAULTS.items():
        if field_name in explicit_fields:
            continue
        current_value = getattr(cfg, field_name)
        default_value = getattr(factory_defaults, field_name)
        if current_value == default_value:
            overrides[field_name] = gpu_value

    return dataclasses.replace(cfg, **overrides)


def _validate_buffer_capacity(cfg: LoopConfig, run: Run) -> None:
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


def _league_trigger(cur_iter: int, num_players: int, every: int) -> bool:
    """Decide whether this (iter, num_players) should use league opponents.

    Previously this was a plain ``cur_iter % every == 0`` check, which was
    order-sensitive when ``mixed_players`` and ``every`` shared factors. For
    example, with ``mixed_players=[2,3,4]`` and ``every=3``, the trigger
    only fired on iterations assigned to the third element (4p), so 2p and
    3p never got league opponents. That starved 3p self-play of opponent
    diversity and was the dominant cause of the 3p-below-random plateau.

    Hashing ``(iter, num_players)`` gives each PC its own independent coin
    flip so the trigger rate per PC is ~1/every regardless of how many
    entries are in ``mixed_players``.
    """
    if every <= 1:
        return every == 1
    # md5 is overkill but cheap and gives a well-distributed mix.
    import hashlib
    h = hashlib.md5(f"{cur_iter}|{num_players}".encode()).digest()
    # Use first 8 bytes -> unsigned 64-bit int.
    n = int.from_bytes(h[:8], "big")
    return (n % every) == 0


def _cleanup_old_checkpoints(ckpt_dir: pathlib.Path, keep: int, run: Run) -> None:
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


def _get_league_opponent_paths(league: League, count: int, seed: int) -> list[str]:
    """Sample `count` random league checkpoint paths for unified eval."""
    # Only consider entries whose checkpoint files still exist on disk.
    # Pruned entries remain in the manifest for rating history but their
    # .pt files are deleted, so we must filter them out here.
    entries = [e for e in league.list_entries() if league._entry_available(e)]
    if not entries:
        return []
    rng = stdlib_random.Random(seed)
    sampled = rng.sample(entries, min(count, len(entries)))
    paths = []
    for entry in sampled:
        resolved = str(league._resolve_path(entry["path"]))
        paths.append(resolved)
    return paths


def _apply_eval_results_to_league(
    league: League,
    eval_results: dict,
    eval_agent_entity: str,
    league_entry_map: dict[str, int],
) -> dict[str, float]:
    """Record pairwise results from unified eval into the league rating system.

    Args:
        league: The League instance.
        eval_results: Results dict from unified eval with "pairwise" key.
        eval_agent_entity: Entity ID for the eval agent (e.g. "ckpt:2487").
        league_entry_map: Maps policy name -> league entry idx
            (e.g. {"league_0": 2480, "league_1": 2475, ...})

    Returns:
        Updated ratings dict.
    """
    pairwise = eval_results.get("pairwise", [])

    # Map policy names to entity IDs for the rating system
    def _to_entity(name: str) -> str:
        if name == "eval_agent":
            return eval_agent_entity
        if name in league_entry_map:
            return f"ckpt:{league_entry_map[name]}"
        # Bot names map directly to rating anchors
        return name

    for result in pairwise:
        winner_entity = _to_entity(result["winner"])
        loser_entity = _to_entity(result["loser"])
        weight = float(result["weight"])
        num_players = int(result.get("num_players", 2))
        # Record as fractional wins with player count
        league.record_result(winner_entity, loser_entity, weight, 0.0, 0.0,
                             num_players=num_players)

    ratings = league.recompute_ratings()
    return ratings



def run_loop(run: Run, cfg: LoopConfig, explicit_fields: set[str] | None = None) -> dict:
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

    cfg = apply_device_defaults(cfg, device, explicit_fields=explicit_fields)
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

    # AMP mixed-precision
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

    # Unified eval handle: runs async on CPU subprocess
    eval_handle = UnifiedEvalHandle(
        UnifiedEvalConfig(
            total_games=cfg.eval_games,
            num_sims=cfg.eval_sims,
            max_turns=cfg.eval_max_turns,
            turns_per_player=cfg.eval_turns_per_player,
            weight_2p=cfg.eval_weight_2p,
            weight_3p=cfg.eval_weight_3p,
            weight_4p=cfg.eval_weight_4p,
            league_opponents=cfg.eval_league_opponents,
        )
    )
    # Track which league entries were used in the last eval launch
    _last_eval_league_map: dict[str, int] = {}
    # Track which entity the eval was launched for (so results go to the right entry)
    _last_eval_entity: str = ""

    # Per-phase timing instrumentation
    phase_timer = PhaseTimer(device=device, sync_cuda=True)

    # -- On resume: check if the current checkpoint has eval games. If not,
    # kick off an async eval immediately so we don't lose a checkpoint_every
    # worth of iterations before getting a rating for it. --
    if start_iter > 0:
        resume_tag = f"i{start_iter}"
        resume_entry = None
        for entry in league.list_entries():
            if entry.get("tag") == resume_tag:
                resume_entry = entry
                break
        if resume_entry is not None and int(resume_entry.get("games", 0)) == 0:
            resume_entity = f"ckpt:{resume_entry['idx']}"
            run.event("resume_eval_needed", {
                "iteration": start_iter,
                "entity": resume_entity,
                "reason": "resumed checkpoint has no eval games",
            })
            snapshot = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            league_paths = _get_league_opponent_paths(
                league, cfg.eval_league_opponents, seed=start_iter * 31
            )
            entries = league.list_entries()
            _last_eval_league_map = {}
            for i, lpath in enumerate(league_paths):
                for entry in entries:
                    if str(league._resolve_path(entry["path"])) == lpath:
                        _last_eval_league_map[f"league_{i}"] = int(entry["idx"])
                        break
            _last_eval_entity = resume_entity
            eval_handle.launch(snapshot, league_paths, start_iter, seed=start_iter * 7)
            run.event("resume_eval_launched", {"iteration": start_iter, "entity": resume_entity})

    while True:
        elapsed_min = (time.monotonic() - t_start) / 60.0
        iters_done = cur_iter - start_iter

        # -- Collect completed unified eval results at each iteration boundary --
        result = eval_handle.try_collect()
        if result is not None:
            iter_tag, eval_results = result
            if "error" in eval_results:
                run.event("unified_eval_failed", {
                    "iteration": iter_tag,
                    "error": eval_results["error"],
                })
            else:
                # Record pairwise results into the league rating system
                eval_agent_entity = _last_eval_entity
                if not eval_agent_entity:
                    league_entry = league.latest_entry()
                    eval_agent_entity = (
                        f"ckpt:{league_entry['idx']}" if league_entry else "eval_agent"
                    )
                ratings = _apply_eval_results_to_league(
                    league, eval_results, eval_agent_entity, _last_eval_league_map
                )
                eval_rating = ratings.get(eval_agent_entity, 0.0)

                metrics = eval_results.get("metrics", {})
                row = {
                    "iter": iter_tag,
                    "elapsed_min": elapsed_min,
                    "rating": eval_rating,
                    **metrics,
                }
                run.metric(row)
                metrics_history.append(row)
                run.event("unified_eval_done", {
                    "iteration": iter_tag,
                    "rating": eval_rating,
                    "league_opponents": eval_results.get("league_opponents", []),
                    **metrics,
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

        # Compute per-PC max_turns for selfplay
        sp_max_turns = (
            cfg.selfplay_turns_per_player * iter_num_players
            if cfg.selfplay_turns_per_player > 0
            else cfg.selfplay_max_turns
        )

        use_league = (
            cfg.league_selfplay_every > 0
            and _league_trigger(cur_iter, iter_num_players, cfg.league_selfplay_every)
            and len(league.list_entries()) > 0
        )
        if use_league:
            with phase_timer.phase("selfplay"):
                sp_metrics = run_league_selfplay(
                    net,
                    buffer,
                    league,
                    num_players=iter_num_players,
                    num_games=cfg.selfplay_games,
                    device=device,
                    max_turns=sp_max_turns,
                    num_sims=cfg.selfplay_sims,
                    seed=cur_iter,
                    league_prob=cfg.league_opponent_prob,
                    time_discount=cfg.time_discount,
                    opponent_sims=cfg.league_opponent_sims,
                    max_distinct_opponents=cfg.league_distinct_opponents,
                )
            run.event("league_selfplay_done", {"iter": cur_iter, "num_players": iter_num_players, **sp_metrics})
        else:
            with phase_timer.phase("selfplay"):
                sp_metrics = run_selfplay(
                    net,
                    buffer,
                    num_players=iter_num_players,
                    num_games=cfg.selfplay_games,
                    device=device,
                    max_turns=sp_max_turns,
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
            with phase_timer.phase("learner"):
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

        # Emit per-iteration phase timing
        iter_times = phase_timer.iter_summary()
        if iter_times:
            run.event("iter_timing", {"iter": cur_iter, **{f"t_{k}_s": round(v, 3) for k, v in iter_times.items()}})

        # -- Checkpoint + Unified Eval (every checkpoint_every iterations) --
        if cur_iter % cfg.checkpoint_every == 0:
            # Save checkpoint
            path = run.ckpt_dir / f"iter_{cur_iter:06d}.pt"
            resume_path = run.ckpt_dir / "latest_resume.pt"
            save_checkpoint(path, net, optim, cur_iter, dataclasses.asdict(cfg), buffer=None)
            save_checkpoint(resume_path, net, optim, cur_iter, dataclasses.asdict(cfg), buffer)
            run.event(
                "checkpoint_saved",
                {
                    "iter": cur_iter,
                    "path": str(path),
                    "resume_path": str(resume_path),
                    "reasons": ["periodic"],
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
            if cfg.keep_recent_checkpoints > 0:
                _cleanup_old_checkpoints(run.ckpt_dir, cfg.keep_recent_checkpoints, run)

            # Add checkpoint to league
            league.add_checkpoint(net, tag=f"i{cur_iter}")
            league_entry = league.latest_entry()
            run.event(
                "league_entry_added",
                {
                    "iter": cur_iter,
                    "league_size": len(league.list_entries()),
                    "league_idx": league_entry["idx"] if league_entry else None,
                },
            )

            # Launch unified eval (async on CPU)
            run.write_heartbeat({"iter": cur_iter, "phase": "unified_eval_launch"})

            # If a previous eval is still running, block until it completes
            # so we don't create an eval backlog.
            if eval_handle.is_active():
                run.event("unified_eval_waiting", {
                    "iteration": cur_iter,
                    "reason": "blocking until previous eval completes",
                })
                prev_result = eval_handle.wait_and_collect()
                if prev_result is not None:
                    prev_iter_tag, prev_eval_results = prev_result
                    if "error" in prev_eval_results:
                        run.event("unified_eval_failed", {
                            "iteration": prev_iter_tag,
                            "error": prev_eval_results["error"],
                        })
                    else:
                        prev_eval_agent_entity = _last_eval_entity
                        if not prev_eval_agent_entity:
                            prev_league_entry = league.latest_entry()
                            prev_eval_agent_entity = (
                                f"ckpt:{prev_league_entry['idx']}" if prev_league_entry else "eval_agent"
                            )
                        ratings = _apply_eval_results_to_league(
                            league, prev_eval_results, prev_eval_agent_entity, _last_eval_league_map
                        )
                        prev_eval_rating = ratings.get(prev_eval_agent_entity, 0.0)
                        prev_metrics = prev_eval_results.get("metrics", {})
                        row = {
                            "iter": prev_iter_tag,
                            "elapsed_min": (time.monotonic() - t_start) / 60.0,
                            "rating": prev_eval_rating,
                            **prev_metrics,
                        }
                        run.metric(row)
                        metrics_history.append(row)
                        run.event("unified_eval_done", {
                            "iteration": prev_iter_tag,
                            "rating": prev_eval_rating,
                            "league_opponents": prev_eval_results.get("league_opponents", []),
                            **prev_metrics,
                        })

            snapshot = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

            # Sample league opponents for this eval
            league_paths = _get_league_opponent_paths(
                league, cfg.eval_league_opponents, seed=cur_iter * 31
            )
            # Build map from policy name -> league entry idx for result recording
            entries = league.list_entries()
            _last_eval_league_map = {}
            for i, lpath in enumerate(league_paths):
                for entry in entries:
                    if str(league._resolve_path(entry["path"])) == lpath:
                        _last_eval_league_map[f"league_{i}"] = int(entry["idx"])
                        break

            # Record which entity this eval is for (the checkpoint we just added)
            _last_eval_entity = f"ckpt:{league_entry['idx']}" if league_entry else "eval_agent"

            eval_handle.launch(snapshot, league_paths, cur_iter, seed=cur_iter * 7)
            run.event("unified_eval_launched", {"iteration": cur_iter})

    # -- Collect any remaining eval results before exit --
    result = eval_handle.wait_and_collect()
    if result is not None:
        iter_tag, eval_results = result
        if "error" in eval_results:
            run.event("unified_eval_failed", {
                "iteration": iter_tag,
                "error": eval_results["error"],
            })
        else:
            eval_agent_entity = _last_eval_entity
            if not eval_agent_entity:
                league_entry = league.latest_entry()
                eval_agent_entity = (
                    f"ckpt:{league_entry['idx']}" if league_entry else "eval_agent"
                )
            ratings = _apply_eval_results_to_league(
                league, eval_results, eval_agent_entity, _last_eval_league_map
            )
            eval_rating = ratings.get(eval_agent_entity, 0.0)
            metrics = eval_results.get("metrics", {})
            row = {
                "iter": iter_tag,
                "elapsed_min": (time.monotonic() - t_start) / 60.0,
                "rating": eval_rating,
                **metrics,
            }
            run.metric(row)
            metrics_history.append(row)
            run.event("unified_eval_done", {
                "iteration": iter_tag,
                "rating": eval_rating,
                **metrics,
            })
    eval_handle.cleanup()

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
