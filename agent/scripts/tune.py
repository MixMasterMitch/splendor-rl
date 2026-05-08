"""Optuna hyperparameter tuning harness for Splendor RL.

Usage:
    python -m agent.scripts.tune --study-name my-sweep --n-trials 20 --iters-per-trial 10 --device cuda

Each trial runs a short training burst, computes a rating via fit_anchored_ratings,
and reports it as the Optuna objective.  Supports fresh and warm-start phases
with configurable wide/narrow search ranges.
"""

from __future__ import annotations

import argparse
import functools
import sys
from typing import Optional

import torch

import optuna


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Splendor RL Optuna hyperparameter tuning")
    p.add_argument(
        "--study-name",
        type=str,
        default="splendor-tune",
        help="Optuna study name (default: splendor-tune).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run (default: 20).",
    )
    p.add_argument(
        "--iters-per-trial",
        type=int,
        default=10,
        help="Training iterations per trial (default: 10). Ignored when --minutes-per-trial is set.",
    )
    p.add_argument(
        "--minutes-per-trial",
        type=float,
        default=0,
        help="Wall-clock minutes per trial. When >0, uses time budget instead of --iters-per-trial.",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Training device: cpu, cuda, or auto (default: cpu).",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (default: in-memory).",
    )
    p.add_argument(
        "--init-from",
        type=str,
        default="",
        help="Checkpoint path for warm-start trials (default: none).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="agent/runs",
        help="Base directory for trial run directories (default: agent/runs).",
    )
    p.add_argument(
        "--narrow-ranges",
        action="store_true",
        help="Use narrow search ranges for fine-tuning sweeps.",
    )
    p.add_argument(
        "--best-checkpoint",
        type=str,
        default=None,
        help="Path to best ML checkpoint for eval as a fourth opponent.",
    )
    p.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players per game (default: 2).",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=192,
        help="Hidden layer size for SplendorNet (default: 192).",
    )
    p.add_argument(
        "--arch",
        type=str,
        choices=["attn", "flat"],
        default="flat",
        help="Network architecture (default: flat).",
    )
    p.add_argument(
        "--rating-games",
        type=int,
        default=512,
        help="Games per opponent for rating computation (default: 512). Higher = less noisy.",
    )
    p.add_argument(
        "--trial-prefix",
        type=str,
        default="tune_trial",
        help="Directory prefix for trial runs (default: tune_trial).",
    )
    return p


def define_search_space(trial: optuna.Trial, narrow: bool = False) -> dict:
    """Sample hyperparameters from the search space.

    Wide ranges (default) cover both throughput and learning quality params.
    Narrow ranges (--narrow-ranges) restrict to a tighter neighbourhood
    suitable for fine-tuning sweeps; throughput params are fixed at GPU defaults.
    """
    if narrow:
        # Throughput params fixed at GPU defaults
        selfplay_games = 4096
        selfplay_sims = 32
        learner_batch = 16384
        replay_capacity = 800_000
        learner_steps_per_iter = 48

        # Narrow learning quality ranges for fine-tuning
        lr = trial.suggest_float("lr", 1e-4, 1e-3)
        entropy_bonus = trial.suggest_float("entropy_bonus", 0.0, 0.02)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.15, 0.5)
        dirichlet_mix = trial.suggest_float("dirichlet_mix", 0.15, 0.35)
        q_scale = trial.suggest_float("q_scale", 5.0, 20.0)
        time_discount = trial.suggest_float("time_discount", 0.99, 1.0)
    else:
        # Wide throughput ranges
        selfplay_games = trial.suggest_categorical("selfplay_games", [1024, 2048, 4096])
        selfplay_sims = trial.suggest_categorical("selfplay_sims", [8, 16, 32, 64])
        learner_batch = trial.suggest_categorical("learner_batch", [4096, 8192, 16384, 32768])
        replay_capacity = trial.suggest_categorical(
            "replay_capacity", [400_000, 600_000, 800_000, 1_000_000]
        )
        learner_steps_per_iter = trial.suggest_int("learner_steps_per_iter", 16, 128)

        # Wide learning quality ranges
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        entropy_bonus = trial.suggest_float("entropy_bonus", 0.0, 0.05)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.03, 1.0, log=True)
        dirichlet_mix = trial.suggest_float("dirichlet_mix", 0.1, 0.5)
        q_scale = trial.suggest_float("q_scale", 1.0, 30.0)
        time_discount = trial.suggest_float("time_discount", 0.98, 1.0)

    return {
        "selfplay_games": selfplay_games,
        "selfplay_sims": selfplay_sims,
        "learner_batch": learner_batch,
        "replay_capacity": replay_capacity,
        "learner_steps_per_iter": learner_steps_per_iter,
        "lr": lr,
        "entropy_bonus": entropy_bonus,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_mix": dirichlet_mix,
        "q_scale": q_scale,
        "time_discount": time_discount,
    }


def compute_rating_objective(
    net: "M.SplendorNet",
    num_players: int,
    num_games: int = 256,
    num_sims: int = 4,
    device: str = "cpu",
    best_checkpoint_path: Optional[str] = None,
    seed: int = 42,
) -> float:
    """Run eval games and compute rating via fit_anchored_ratings.

    Opponents: random, heuristic, heuristic_opus, and optionally best_checkpoint.
    Returns the fitted rating of the trial's agent.
    """
    from agent.eval import bots as B
    from agent.eval import heuristic_opus as HO
    from agent.eval import ladder as LAD
    from agent.net import model as M
    from agent.search import gumbel_mcts as G
    from agent.train import checkpointing as CK
    from agent.train import ranking as RK

    AGENT_ENTITY = "trial_agent"

    # --- Build opponent map ------------------------------------------------
    opponents: dict = {
        "random": lambda: B.RandomBot(seed=seed).choose,
        "heuristic": lambda: B.HeuristicBot().choose,
        "heuristic_opus": lambda: HO.HeuristicOpusV15().choose,
    }

    if best_checkpoint_path is not None:
        best_net, _ = CK.load_net_from_checkpoint(
            best_checkpoint_path, map_location=device
        )
        best_net.to(device)
        best_net.eval()

        def _best_ckpt_choose(engine):  # type: ignore[no-untyped-def]
            with torch.no_grad():
                act, _ = G.gumbel_root_act(
                    engine, best_net, num_sims=num_sims
                )
            return act

        opponents["best_checkpoint"] = lambda: _best_ckpt_choose

    # --- Run evaluation ----------------------------------------------------
    metrics = LAD.evaluate(
        net,
        num_players=num_players,
        num_games=num_games,
        opponents=opponents,
        device=device,
        num_sims=num_sims,
        seed=seed,
    )

    # --- Convert eval metrics to match results for rating fitting -----------
    # evaluate() returns fractional winrate/ties per opponent.
    # Total games per opponent = num_players * max(1, num_games // num_players)
    per_seat = max(1, num_games // num_players)
    total_per_opponent = float(num_players * per_seat)

    match_results: list[dict] = []
    for opp_name in opponents:
        winrate = metrics.get(f"winrate_vs_{opp_name}", 0.0)
        tie_rate = metrics.get(f"ties_vs_{opp_name}", 0.0)

        wins_agent = winrate * total_per_opponent
        ties = tie_rate * total_per_opponent
        wins_opp = total_per_opponent - wins_agent - ties

        RK.add_match_result(
            match_results,
            a=AGENT_ENTITY,
            b=opp_name,
            wins_a=wins_agent,
            wins_b=wins_opp,
            ties=ties,
        )

    # --- Fit ratings ---------------------------------------------------------
    ratings = RK.fit_anchored_ratings(
        match_results,
        anchors=dict(RK.DEFAULT_ANCHORS),
    )

    return ratings.get(AGENT_ENTITY, 0.0)


def trial_fn(
    trial: optuna.Trial,
    base_cfg: dict,
    output_dir: str,
    iters_per_trial: int,
    minutes_per_trial: float,
    narrow: bool,
    best_checkpoint: Optional[str],
    rating_games: int = 512,
    trial_prefix: str = "tune_trial",
) -> float:
    """Single Optuna trial: sample params, train, compute rating."""
    import logging
    import shutil

    from agent.net import model as M
    from agent.obs.run import Run
    from agent.train.checkpointing import load_net_from_checkpoint
    from agent.train.loop import LoopConfig, run_loop

    logger = logging.getLogger(__name__)

    try:
        # 1. Sample hyperparameters from the search space.
        sampled = define_search_space(trial, narrow=narrow)

        # 2. Merge sampled params with base config; sampled values take priority.
        merged = {**base_cfg, **sampled}

        # 3. Configure the training budget.
        if minutes_per_trial > 0:
            # Time-budget mode: set a high iter cap and let wall time be the
            # binding constraint.  Disable mid-training eval, league, and
            # checkpointing so every iteration is pure train.
            merged["max_wall_minutes"] = minutes_per_trial
            merged["max_iters"] = 999_999
            merged["checkpoint_every"] = 999_999
            merged["league_selfplay_every"] = 0
            merged["eval_games"] = 0
        else:
            merged["max_iters"] = iters_per_trial

        # 4. If base_cfg specifies init_from (warm-start), carry it through.
        if base_cfg.get("init_from"):
            merged["init_from"] = base_cfg["init_from"]

        # 5. Build LoopConfig from the merged dict.
        cfg = LoopConfig(**merged)

        # 6. Create a Run directory for this trial (clean slate).
        trial_run_id = f"{trial_prefix}_{trial.number:03d}"
        import pathlib
        trial_dir = pathlib.Path(output_dir) / trial_run_id
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        run = Run(trial_run_id, runs_root=output_dir)

        # 7. Train.
        run_loop(run, cfg)

        # 8. Load the final net weights and compute rating objective.
        ckpt_path = run.ckpt_dir / "latest_resume.pt"
        net, _ = load_net_from_checkpoint(ckpt_path, map_location="cpu")
        net.eval()

        rating = compute_rating_objective(
            net,
            num_players=cfg.num_players,
            num_games=rating_games,
            device="cpu",
            best_checkpoint_path=best_checkpoint,
        )

        logger.info("Trial %d finished with rating %.1f", trial.number, rating)
        run.close()

        # 10. Clean up trial directory to save disk space — the rating is
        # persisted in the Optuna database, so we don't need the artifacts.
        import shutil as _shutil
        if trial_dir.exists():
            _shutil.rmtree(trial_dir, ignore_errors=True)

        # 9. Return the rating as the objective value.
        return rating

    except KeyboardInterrupt:
        raise  # Let Ctrl+C propagate
    except Exception as exc:
        logging.getLogger(__name__).error("Trial %d failed: %s", trial.number, exc)
        return float("-inf")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for splendor-tune."""
    args = build_parser().parse_args(argv)

    # 1. Build base config dict from CLI args.
    base_cfg: dict = {
        "device": args.device,
        "num_players": args.num_players,
        "hidden": args.hidden,
        "arch": args.arch,
    }
    if args.init_from:
        base_cfg["init_from"] = args.init_from

    # 2. Create Optuna study.
    storage = args.storage
    if storage is None:
        # Default to SQLite so results persist across runs.
        storage = f"sqlite:///{args.output_dir}/optuna_{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # 3. Bind trial_fn with the base config and other fixed args.
    objective = functools.partial(
        trial_fn,
        base_cfg=base_cfg,
        output_dir=args.output_dir,
        iters_per_trial=args.iters_per_trial,
        minutes_per_trial=args.minutes_per_trial,
        narrow=args.narrow_ranges,
        best_checkpoint=args.best_checkpoint,
        rating_games=args.rating_games,
        trial_prefix=args.trial_prefix,
    )

    # 4. Run optimization.
    study.optimize(objective, n_trials=args.n_trials)

    # 5. Print summary table of top-5 trials ranked by rating (descending).
    completed = [t for t in study.trials if t.value is not None]
    ranked = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    if ranked:
        # Collect all param names from the first trial for consistent columns.
        param_names = list(ranked[0].params.keys()) if ranked[0].params else []

        # Header
        header = f"{'Trial':>6}  {'Rating':>10}"
        for name in param_names:
            header += f"  {name:>22}"
        print()
        print("=" * len(header))
        print("Top-5 Trials by Rating")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Rows
        for trial in ranked:
            row = f"{trial.number:>6}  {trial.value:>10.1f}"
            for name in param_names:
                val = trial.params.get(name, "")
                if isinstance(val, float):
                    row += f"  {val:>22.6g}"
                else:
                    row += f"  {val!s:>22}"
            print(row)

        print("=" * len(header))
    else:
        print("\nNo completed trials to display.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
