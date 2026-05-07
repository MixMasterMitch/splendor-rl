# Splendor RL Agent

A Splendor-playing agent trained via Gumbel AlphaZero self-play with a fully
vectorized game engine in PyTorch. Supports 2-4 players with a single shared
network.

See `RULES.md` for the complete rules spec and data format. Card and noble data
live in `env/splendor_cards.csv` and `env/splendor_nobles.csv`.

## Layout

- `env/` - card/noble data, single-game engine, batched PyTorch engine, action encoding
- `net/` - policy/value network and state encoder
- `search/` - batched Gumbel MCTS
- `train/` - self-play, replay buffer, learner loop, league, health checks
- `obs/` - durable run/journal/metrics logging
- `eval/` - reference bots (random + heuristic) and ladder evaluation
- `scripts/` - CLI entrypoints (train, eval, status, smoke_train)
- `tests/` - unit tests and single-vs-batched parity tests
- `runs/<run_id>/` - per-training-run artifacts (not committed)

## Setup

From the repo root:

```bash
pip install -e ".[dev]"
```

This installs the project in editable mode with all dependencies (torch, numpy,
pyyaml, tqdm, pytest).

## Running

All commands should be run from the **repo root** directory.

### Quick smoke test

```bash
python -m agent.scripts.smoke_train
```

### Training

```bash
python -m agent.scripts.train \
    --run-id my_run --num-players 2 --max-iters 40 --max-wall-minutes 120
```

### Iterative training with auto-decisions

```bash
python -m agent.scripts.iterate \
    --run-id my_run --max-bursts 8 --burst-max-iters 10 --burst-max-wall-minutes 30
```

### Evaluate a checkpoint

```bash
python -m agent.scripts.eval_ckpt \
    --ckpt agent/runs/real30_v9/checkpoints/latest_resume.pt \
    --num-games 256 --num-sims 16
```

### Check run status

```bash
python -m agent.scripts.status --run-id my_run
```

### Show league ratings

```bash
python -m agent.scripts.league_table --run-id real30_v9
```

## Building on prior progress

Each `--run-id` has its own checkpoint directory; re-running the same run-id
resumes from the latest checkpoint. To start a *new* experiment that still
builds on a prior run's weights, pass `--init-from <path-to-checkpoint.pt>`.

## Tests

```bash
pytest
```

## Recommended training settings

### GPU (RTX 3080 Ti, attn/192)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m agent.scripts.train \
    --run-id my_run --device cuda --arch attn \
    --selfplay-games 1024 --selfplay-sims 32 \
    --learner-batch 4096 --learner-steps-per-iter 64 \
    --replay-capacity 820000 --lr 3e-5 \
    --entropy-bonus 0.015 --dirichlet-alpha 0.15 \
    --dirichlet-mix 0.40 --q-scale 22.0 --time-discount 0.995 \
    --max-iters 2000 --max-wall-minutes 720
```

### CPU (flat/192, for quick experiments)

```
--device cpu
--arch flat
--hidden 192
--selfplay-games 512 --selfplay-sims 16
--learner-batch 256 --learner-steps-per-iter 128
--eval-every 2 --eval-games 256 --eval-sims 4
--max-iters 40 --max-wall-minutes 120
```

### Hyperparameter tuning

```bash
python3 -m agent.scripts.tune --study-name my-sweep --n-trials 30 \
    --minutes-per-trial 5 --device cuda --rating-games 512
```

### Flagging bad games for training

```bash
python3 -m agent.scripts.flag_game --game-id <id_from_dynamodb>
```

See `AGENTS.md` for detailed operational knowledge including tuning results,
architecture comparison data, and known issues.

## Dependencies

Core: `torch`, `numpy`, `pytest`, `pyyaml`, `tqdm`, `optuna`
