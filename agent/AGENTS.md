# Agent Operations Guide

Operational knowledge for training, tuning, and deploying the Splendor RL agent.
This file is intended for AI assistants and developers working on the project.

## Current Model

- **Architecture**: attn/256 with per-PC heads (attention-based, 256 hidden, ~900K params)
- **Best checkpoint**: `agent/runs/league/ckpt_02699_i1200.pt` (league rating 3224, shared-head era)
- **Active training run**: `attn256_v4` (per-PC heads, warm-started from peak via migration)
- **Shared league**: `agent/runs/league/` (shared across all runs, auto-migrates old checkpoints)

## GPU Training

### Tuned Hyperparameters (from Optuna cold-phase, 56 trials)

These are the best-known values for GPU training from scratch:

| Parameter | Value | Notes |
|-----------|-------|-------|
| selfplay_games | 4096 (flat) / 1024 (attn) | attn OOMs at 2048+ on 11.6GB |
| selfplay_sims | 32 | Higher = better targets, fewer iters/min |
| learner_batch | 4096 | Top trials converged here |
| learner_steps_per_iter | 64 | |
| replay_capacity | 820,000 | Just above selfplay_games * 200 threshold |
| lr | 3e-5 (warm-start) / 3e-4 (cold) | 3e-4 causes catastrophic forgetting on pre-trained models |
| q_scale | 22.0 | Top trials: 19-25 range |
| dirichlet_alpha | 0.15 | |
| dirichlet_mix | 0.40 | |
| entropy_bonus | 0.015 | |
| time_discount | 1.0 | Disabled; stall penalty handles "don't dawdle" |
| use_amp | True (GPU) | ~30% faster value forward; auto-enabled on CUDA |
| compile_net | True (GPU) | ~15% on top of AMP for attn arch; auto-enabled on CUDA |
| league_opponent_sims | 4 | Opponents use fewer sims; saves ~5× on league iters |
| async_eval | True | Eval runs on CPU subprocess, doesn't block GPU |

### Key Findings

- **Learning rate for warm-start must be low** (3e-5). Using 3e-4 on a pre-trained model causes catastrophic forgetting within 2-4 hours.
- **AMP + torch.compile + TF32 are the GPU defaults** (auto-enabled on CUDA). Combined they give ~1.7× self-play throughput. Disable with `--no-amp` / `--no-compile-net` if needed.
- **Self-play bottleneck is the MCTS value forward pass** (~60% of wall time without optimizations). With AMP+compile+TF32 the bottleneck shifts to `engine.apply` (branchless game logic), which is already near-optimal.
- **torch.compile helps attn/256 but hurt flat/192**. The attn architecture has enough compute per kernel for fusion to pay off; flat/192 is too small. The GPU defaults enable compile because the production model is attn/256.
- **attn architecture OOMs** at selfplay_games > 1024 on RTX 3080 Ti (11.6GB). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Shared league** lives at `agent/runs/league/` and persists across runs. New runs inherit historical opponents.
- **Checkpoint cleanup**: `keep_recent_checkpoints=3` auto-deletes old `iter_*.pt` files. League checkpoints are never touched.

### GPU Performance Tuning

Profiled with `python -m agent.scripts.profile_selfplay` (RTX 3080 Ti, attn/256, 256 games, 32 sims):

| Configuration | Total (100 turns) | Dominant phase | Speedup |
|---|---|---|---|
| Baseline | 2.83s | value_fwd 59% | 1.0× |
| AMP only | 2.17s | value_fwd 44% | 1.3× |
| Compile only | 2.21s | value_fwd 51% | 1.3× |
| Compile + AMP | 1.84s | value_fwd 35% | 1.5× |
| **Compile + AMP + TF32** | **1.66s** | engine_apply 43% | **1.7×** |

All three are now enabled by default on CUDA via `configure_device()` (TF32) and `_GPU_DEFAULTS` (AMP, compile).

**Remaining optimization opportunities** (diminishing returns):
- Pipeline self-play and learning on separate CUDA streams (moderate complexity)
- Reduce `selfplay_sims` from 32 to 16 (trades target quality for speed)
- Fuse the encode + forward into a single compiled graph (requires refactoring)

### League Selfplay Performance

League selfplay iterations are **8× slower** than regular selfplay because each
distinct opponent checkpoint runs a separate MCTS expansion. With 24 league
entries and `league_prob=0.5`, a 4p game has ~24 tiny-batch MCTS calls per turn.

Fix: `league_opponent_sims=4` (default) uses only 4 sims for opponents vs 32
for the main agent. Opponents don't need high-quality search — they just provide
diverse opposition. Expected speedup: ~5× on league iterations (from ~45s to ~10s
for 4p/1024 games).

### Starting a Training Run

```bash
# Create a migrated checkpoint for a new architecture version
python3 -m agent.scripts.migrate_checkpoint \
    --input agent/runs/league/ckpt_02699_i1200.pt \
    --output agent/runs/attn256_v4/checkpoints/iter_000000.pt

# Start training with per-PC heads (v4+)
# AMP, torch.compile, and TF32 are auto-enabled on CUDA.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m agent.scripts.train \
    --run-id attn256_v4 --device cuda --arch attn \
    --max-iters 1000 --max-wall-minutes 240 \
    --selfplay-games 1024 --selfplay-sims 32 \
    --learner-batch 4096 --learner-steps-per-iter 64 \
    --replay-capacity 820000 --lr 2e-5 \
    --entropy-bonus 0.015 --dirichlet-alpha 0.15 \
    --dirichlet-mix 0.40 --q-scale 22.0 --time-discount 1.0 \
    --eval-games 1024 --checkpoint-every 100 \
    --init-from agent/runs/attn256_v4/checkpoints/iter_000000.pt

# Resume an existing run (picks up from latest_resume.pt automatically)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m agent.scripts.train \
    --run-id attn256_v4 --device cuda --arch attn \
    --max-iters 1000 --max-wall-minutes 240 \
    --selfplay-games 1024 --selfplay-sims 32 \
    --learner-batch 4096 --learner-steps-per-iter 64 \
    --replay-capacity 820000 --lr 2e-5 \
    --entropy-bonus 0.015 --dirichlet-alpha 0.15 \
    --dirichlet-mix 0.40 --q-scale 22.0 --time-discount 1.0 \
    --eval-games 1024 --checkpoint-every 100

# Disable optimizations if debugging numerical issues:
#   --no-amp --no-compile-net
```

## Hyperparameter Tuning

### Optuna Tuning Harness

```bash
# Cold phase: wide search from scratch
python3 -m agent.scripts.tune \
    --study-name gpu-tune-cold --n-trials 40 \
    --minutes-per-trial 5 --device cuda \
    --rating-games 512 --trial-prefix cold_trial

# Warm phase: narrow search from checkpoint
python3 -m agent.scripts.tune \
    --study-name gpu-tune-warm --n-trials 25 \
    --minutes-per-trial 15 --device cuda \
    --narrow-ranges --init-from <checkpoint.pt> \
    --rating-games 512 --trial-prefix warm_trial
```

### Key Design Decisions

- **Time-budget mode** (`--minutes-per-trial`): Each trial trains for a fixed wall-clock duration. This naturally rewards configs with better throughput.
- **SQLite persistence**: Results stored in `agent/runs/optuna_<study>.db`. Studies can be resumed across restarts with `load_if_exists=True`.
- **Trial cleanup**: Trial directories are deleted after rating is computed to save disk space.
- **Rating objective**: 512 games per opponent (random, heuristic, heuristic_opus, optionally best ML checkpoint). Anchor: random=1000.

## Architecture Comparison

```bash
python3 -m agent.scripts.arch_comparison --device cuda --minutes 60 --eval-every-mins 5
```

Runs flat×{128,192,256} and attn×{128,192,256} for 1 hour each with periodic eval. Results saved to `agent/runs/arch_comparison_results.json`.

### Results (1-hour runs)

| Config | Final WR Heuristic | Final WR Opus | Notes |
|--------|-------------------|---------------|-------|
| flat_192 | 87.9% | 68.2% | Best overall in 1 hour |
| flat_128 | 85.9% | 66.0% | Close second, fastest throughput |
| attn_256 | 70.1% | 44.1% | Still climbing at 60 min, highest ceiling |
| flat_256 | 18.4% | 8.2% | Diverged (lr too high) |
| attn_128 | 18.6% | 8.0% | Diverged (lr too high) |
| attn_192 | 23.2% | 8.6% | Diverged (lr too high) |

**Conclusion**: flat_192 converges fastest. attn_256 has higher ceiling but needs lower lr and more time. Current production model is attn/192 (from v9 training).

## Training Replay Injection

Flag games where the AI played badly for inclusion in training data:

```bash
# Pull a game from DynamoDB and save to training_replays/
python3 -m agent.scripts.flag_game --game-id <game_id>

# Games are stored in agent/training_replays/*.json
# They are automatically injected into the replay buffer at training startup
```

Both players' moves are included as training samples:
- Winner's moves get positive value targets
- Loser's moves get negative value targets
- Time-discounted like selfplay samples

## Web App / Play Server

- **Default sims**: 64 (set in `play/service.py`, passed via `num_sims` in game creation body)
- **DynamoDB tables** (us-west-2):
  - Games: `SplendorStack-GamesTableB32AB610-1C9JML169FTJT`
  - Users: `SplendorStack-UsersTable9725E9C8-13NUJEB0F08E3`
  - GSI: `user_sub-updated_at-index`

### Architecture: Two-Call Action Flow (No Polling)

The human action flow uses two sequential requests — NOT polling:

1. `POST /api/games/{id}/action` — applies ONLY the human's move. Returns immediately (<100ms) with the updated game state showing the human's action in the log, updated scores, tokens, board, etc. Status will be `"ai_thinking"` if it's now the AI's turn, or `"human_turn"` if a sub-phase (discard/noble pick) is needed.

2. `POST /api/games/{id}/step-ai` — runs all AI moves synchronously until it's the human's turn again (or game ends). This can take 5-15s for LLM agents. Returns the final state.

The client calls #1, immediately updates the UI with the result (human sees their move reflected), then calls #2 while showing "AI is thinking…". When #2 returns, the full state replaces the intermediate view.

**Do NOT collapse these back into a single call or introduce polling.** The split ensures the human's action is visually confirmed instantly, even when AI takes seconds. Both calls work correctly on Lambda (synchronous, no background threads, no persistent state between invocations).

## Fetching Game Data from DynamoDB

Games played on the deployed Lambda app are stored in DynamoDB (not locally).

### Quick lookup by game_id

```bash
aws dynamodb get-item \
  --table-name "SplendorStack-GamesTableB32AB610-1C9JML169FTJT" \
  --key '{"game_id": {"S": "<GAME_ID>"}}' \
  --region us-west-2 --output json \
  | python3 -c "import json,sys; item=json.load(sys.stdin)['Item']; print(json.dumps(json.loads(item['data']['S']), indent=2))"
```

### Save to local file for analysis

```bash
aws dynamodb get-item \
  --table-name "SplendorStack-GamesTableB32AB610-1C9JML169FTJT" \
  --key '{"game_id": {"S": "<GAME_ID>"}}' \
  --region us-west-2 --output json \
  | python3 -c "
import json, sys
data = json.loads(json.load(sys.stdin)['Item']['data']['S'])
with open('play/play_data/games/<GAME_ID>.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Saved. Steps: {len(data[\"steps\"])}, Status: {data[\"status\"]}')
"
```

### Game record structure

The `data` field (JSON string) contains:
- `game_id`, `num_players`, `human_seat`, `seed`, `num_sims`
- `seat_models`: `{"<seat>": {"id", "label", "kind", "ckpt", "rating", ...}}`
- `steps[]`: each has `step`, `player`, `phase`, `action`, `action_name`, `action_detail`, `legal_actions`, `state_after`
- `initial_state`: full board snapshot at game start
- `rating_update`: `{old_rating, new_rating, delta, per_opponent[]}`
- `status`: `"completed"` | `"human_turn"` | `"ai_thinking"` | `"aborted"`
- `user_sub`: username who owns the game

### Replaying a game locally

To recreate a DDB game with the local server, you need:
1. The `seed` (determines deck permutation)
2. The `initial_state` (full board snapshot)
3. The `seat_models` (which checkpoint was used)
4. The `num_sims` setting

The `GameSession` constructor accepts `initial_state_override` to reproduce the exact same board. The checkpoint must be available locally (check `agent/runs/league/` or download from S3 bucket in `seat_models._s3_bucket`/`_s3_key`).

## League Maintenance

### Cleaning inactive agents

Over time the league accumulates inactive checkpoints that no longer participate
in self-play but still inflate the results table and slow down rating fits.

```bash
# Preview what would be removed
python -m agent.scripts.clean_league --dry-run

# Actually clean (backs up to league.json.bak first)
python -m agent.scripts.clean_league
```

This removes all entries with `"active": false`, deletes their `.pt` files from
disk, prunes pairwise records involving those entries, and recomputes ratings
from the remaining data. Anchors (random, heuristic, heuristic_opus) and
floating entities (e.g. bedrock_claude_sonnet) are always preserved.

## Known Issues

- **Token hoarding**: The agent sometimes takes tokens when at 10, forcing a discard. This is a training data gap — the agent rarely sees 10-token states in selfplay. Flagging bad games via replay injection helps.
- **attn OOM**: The attention architecture can't batch 2048+ games on 11.6GB VRAM. Use 1024 games for attn.
- **Checkpoint resume with different config**: Resuming a run with different `replay_capacity` or `learner_batch` than the checkpoint was saved with causes tensor size mismatches. Use `--init-from` for a clean warm-start instead.

## File Locations

| What | Where |
|------|-------|
| Training runs | `agent/runs/<run_id>/` |
| Shared league | `agent/runs/league/` |
| Training replays | `agent/training_replays/` |
| Optuna databases | `agent/runs/optuna_<study>.db` |
| Tuning scripts | `agent/scripts/tune.py`, `agent/scripts/arch_comparison.py` |
| Profiling script | `agent/scripts/profile_selfplay.py` |
| Flag game script | `agent/scripts/flag_game.py` |
| Clean league script | `agent/scripts/clean_league.py` |
