# Agent Operations Guide

Operational knowledge for training, tuning, and deploying the Splendor RL agent.
This file is intended for AI assistants and developers working on the project.

## Current Model

- **Architecture**: attn/192 (attention-based, 192 hidden, ~407K params)
- **Best checkpoint**: `agent/runs/real30_v9/checkpoints/iter_005290.pt`
- **Active training run**: `real30_v10` (warm-started from v9 iter 5290)
- **Shared league**: `agent/runs/league/` (shared across all runs)

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
| time_discount | 0.995 | |
| async_eval | True | Eval runs on CPU subprocess, doesn't block GPU |

### Key Findings

- **Learning rate for warm-start must be low** (3e-5). Using 3e-4 on a pre-trained model causes catastrophic forgetting within 2-4 hours.
- **torch.compile is slower** for this model size. flat/192 runs at 1059 games/s without compile vs 730 games/s with compile. The model is too small for kernel fusion to help.
- **attn architecture OOMs** at selfplay_games > 1024 on RTX 3080 Ti (11.6GB). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Shared league** lives at `agent/runs/league/` and persists across runs. New runs inherit historical opponents.
- **Checkpoint cleanup**: `keep_recent_checkpoints=3` auto-deletes old `iter_*.pt` files. League checkpoints are never touched.

### Starting a Training Run

```bash
# Warm-start from existing checkpoint (recommended)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m agent.scripts.train \
    --run-id real30_v10 --device cuda --arch attn \
    --max-iters 2000 --max-wall-minutes 720 \
    --selfplay-games 1024 --selfplay-sims 32 \
    --learner-batch 4096 --learner-steps-per-iter 64 \
    --replay-capacity 820000 --lr 3e-5 \
    --entropy-bonus 0.015 --dirichlet-alpha 0.15 \
    --dirichlet-mix 0.40 --q-scale 22.0 --time-discount 0.995 \
    --init-from agent/runs/real30_v9/checkpoints/iter_005290.pt

# Resume an existing run (picks up from latest_resume.pt automatically)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m agent.scripts.train \
    --run-id real30_v10 --device cuda --arch attn \
    --max-iters 5000 --max-wall-minutes 720 \
    --selfplay-games 1024 --selfplay-sims 32 \
    --learner-batch 4096 --learner-steps-per-iter 64 \
    --replay-capacity 820000 --lr 3e-5 \
    --entropy-bonus 0.015 --dirichlet-alpha 0.15 \
    --dirichlet-mix 0.40 --q-scale 22.0 --time-discount 0.995
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
| Flag game script | `agent/scripts/flag_game.py` |
