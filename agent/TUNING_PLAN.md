# attn256_v1 Tuning Plan

Working notes and the agreed plan for maximizing `attn256_v1` performance
and balancing play across 2p/3p/4p. Written 2026-05-09.

## 1. Context

### 1.1 Current model and training state

- **Focus model**: `attn256_v1` (hidden=256, arch=attn). Other runs (`real30_*`)
  are abandoned and can be ignored.
- **Active run status** (latest check, 2026-05-09 ~18:27 UTC): iter **2739** /
  5000, ~**12.0h wall**, `lr=2e-5`, `selfplay 1024g x 32sims`, device=cuda,
  heartbeat active, still training.
- **League peak**: rating ~3224 at ckpt idx 2699 (tag i1200) and at idx 2746
  (tag i2375). Top-10 entries span only ~41 Elo (3183-3224) — within rating
  noise of each other.
- **Current eval rating**: oscillating **~2990** (range 2886-3158 over the
  last 15 evals, mean 3038). **~200 Elo below peak, not recovering.**
  Latest eval (iter 2725) at 2992.
- **Per-PC winrates (trailing 15 evals, iters 2375-2725)**:
  - 2p: mean ~68% (range 65-72%). Stable.
  - 3p: mean ~**26%** (range 22-35%). **Persistently below the 33%
    random-baseline against league opponents.** No trend toward recovery.
  - 4p: mean ~38% (range 32-44%). Above random baseline of 25% but modest.
- **Per-PC avg turns** (latest 15 evals): 2p ~65, 3p ~98, 4p ~122.
  - 2p: plenty of headroom under current cap of 160.
  - 3p: avg 98 against cap 160 — 60% of cap, fine.
  - 4p: avg 122 against cap 160 — 76% of cap, meaningful tail truncation.
- **Training trajectory summary**: since iter ~2375 the run has burned
  ~6 hours (iters 2375-2725) producing no sustained improvement. Combined
  rating oscillation is ~272 Elo peak-to-trough over that window, dwarfing
  any rating signal smaller than ~50 Elo.

The 3p-below-random pattern is **exactly the same** as when the plan was
first drafted — ~164 additional iters did not change it. This is strong
confirmation that the root cause is a signal/training bug (time_discount,
eval weighting, or selfplay distribution) rather than anything tuning can
fix. Moving forward without the Phase 1 immediate fixes is wasted GPU time.

### 1.2 Why 3p looks broken

Ranked by likelihood; all still hypotheses until Phase 1 runs:

1. **`time_discount=0.995` biases against 3p/4p.** 3p games average
   ~100 turns, so terminal reward is scaled by `0.995^100 ≈ 0.61` vs
   `0.995^64 ≈ 0.73` for 2p. Over thousands of iters the net drifts toward
   2p patterns. Agreed fix: set `time_discount=1.0`. Stall penalty already
   handles "don't dawdle" without distorting the value target.

2. **`eval_weight_3p=0.5` halves 3p eval allocation.** It's purely a game
   allocation knob (not a training signal), but undersampling 3p means:
   - Noisier 3p winrate (~205/102/205 per 512-game eval round).
   - Under-weighted 3p contribution in the combined league rating, since
     `compute_ratings` weights per-PC calibrated ratings by actual games
     played. Agreed fix: set `eval_weight_3p=1.0` so the split is
     ~170/170/170.

3. **Self-play distribution and kingmaker dynamics.** 3p has different
   strategic structure than 2p. If self-play opponent mix doesn't match
   eval opponent mix, the net sees a biased 3p training distribution.
   Monitor after (1) and (2); revisit if 3p winrate doesn't recover.

### 1.3 What we learned about the rating system

Investigated `CALIBRATION_SCALE = {2: 1.0, 3: 2.0, 4: 3.0}` in
`agent/train/ranking.py`. These multipliers scale raw per-PC
Bradley-Terry ratings onto a common axis for the combined rating.

A quick weighted-least-squares fit across 133 league entries produced
empirical scales of roughly `1.0 / 1.43 / 0.65` (95% CI excluded both
2.0 and 3.0). But this fit cannot distinguish two hypotheses:

- **A (game structure)**: 3p/4p BT ratings are structurally
  dilated/inflated for pure game-theoretic reasons.
- **B (training asymmetry)**: ML agents are systematically weaker at 3p
  and inflate against random at 4p because of how they trained.

With only one cross-PC anchor (`random=1000`) and most of the rating
graph being ML-vs-ML, both hypotheses predict the same data pattern.
Blindly updating the constants to 1.43 / 0.65 would bake the current
training asymmetry into the rating metric — circular, since we then
tune against a metric that reflects the asymmetry we're trying to fix.

### 1.4 Decision on ratings

- **Keep** `CALIBRATION_SCALE = {2: 1.0, 3: 2.0, 4: 3.0}`.
- **Keep** the combined rating as the game-count-weighted average of
  calibrated per-PC ratings. Useful summary.
- **Don't** use combined rating as the sole tuning objective when we care
  about balance across PCs.
- **For league culling**: look at the per-PC ratings (`rating_2p`,
  `rating_3p`, `rating_4p`) not just the combined rating. Preserve agents
  that are best at *some* PC even if weak elsewhere — they're valuable
  diversity in the self-play opponent pool. Criterion: keep an entry if
  `max(rating_2p, rating_3p, rating_4p)` is top-K at any PC.
- **For auto-tuning objectives**: use `min(rating_2p, rating_3p, rating_4p)`
  (or equivalent per-PC winrate floor against a fixed panel). This
  directly rewards balance without depending on whether the scale
  multipliers are right.

## 2. Immediate fix actions

Apply these before any tuning work. Each is a small, reversible change.

| Action | File(s) | Change |
|---|---|---|
| Disable time discount on terminal rewards | `agent/train/selfplay.py` and/or `agent/train/loop.py` | Set `time_discount=1.0` default; verify value targets no longer multiply by `time_discount^(end_step - sample_step)` on terminal reward. Keep stall penalty as the only "don't dawdle" signal. |
| Balance eval across PCs | `agent/train/loop.py` (`LoopConfig`), `agent/train/unified_eval.py` | Set `eval_weight_3p=1.0` (was 0.5). Splits ~170/170/170 instead of ~205/102/205 per 512-game eval. |
| Restore to peak before continuing | workflow | Stop training on top of the declining trajectory. The peak ckpt is `idx 2699 (i1200)`, rating 3224. All Phase 2 warm-starts use this ckpt. |
| Update league cull criterion | `agent/train/league.py` (wherever cull logic lives) | Preserve entries that are top-K at *any* per-PC rating, not only those with top combined rating. |
| Scale `max_turns` by player count | `agent/train/loop.py` (`LoopConfig`), `agent/train/unified_eval.py` (`UnifiedEvalConfig`), call sites in `loop.py` and `unified_eval.py:_run_multiplayer_games` | Replace scalar `selfplay_max_turns=160` / `eval_max_turns=200` with per-PC caps derived from `50 * num_players` (selfplay) and `60 * num_players` (eval, 20% buffer). **No engine changes needed** — `max_turns` is a scalar loop bound at the Python game-runner level, and each selfplay iteration already runs one PC (`iter_num_players` selected from `mixed_players`), so scalar-per-PC is sufficient. Suggested shape: add `selfplay_turns_per_player=50` / `eval_turns_per_player=60` to configs, compute `max_turns = pc * turns_per_player` at call sites. |

**Rationale for per-PC `max_turns`**: with scalar caps, 2p is wildly
over-provisioned (avg 65 vs cap 160) while 4p is cramped (avg 122 vs
cap 160) — 4p games get artificially truncated into stall penalties,
injecting noise into exactly the PC we already under-train on.
`50*N` scales with natural game length, tightens 2p to ~1.5x average,
and gives 4p the headroom it needs.

Expected outcomes after these five fixes, running 200-400 iters from
idx 2699:

- 3p winrate trajectory turns from declining to flat or rising. If it
  doesn't, the cause is deeper than time_discount and we revisit
  hypothesis 3 (self-play distribution).
- Terminal reward sign is unchanged; policy shouldn't destabilize.
- Eval wall time per cycle stays roughly the same (still 512 total
  games), just redistributed.
- 4p game completion rate rises (fewer artificial truncations).
  2p eval wall time drops slightly from the tighter cap.

## 3. Phased plan

### Phase 1 — Stop the bleeding, fix the signal

**Goal**: confirm the immediate fixes actually help before committing to
any tuning. Cheap, fast, one-shot.

1. Apply the four immediate fixes above.
2. Warm-start from `ckpt_02699_i1200.pt` into a fresh run
   (e.g. `attn256_v2`). Train 200-400 iters (~1-3 hours GPU).
3. Watch per-PC winrates per eval:
   - 3p winrate trend over last 10 evals: should be flat-or-up.
   - 2p and 4p winrates: shouldn't regress more than a few percentage
     points.
   - Combined rating: if it stabilizes above ~3100, call this phase done.
4. Snapshot the per-PC rating profile of the resulting checkpoint; this
   becomes the baseline for Phase 2 warm-start.

**Exit criterion**: 3p winrate against league opponents consistently
above 35% (above random baseline, not just parity) across 5+ consecutive
evals. If this doesn't happen, loop back and investigate self-play
3p dynamics before moving on.

### Phase 2 — Narrow warm-start Optuna sweep

**Goal**: fine-tune the fixed model for maximum balanced strength.
This is the "real" autotuning pass.

**Base**: whichever ckpt closed Phase 1 with good 3p behavior.

**Discard** the existing `optuna_gpu-tune-cold.db` — its trials were
short cold-starts in a different training regime; results don't transfer
to warm-start fine-tuning.

**Search space** (6 dims, narrow ranges — use `tune.py --narrow-ranges`
or a new `--warm-narrow` mode):

| Param | Range | Notes |
|---|---|---|
| `lr` | log-uniform [5e-6, 5e-5] | Narrow around 2e-5; go *lower*, not higher, on a warm-start. |
| `entropy_bonus` | [0.005, 0.03] | Current 0.015 is reasonable, explore nearby. |
| `dirichlet_alpha` | [0.10, 0.35] | Exploration noise concentration. |
| `dirichlet_mix` | [0.20, 0.45] | Fraction of prior replaced by noise. |
| `q_scale` | [10, 30] | MCTS root Q weighting. |
| `time_discount` | fixed at 1.0 | Locked in by immediate fixes. |

**Fixed** (not in search space):
- `hidden=256, arch=attn` — architecture is locked.
- `selfplay_games=1024, selfplay_sims=32` — current GPU-friendly values.
- `learner_batch=16384, replay_capacity=800000,
  learner_steps_per_iter≈48` — existing GPU defaults.
- `weight_decay=1e-4`, `use_amp=false` (AMP separately evaluated).

**Trial protocol**:
- Warm-start with `--init-from <peak_ckpt_from_phase_1>`.
- 200 iters per trial (~45-60 min wall).
- Terminal eval: 1024 games, weights `1.0/1.0/1.0` across 2p/3p/4p,
  opponent pool = `{random, heuristic, heuristic_opus, 3 random league
  peers, Phase-1 baseline ckpt as anchor}`.
- **Primary objective**: mean over the trial's last 5 evals of
  `min(calibrated_2p, calibrated_3p, calibrated_4p)`. This is the
  balance-floor metric.
- **Secondary (tracked, not optimized)**: combined rating, per-PC
  winrates, avg game length.
- Optionally run a multi-objective Optuna study with two objectives
  (`mean_combined_rating`, `min_per_pc_rating`) and look at the Pareto
  front. First pass single-objective is fine.

**Budget**: 25-40 trials. ~25-40 hours of GPU time. Feasible over a
weekend.

**Signal check**: your top-10 league band is ~41 Elo wide. To reliably
rank trials you need per-trial Elo noise well under ~15 Elo:
- 1024 games / 6 opponents averaged over last 5 evals ≈ ~10 Elo noise.
- Per-PC winrate noise at 256+ games per PC is ±3-4%, sufficient to
  detect the kind of balance improvements we care about.

### Phase 3 — In-run adaptive schedule (PBT-of-one)

**Goal**: beyond one-shot tuning, make training self-correct during
long runs. Replaces manual checkpoint-babysitting.

Add a lightweight callback inside `run_loop` that every N iters
(e.g. N=50) reads the last K rows of `metrics.jsonl` and:

- If combined rating slope < 0 for 200 consecutive iters → halve `lr`.
  Log an event describing the adjustment. Cap at 2-3 halvings total to
  avoid drifting to zero-lr.
- If min-per-PC winrate slope < 0 or the min falls below ~0.40 for
  100 iters → bump `entropy_bonus` by 20% and temporarily boost the
  underperforming PC's weight in `mixed_players` rotation for the next
  100 iters.
- If combined rating has improved but min-per-PC is flat → reduce
  weight on the best PC in `mixed_players` to force redistribution.

**Design notes**:
- All decisions logged as Run events (same `events.log` path) for later
  analysis.
- No extra processes; uses metrics the async eval already produces.
- Reversibility: any adjustment can be undone by the callback in later
  iterations; nothing written is permanent.
- Expected code footprint: ~80-120 lines in a new
  `agent/train/adaptive_schedule.py` module, hooked into `run_loop`
  via a callback interface.

**Trigger conditions are intentionally conservative**: an adaptation
every ~50 iters at most, with noise-aware thresholds. The goal is
steady course correction, not reactive oscillation.

**Validation**: run one 1000-iter training session with adaptation
enabled and one without, both warm-started from the same ckpt. Compare
terminal per-PC profiles. If adaptation-on doesn't beat adaptation-off
by at least 20 Elo on the min-per-PC metric, the scheduler needs
rework before further use.

## 4. Metrics to report in `evaluate_training.py`

Extend the existing script to print:

- Current combined rating, rating trend.
- Per-PC ratings and winrates for the most recent checkpoint.
- `min(rating_2p, rating_3p, rating_4p)` — the balance floor.
- `max - min` per-PC rating spread — the imbalance gap.
- League top-K per PC (top 3 at 2p, top 3 at 3p, top 3 at 4p) so we can
  see which agents are strongest per dimension.
- Any adaptive-schedule events from Phase 3 that fired during the
  evaluated window.

## 5. Open questions / future work

- **Anchors across PCs**: pin `heuristic_opus` at a fixed rating across
  2p/3p/4p to tie down the rating graph better. Deferred; not needed
  for the plan above but would make combined ratings more comparable
  across model families.
- **AMP (mixed precision)**: separate evaluation — one-shot A/B on
  throughput and numerical stability. Not in the tuning search space.
- **Architecture scaling**: hidden=384 or hidden=512. Out of scope for
  this plan; evaluate only after Phase 2 finds the fine-tune ceiling
  at hidden=256.
- **Periodic rating re-calibration**: rather than fixed constants,
  re-fit `CALIBRATION_SCALE` from the league every N iters and log the
  drift. Still doesn't disentangle game-structure from training-bias,
  but at least makes the assumption visible. Consider after Phase 3.
