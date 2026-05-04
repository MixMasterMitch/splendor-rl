# heuristic-opus: candidate development log and final ratings

The "heuristic-opus" Splendor agent was developed iteratively across 15
candidates. Each candidate self-played a round-robin tournament against the
existing reference bots (`random`, `heuristic`) and the previous candidates,
fitting anchored Bradley-Terry ratings (random=1000, heuristic=2500). All
candidate-tournament games are kept separate from the main training league.

The strongest candidate by aggregate rating across all evaluation pools is
**`heuristic_opus_v15`**.

## Final aggregate ratings (anchored)

From a 7-candidate tournament with 48 games per matchup across 2/3/4-player
games (total 1152 games per entity, seed=7777):

| Entity | Aggregate | 2p | 3p | 4p |
| --- | ---: | ---: | ---: | ---: |
| heuristic_opus_v15 | **2644.8** | 2680.8 | **2684.5** | 2574.5 |
| heuristic_opus_v13 | 2636.2 | 2666.7 | 2659.0 | 2587.3 |
| heuristic_opus_v14 | 2623.8 | 2659.3 | 2636.6 | 2580.0 |
| heuristic_opus_v8  | 2623.2 | 2659.3 | 2623.7 | 2591.0 |
| heuristic_opus_v10 | 2619.9 | 2663.0 | 2630.2 | 2570.8 |
| heuristic_opus_v9  | 2619.9 | **2684.6** | 2633.9 | 2546.2 |
| heuristic_opus_v3  | 2565.4 | 2582.5 | 2581.0 | 2536.2 |
| heuristic          | 2500.0 | 2500.0 | 2500.0 | 2500.0 |
| random             | 1000.0 | 1000.0 | 1000.0 | 1000.0 |

A separate 4-candidate run including the top trained ML checkpoint
(`real30_v9` greedy, no MCTS) shows V15 ahead of the net by ~60 ELO at 2p:

| Entity | 2p |
| --- | ---: |
| heuristic_opus_v8  | 2719.3 |
| heuristic_opus_v15 | 2668.4 |
| net:real30_v9:877  | 2625.6 |

The top heuristic candidates and the trained net are all within ~100 ELO of
each other, consistent with a tight skill ceiling for handcrafted heuristics
augmented with shallow planning.

## Candidate progression

* **V1 -- "Tall, target-driven greedy"**: Scores affordable cards by points
  + PPT, picks a single point-bearing target, steers token-taking toward it.
* **V2 -- denial reserves**: Reserve high-PV cards an opponent could buy
  next turn (threat threshold 4 PV).
* **V3 -- endgame mode**: Switch to "shortest path to 15" once any seat
  has 11+ points; aggressive point-buying + leader denials.
* **V4 -- one-ply lookahead**: Top-K legal actions evaluated on a cloned
  engine. Significant CPU cost; competitive but slower.
* **V5 -- player-count-aware "wide"**: Buys 0-PV cards for noble progress;
  regressed (over-corrected away from PPT focus).
* **V7 -- pc-scaled denial**: V3 + threat threshold scaled by 4 *
  (num_players-1). Strong at 3p/4p; identical to V3 at 2p.
* **V8 -- bridge buys + post-distance take-tokens**: Buy a 0-PV affordable
  card whose bonus strictly shortens the chosen target's `turns_to_afford`
  by >=1; pick take actions by minimizing post-take target distance.
  +50 ELO over V3.
* **V9 -- 2-card race-to-15 path planner**: Search 1- and 2-card plans
  reaching 15 PV; pick the plan with the smallest cumulative
  turns_to_afford. Strong at 3p, slightly weaker at 2p/4p than V8.
* **V10 -- selective denial + noble bridge**: V8 + skip a denial reserve
  if the threatened opponent has 2+ alternative point-bearing affordable
  buys; secondary 0-PV bridge buy that progresses a near-claimable noble.
* **V11 -- self-target reserve**: Reserve our own contested target for
  gold + lock-in. Regressed (too eager to spend actions on reserves).
* **V12 -- pc-adaptive (V8/V9/V10)**: Dispatch by num_players to the
  per-pc strongest. Marginal vs the components.
* **V13 -- V10 + V9 path planner**: Stack V10's denial/bridge structure
  with V9's path-target. Strong at 2p/3p.
* **V14 -- contention-aware target**: V10 + penalize targets opponents
  could buy sooner than us. Helps 3p, slightly weaker overall.
* **V15 -- final**: V13 at 2p/3p + V10 at 4p. Best aggregate by combining
  multi-card path planning where it shines (tighter races) with simpler
  single-card targeting where chaos breaks plans.
* **V16 -- V8 at 3p variant (regression)**: V13 at 2p, V8 at 3p, V10 at 4p.
  Hypothesized that V13's path planner hurts at 3p. 96-game tournament
  showed V16 aggregate 2637 vs V15 2645; V15 vs V16 head-to-head at 3p
  is 0.510 (V15 slight edge). Plateau confirmed: V8/V10/V15/V16 all sit
  within ~10 ELO at 96 games per matchup, which is the sample noise floor.
* **V17 -- 1-ply opponent-aware buy + take-drain + smart self-reserve**:
  V15 dispatch with each affordable buy penalized by max-PV any opponent
  could afford post-buy. Per-pc impact was non-uniform: helped at 3p
  (+19 ELO) but hurt at 2p (-22) and 4p (-25), giving a -9 aggregate.
  Investigation revealed at 2p single-opponent threats invalidate the
  "absorb-exposure" heuristic, and at 4p the max-over-3-opponents threat
  estimate is inflated.
* **V18 -- V17's lookahead at 3p only**: Quarantine V17's logic to 3p
  where it helped, leave V15 logic at 2p/4p. Aggregate +7 ELO over V15
  in the V18 tournament; head-to-head with V15 a tie. Real signal at
  3p (+13 ELO over V15).
* **V19 -- 2-ply minimax buy at 3p**: Replace V18's coarse
  "max-PV opponent could afford" threat scan with a true 2-ply
  simulation: apply our buy, simulate each opponent's best one-ply buy
  response (V8-style highest-PV affordable), score the candidate by
  ``our_PV_gained - opp_PV_gained``.
* **V20 -- 2-ply minimax extended to 2p**: V19's logic also applied at
  2 players (still V10 at 4p where the multi-opponent chain dilutes
  signal).
* **V15 ships as production (V17-V20 superseded)**: A 192-game
  V15-vs-V20 head-to-head settled at V20=49.3% / V15=48.4% / ties=2.3%
  -- statistically identical (95% CI on a 50/50 binomial at n=576 is
  +/- 4pp). The aggregate-rating differences seen in 96-game
  tournaments (V18 +7 ELO, V19 +13 ELO, V20 +9 ELO) were sample noise
  on the rating fit, not real strength differences. V15 is at the
  handcrafted heuristic ceiling; the lookahead variants add ~25%
  per-turn compute cost for no measurable strength gain. Further
  improvement requires learned evaluators (trained-net + MCTS) rather
  than more elaborate heuristics. The V17-V20 candidates are kept in
  the registry for reproducibility but not wired into production.

## Key strategic insights

1. **Target selection is the single biggest lever**. Scoring cards by
   `points * 100 + PPT * 50 - distance * penalty` outperforms naive
   "buy whatever I can afford".
2. **Bridge buys compound**. A 0-PV affordable card whose bonus shortens
   the chosen target's `turns_to_afford` is strictly better than another
   take-3 round when the strict-improvement check passes.
3. **Denial threats scale with player count**. Reserving a high-PV card
   away from one of three opponents wastes tempo against the other two;
   the threshold needs `~4 * (num_players - 1)`.
4. **Selective denial matters**. If the threatened opponent has multiple
   alternative buys, denying one just reroutes them.
5. **Path planning helps tighter races**. The 2-card race-to-15 planner
   wins 2p/3p but loses 4p where chaos invalidates plans.
6. **Endgame triggers at 11 PV**. Earlier triggers risk swapping out of
   strong PPT play; later triggers miss winning races.

## Files

* `agent/eval/heuristic_opus.py` -- candidate definitions (V1-V15)
* `agent/eval/tournament.py` -- round-robin harness, anchored ratings,
  game-log analyzer
* `agent/eval/match_cache.py` -- SHA-256-keyed persistent matchup cache
* `agent/scripts/heuristic_opus_tournament.py` -- CLI driver
* `runs_heuristic_opus/` -- tournament JSON reports

## Reproducing

```
bazel run --config=mlinfra_v7 \
    //experimental/mloeppky/splendor/agent/scripts:heuristic_opus_tournament -- \
    --candidates v3,v8,v9,v10,v13,v14,v15 \
    --num-games 48 --player-counts 2,3,4 --max-turns 300 \
    --workers 4 --no-include-net \
    --cache-dir experimental/mloeppky/splendor/agent/runs_heuristic_opus/cache_final \
    --seed 7777 \
    --output runs_heuristic_opus/tournament_v15.json
```
