from __future__ import annotations

import pytest
import torch

from agent.eval import ladder as LAD
from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.net import encoder as ENC
from agent.net import model as M
from agent.search import gumbel_mcts as G
from agent.train import active_batching as AB
from agent.train import league_selfplay as LS
from agent.train.league import League
from agent.train.replay_buffer import ReplayBuffer


def test_engine_batch_views_preserve_state_rows() -> None:
    engine = BE.BatchedEngine(batch_size=3, num_players=2, device="cpu", seed=11)

    idx = torch.tensor([2, 0], dtype=torch.long)
    sub = engine.index_select(idx)
    assert sub.batch_size == 2
    assert torch.equal(sub.gem_pool, engine.gem_pool.index_select(0, idx))
    assert torch.equal(sub.grid_card, engine.grid_card.index_select(0, idx))
    assert torch.equal(sub.current_player, engine.current_player.index_select(0, idx))

    expanded = engine.repeat_interleave(2)
    expected_idx = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    assert expanded.batch_size == 6
    assert torch.equal(expanded.tokens, engine.tokens.index_select(0, expected_idx))
    assert torch.equal(
        expanded.current_player,
        engine.current_player.index_select(0, expected_idx),
    )


def test_batched_root_child_eval_matches_sequential() -> None:
    torch.manual_seed(0)
    engine = BE.BatchedEngine(batch_size=5, num_players=2, device="cpu", seed=7)
    net = M.SplendorNet(hidden=64, arch="flat").eval()

    gem_pool_before = engine.gem_pool.clone()
    grid_before = engine.grid_card.clone()
    current_before = engine.current_player.clone()

    logits, _, legal = net.inference(engine)
    masked_score = logits.masked_fill(~legal, -1e9)
    k = min(4, int(legal.sum(dim=-1).max().item()))
    _, topk_idx = torch.topk(masked_score, k=k, dim=-1)

    sequential = G._evaluate_root_children_sequential(engine, net, topk_idx, legal)
    batched = G._evaluate_root_children_batched(engine, net, topk_idx, legal)

    assert torch.allclose(sequential, batched, atol=1e-6, rtol=0.0)
    assert torch.equal(engine.gem_pool, gem_pool_before)
    assert torch.equal(engine.grid_card, grid_before)
    assert torch.equal(engine.current_player, current_before)


def test_replay_buffer_cpu_uses_float32_storage_and_migrates_legacy_state() -> None:
    buffer = ReplayBuffer(capacity=8, device="cpu")
    assert buffer.storage_dtype == torch.float32

    global_feat = torch.randn(4, ENC.D_GLOBAL)
    card_feat = torch.randn(4, ENC.N_CARDS, ENC.D_CARD)
    legal_mask = torch.zeros((4, A.NUM_ACTIONS), dtype=torch.bool)
    legal_mask[:, 0] = True
    policy = torch.rand(4, A.NUM_ACTIONS)
    value = torch.randn(4, BE.MAX_PLAYERS)
    buffer.add_batch(global_feat, card_feat, legal_mask, policy, value)

    batch = buffer.sample(2, generator=torch.Generator(device="cpu").manual_seed(1))
    assert batch["global_feat"].dtype == torch.float32
    assert batch["card_feat"].dtype == torch.float32
    assert batch["policy"].dtype == torch.float32
    assert batch["value"].dtype == torch.float32

    legacy = ReplayBuffer(capacity=8, device="cpu", storage_dtype=torch.float16)
    legacy.add_batch(global_feat, card_feat, legal_mask, policy, value)
    migrated = ReplayBuffer(capacity=8, device="cpu")
    migrated.load_state_dict(legacy.state_dict())

    assert migrated.global_feat.dtype == torch.float32
    assert migrated.card_feat.dtype == torch.float32
    assert migrated.policy.dtype == torch.float32
    assert migrated.value.dtype == torch.float32


def test_take3_zero_pile_edge_case_matches_single_engine_semantics() -> None:
    engine = BE.BatchedEngine(batch_size=1, num_players=2, device="cpu", seed=5)
    engine.phase.zero_()
    engine.ended.zero_()
    engine.gem_pool[0, :5] = 0
    mask = engine.legal_action_mask()[0]
    assert mask[A.TAKE3_BASE : A.TAKE3_BASE + A.TAKE3_COUNT].all()


def test_active_batching_splits_into_power_of_two_buckets() -> None:
    idx = torch.arange(300, dtype=torch.long)
    groups = AB.bucket_indices(idx, max_bucket=512)
    assert [int(group.numel()) for group in groups] == [256, 32, 8, 4]
    rebuilt = torch.cat(groups, dim=0)
    assert torch.equal(rebuilt, idx)


def test_league_load_cached_net_reuses_instance(tmp_path) -> None:
    torch.manual_seed(3)
    league = League(tmp_path / "league")
    ckpt = league.add_checkpoint(M.SplendorNet(hidden=32, arch="flat"), tag="flat")

    first = league.load_cached_net(ckpt, device="cpu")
    second = league.load_cached_net(str(ckpt), device=torch.device("cpu"))

    assert first is second
    assert first.arch == "flat"
    assert first.hidden == 32


def test_league_turn_subbatches_opponent_searches(monkeypatch, tmp_path) -> None:
    torch.manual_seed(5)
    main_net = M.SplendorNet(hidden=32, arch="flat").eval()
    main_net._compiled = True

    league = League(tmp_path / "league")
    path_a = str(league.add_checkpoint(M.SplendorNet(hidden=32, arch="flat"), tag="a"))
    path_b = str(league.add_checkpoint(M.SplendorNet(hidden=32, arch="flat"), tag="b"))

    engine = BE.BatchedEngine(batch_size=4, num_players=2, device="cpu", seed=11)
    engine.current_player[:] = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    g, c, _, legal = ENC.encode_state_with_legal(engine)

    seat_path_idx = torch.full((4, BE.MAX_PLAYERS), -1, dtype=torch.long)
    seat_path_idx[0, 0] = -1
    seat_path_idx[1, 0] = 0
    seat_path_idx[2, 0] = -1
    seat_path_idx[3, 1] = 1
    seat_path_idx[0, 1] = 0
    seat_path_idx[1, 1] = -1
    seat_path_idx[2, 1] = 1
    seat_path_idx[3, 0] = -1

    calls: list[int] = []

    def fake_gumbel_root_act(
        engine_arg: BE.BatchedEngine,
        net_arg: M.SplendorNet,
        num_sims: int = 8,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del net_arg, num_sims
        calls.append(engine_arg.batch_size)
        actions = torch.full(
            (engine_arg.batch_size,),
            A.PASS_ACTION,
            dtype=torch.int64,
            device=engine_arg.device,
        )
        improved = torch.zeros(
            (engine_arg.batch_size, A.NUM_ACTIONS),
            dtype=torch.float32,
            device=engine_arg.device,
        )
        improved[:, A.PASS_ACTION] = 1.0
        return actions, improved

    monkeypatch.setattr(LS.G, "gumbel_root_act", fake_gumbel_root_act)
    actions, main_idx, main_improved = LS._choose_seat_actions(
        engine,
        main_net,
        league,
        seat_path_idx,
        [path_a, path_b],
        num_sims=2,
        precomputed=(g, c, legal),
    )

    assert calls == [2, 1, 1]
    assert torch.equal(main_idx, torch.tensor([0, 2], dtype=torch.long))
    assert main_improved is not None
    assert main_improved.shape == (2, A.NUM_ACTIONS)
    assert actions.shape == (4,)


def test_league_selfplay_keeps_unfinished_main_samples_with_stall_penalty(
    tmp_path,
) -> None:
    torch.manual_seed(23)
    main_net = M.SplendorNet(hidden=32, arch="flat").eval()
    league = League(tmp_path / "league")
    buffer = ReplayBuffer(capacity=128, device="cpu")

    metrics = LS.run_league_selfplay(
        main_net,
        buffer,
        league,
        num_players=2,
        num_games=4,
        device="cpu",
        max_turns=1,
        num_sims=1,
        seed=3,
        league_prob=0.0,
        time_discount=0.99,
    )

    assert metrics["finished"] == 0
    assert metrics["samples_added"] > 0
    assert metrics["samples_from_finished"] == 0
    assert len(buffer) == metrics["samples_added"]
    stored = buffer.value[: len(buffer)]
    assert torch.all(stored <= 0.0)
    assert torch.all(stored.sum(dim=1) == -2.0)


def test_evaluate_reports_finished_step_metrics(monkeypatch) -> None:
    net = M.SplendorNet(hidden=32, arch="flat").eval()

    def fake_play_match(*args, **kwargs) -> dict[str, float]:
        del args, kwargs
        return {
            "games_finished": 1.0,
            "games_capped": 1.0,
            "games_total": 2.0,
            "agent_wins": 1.0,
            "agent_ties": 0.0,
            "turns_sum": 5.0,
            "finished_turns_sum": 2.0,
            "max_finished_step": 2.0,
            "agent_overlimit_count": 2.0,
            "agent_main_actions": 10.0,
            "agent_overlimit_rate": 0.2,
        }

    monkeypatch.setattr(LAD, "_play_match", fake_play_match)
    result = LAD.evaluate(
        net,
        num_players=2,
        num_games=4,
        opponents={
            "heuristic": lambda: (
                lambda engine: torch.zeros(
                    engine.batch_size, dtype=torch.int64, device=engine.device
                )
            )
        },
        device="cpu",
        num_sims=1,
        max_turns=4,
    )

    assert result["finished_vs_heuristic"] == pytest.approx(0.5)
    assert result["capped_vs_heuristic"] == pytest.approx(0.5)
    assert result["avg_turns_vs_heuristic"] == pytest.approx(2.5)
    assert result["avg_finished_step_vs_heuristic"] == pytest.approx(2.0)
    assert result["max_finished_step_vs_heuristic"] == pytest.approx(2.0)
