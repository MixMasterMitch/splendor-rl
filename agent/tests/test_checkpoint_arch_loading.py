from __future__ import annotations

import pathlib

import pytest
import torch

from agent.env import batched_engine as BE
from agent.net import model as M
from agent.train import checkpointing as CK
from agent.train.health import decide_next_action
from agent.train.loop import _is_new_best_eval, _latest_ckpt
from agent.train.league import League
from agent.train import ranking as R
from agent.train.replay_buffer import ReplayBuffer
from replay import players as P


def test_load_net_from_raw_flat_state_dict_infers_arch_and_hidden(
    tmp_path: pathlib.Path,
) -> None:
    torch.manual_seed(7)
    net = M.SplendorNet(hidden=48, arch="flat")
    path = tmp_path / "flat_raw.pt"
    torch.save(net.state_dict(), path)

    loaded_net, payload = CK.load_net_from_checkpoint(path, map_location="cpu")
    spec = CK.checkpoint_net_spec(payload)

    assert spec == CK.NetSpec(hidden=48, arch="flat")
    assert loaded_net.hidden == 48
    assert loaded_net.arch == "flat"
    for key, value in net.state_dict().items():
        assert torch.equal(loaded_net.state_dict()[key], value), key


@pytest.mark.parametrize(
    ("sims", "expected_type"),
    [
        (0, P.GreedyNetPolicy),
        (2, P.NetPolicy),
    ],
)
def test_parse_player_spec_loads_flat_checkpoint_for_net_players(
    tmp_path: pathlib.Path,
    sims: int,
    expected_type: type[P.GreedyNetPolicy] | type[P.NetPolicy],
) -> None:
    torch.manual_seed(11)
    net = M.SplendorNet(hidden=32, arch="flat")
    ckpt_path = tmp_path / "flat_ckpt.pt"
    CK.save_checkpoint(
        ckpt_path,
        net,
        optim=None,
        iteration=3,
        config={"num_players": 2},
        buffer=None,
    )

    policy = P.parse_player_spec(f"net:{ckpt_path}:sims={sims}", device="cpu")
    assert isinstance(policy, expected_type)
    assert policy._net.arch == "flat"
    assert policy._net.hidden == 32

    engine = BE.BatchedEngine(batch_size=1, num_players=2, device="cpu", seed=5)
    action = policy.choose(engine)
    assert action.shape == (1,)
    assert engine.legal_action_mask()[0, int(action.item())].item()


def test_league_checkpoint_persists_flat_arch_metadata(
    tmp_path: pathlib.Path,
) -> None:
    torch.manual_seed(13)
    net = M.SplendorNet(hidden=40, arch="flat")
    league = League(tmp_path / "league")

    ckpt_path = league.add_checkpoint(net, tag="flat")
    entry = league.list_entries()[0]
    loaded_net, payload = CK.load_net_from_checkpoint(ckpt_path, map_location="cpu")
    spec = CK.checkpoint_net_spec(payload)

    assert entry["arch"] == "flat"
    assert entry["hidden"] == 40
    assert spec == CK.NetSpec(hidden=40, arch="flat")
    assert loaded_net.arch == "flat"
    assert loaded_net.hidden == 40


def test_latest_resume_checkpoint_is_preferred_and_archive_is_lightweight(
    tmp_path: pathlib.Path,
) -> None:
    torch.manual_seed(17)
    net = M.SplendorNet(hidden=32, arch="flat")
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=4, device="cpu")

    archive_path = tmp_path / "iter_000002.pt"
    resume_path = tmp_path / "latest_resume.pt"
    CK.save_checkpoint(
        archive_path,
        net,
        optim=optim,
        iteration=2,
        config={"num_players": 2},
        buffer=None,
    )
    CK.save_checkpoint(
        resume_path,
        net,
        optim=optim,
        iteration=2,
        config={"num_players": 2},
        buffer=buffer,
    )

    archive_payload = CK.load_checkpoint_payload(archive_path, map_location="cpu")
    resume_payload = CK.load_checkpoint_payload(resume_path, map_location="cpu")

    assert "buffer" not in archive_payload
    assert "buffer" in resume_payload
    assert _latest_ckpt(tmp_path) == resume_path


def test_is_new_best_eval_prefers_heuristic_score_then_tiebreakers() -> None:
    history = [
        {
            "iter": 10,
            "winrate_vs_random": 0.95,
            "ties_vs_random": 0.0,
            "winrate_vs_heuristic": 0.50,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_turns_vs_heuristic": 150.0,
        }
    ]
    assert _is_new_best_eval(
        history,
        {
            "iter": 12,
            "winrate_vs_random": 0.90,
            "ties_vs_random": 0.0,
            "winrate_vs_heuristic": 0.515625,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_turns_vs_heuristic": 200.0,
        },
    )
    assert _is_new_best_eval(
        history,
        {
            "iter": 14,
            "winrate_vs_random": 0.96,
            "ties_vs_random": 0.0,
            "winrate_vs_heuristic": 0.50,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_turns_vs_heuristic": 140.0,
        },
    )
    assert not _is_new_best_eval(
        history,
        {
            "iter": 16,
            "winrate_vs_random": 0.94,
            "ties_vs_random": 0.0,
            "winrate_vs_heuristic": 0.50,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_turns_vs_heuristic": 160.0,
        },
    )


def test_is_new_best_eval_prefers_finished_step_metrics_when_present() -> None:
    history = [
        {
            "iter": 20,
            "winrate_vs_random": 0.95,
            "ties_vs_random": 0.0,
            "winrate_vs_heuristic": 0.55,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_finished_step_vs_heuristic": 118.0,
            "max_finished_step_vs_heuristic": 150.0,
        }
    ]
    assert _is_new_best_eval(
        history,
        {
            "iter": 22,
            "winrate_vs_random": 0.95,
            "ties_vs_random": 0.0,
            "winrate_vs_heuristic": 0.55,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_finished_step_vs_heuristic": 117.0,
            "max_finished_step_vs_heuristic": 149.0,
        },
    )


def test_is_new_best_eval_prefers_rank_eval_metrics_when_present() -> None:
    history = [
        {
            "iter": 30,
            "winrate_vs_heuristic": 0.55,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_finished_step_vs_heuristic": 120.0,
            "rank_winrate_vs_heuristic": 0.50,
            "rank_ties_vs_heuristic": 0.0,
            "rank_finished_vs_heuristic": 1.0,
            "rank_avg_finished_step_vs_heuristic": 115.0,
        }
    ]
    assert _is_new_best_eval(
        history,
        {
            "iter": 32,
            "winrate_vs_heuristic": 0.54,
            "ties_vs_heuristic": 0.0,
            "finished_vs_heuristic": 1.0,
            "avg_finished_step_vs_heuristic": 130.0,
            "rank_winrate_vs_heuristic": 0.56,
            "rank_ties_vs_heuristic": 0.0,
            "rank_finished_vs_heuristic": 1.0,
            "rank_avg_finished_step_vs_heuristic": 114.0,
        },
    )


def test_decide_next_action_reduces_lr_for_strong_flat_window() -> None:
    history = [
        {
            "iter": it,
            "lr": 3e-4,
            "loss": 1.35 + 0.01 * (it % 2),
            "winrate_vs_random": 0.95,
            "winrate_vs_heuristic": wr,
        }
        for it, wr in zip(
            [2, 4, 6, 8, 10, 12, 14, 16],
            [0.58, 0.60, 0.57, 0.59, 0.58, 0.57, 0.56, 0.55],
            strict=True,
        )
    ]
    assert (
        decide_next_action(
            history,
            late_stage_cooldown_rows=7,
            late_stage_regress_epsilon=-0.03,
        )
        == "reduce_lr"
    )


def test_league_prune_keeps_recent_and_strong_entries(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(19)
    league = League(tmp_path / "league", max_entries=3, keep_recent=1)
    for tag, score in [("a", 0.90), ("b", 0.10), ("c", 0.20), ("d", 0.05)]:
        league.add_checkpoint(
            M.SplendorNet(hidden=32, arch="flat"),
            tag=tag,
            metadata={"score_hint": score},
        )
    entries = league.list_entries()
    kept = {entry["tag"] for entry in entries}
    assert len(entries) == 3
    assert "a" in kept
    assert "d" in kept


def test_fit_anchored_ratings_is_order_invariant() -> None:
    results_a: list[dict] = []
    R.add_match_result(results_a, "ckpt:a", "random", 63.0, 1.0, 0.0)
    R.add_match_result(results_a, "ckpt:b", "random", 52.0, 12.0, 0.0)
    R.add_match_result(results_a, "ckpt:a", "heuristic", 40.0, 24.0, 0.0)
    R.add_match_result(results_a, "heuristic", "ckpt:b", 42.0, 18.0, 4.0)
    R.add_match_result(results_a, "ckpt:a", "ckpt:b", 46.0, 14.0, 4.0)

    results_b: list[dict] = []
    R.add_match_result(results_b, "ckpt:b", "ckpt:a", 14.0, 46.0, 4.0)
    R.add_match_result(results_b, "ckpt:b", "heuristic", 18.0, 42.0, 4.0)
    R.add_match_result(results_b, "random", "ckpt:b", 12.0, 52.0, 0.0)
    R.add_match_result(results_b, "heuristic", "ckpt:a", 24.0, 40.0, 0.0)
    R.add_match_result(results_b, "random", "ckpt:a", 1.0, 63.0, 0.0)

    ratings_a = R.fit_anchored_ratings(results_a)
    ratings_b = R.fit_anchored_ratings(results_b)

    assert ratings_a["random"] == pytest.approx(1000.0)
    assert ratings_a["heuristic"] == pytest.approx(2500.0)
    assert ratings_a["ckpt:a"] > ratings_a["heuristic"]
    assert ratings_a["ckpt:b"] > ratings_a["random"]
    assert ratings_a["ckpt:a"] > ratings_a["ckpt:b"]
    assert ratings_b["ckpt:a"] == pytest.approx(ratings_a["ckpt:a"], abs=1e-6)
    assert ratings_b["ckpt:b"] == pytest.approx(ratings_a["ckpt:b"], abs=1e-6)


def test_league_recompute_ratings_updates_entry_fields(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(23)
    league = League(tmp_path / "league")
    league.add_checkpoint(M.SplendorNet(hidden=32, arch="flat"), tag="a")
    league.add_checkpoint(M.SplendorNet(hidden=32, arch="flat"), tag="b")

    row_a = {
        "rank_winrate_vs_random": 63.0 / 64.0,
        "rank_ties_vs_random": 0.0,
        "rank_winrate_vs_heuristic": 40.0 / 64.0,
        "rank_ties_vs_heuristic": 0.0,
    }
    row_b = {
        "rank_winrate_vs_random": 52.0 / 64.0,
        "rank_ties_vs_random": 0.0,
        "rank_winrate_vs_heuristic": 18.0 / 64.0,
        "rank_ties_vs_heuristic": 4.0 / 64.0,
    }
    league.record_checkpoint_baselines(0, row_a, rank_games=64, eval_games=16)
    league.record_checkpoint_baselines(1, row_b, rank_games=64, eval_games=16)
    league.record_result("ckpt:0", "ckpt:1", 46.0, 14.0, 4.0)

    ratings = league.recompute_ratings()
    entries = {int(entry["idx"]): entry for entry in league.list_entries()}

    assert ratings["random"] == pytest.approx(1000.0)
    assert ratings["heuristic"] == pytest.approx(2500.0)
    assert float(entries[0]["rating"]) > float(entries[1]["rating"])
    assert int(entries[0]["games"]) == 192
    assert int(entries[1]["games"]) == 192


def test_league_migrates_legacy_entries_into_results(tmp_path: pathlib.Path) -> None:
    league_root = tmp_path / "league"
    league_root.mkdir(parents=True, exist_ok=True)
    legacy_path = league_root / "ckpt_00000_i5.pt"
    torch.save({}, legacy_path)
    manifest_path = league_root / "league.json"
    manifest_path.write_text(
        """
{
  "entries": [
    {
      "idx": 0,
      "tag": "i5",
      "path": "%s",
      "elo": 0.0,
      "games": 0,
      "hidden": 192,
      "arch": "attn",
      "rank_winrate_vs_random": 0.95703125,
      "rank_ties_vs_random": 0.0,
      "rank_winrate_vs_heuristic": 0.544921875,
      "rank_ties_vs_heuristic": 0.00390625
    }
  ]
}
""".strip()
        % str(legacy_path)
    )

    league = League(league_root)
    entry = league.list_entries()[0]

    assert league.manifest["anchors"]["random"] == pytest.approx(1000.0)
    assert league.manifest["anchors"]["heuristic"] == pytest.approx(2500.0)
    assert len(league.manifest["results"]) == 2
    assert float(entry["rating"]) > 2500.0
    assert int(entry["games"]) == 1024
