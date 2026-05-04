"""Tests for durable play store and setup validation rules."""

from __future__ import annotations

import os
import pathlib

import pytest

from play import auth as AU
from play.service import PlayService
from play.store import JsonPlayStore


def _workspace_root() -> pathlib.Path:
    ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    return pathlib.Path(ws) if ws else pathlib.Path.cwd()


@pytest.fixture()
def isolated_store(tmp_path: pathlib.Path) -> JsonPlayStore:
    return JsonPlayStore(tmp_path / "pdata")


@pytest.fixture()
def svc(isolated_store: JsonPlayStore, tmp_path: pathlib.Path) -> PlayService:
    return PlayService(workspace_root=tmp_path, play_store=isolated_store, device="cpu")


@pytest.fixture()
def svc_workspace(isolated_store: JsonPlayStore) -> PlayService:
    return PlayService(workspace_root=_workspace_root(), play_store=isolated_store, device="cpu")


def test_validate_rejects_net_in_three_player_game(svc_workspace: PlayService) -> None:
    nets = [m for m in svc_workspace.list_models() if m["kind"] == "net"]
    if not nets:
        pytest.skip("no scanned net checkpoints in this workspace checkout")
    net_id = str(nets[0]["id"])
    with pytest.raises(ValueError, match="net checkpoints"):
        svc_workspace.validate_and_build_opponents(
            num_players=3,
            human_seat=0,
            opponents={1: "random", 2: net_id},
            num_sims=8,
            policy_seed_basis=42,
        )


def test_fit_human_rating_respects_entity_id() -> None:
    from play import human_elo as HE

    anchors = {"opp": 2000.0}
    rows = [{"a": "human:alice", "b": "opp", "wins_a": 10.0, "wins_b": 0.0, "ties": 0.0}]
    r = HE.fit_human_rating(rows, anchors, human_entity="human:alice")
    assert r > anchors["opp"]


def test_reconstruct_session_matches_steps(svc: PlayService) -> None:
    id1 = AU.UserIdentity(username="u1")
    body = {"num_players": 2, "human_seat": 0, "opponents": {1: "random"}, "num_sims": 0}
    sess = svc.create_game(id1, body)
    gid = sess.game_id
    rec = svc.play_store.load_game(gid)
    assert rec is not None
    rebuilt = svc._session_from_record(rec)
    assert len(rebuilt.steps) == len(sess.steps)
    with rebuilt.lock:
        assert rebuilt.ended() == sess.ended()


def test_cannot_open_second_game_while_first_in_flight(svc: PlayService) -> None:
    id1 = AU.UserIdentity(username="u2")
    body = {"num_players": 2, "human_seat": 0, "opponents": {1: "random"}, "num_sims": 0}
    svc.create_game(id1, body)
    with pytest.raises(ValueError, match="in-flight"):
        svc.create_game(id1, body)
