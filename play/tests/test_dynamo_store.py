"""Tests for DynamoPlayStore conditional write (put_game_if_not_exists)."""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from play.dynamo_store import DynamoPlayStore


GAMES_TABLE = "test-games"
USERS_TABLE = "test-users"
REGION = "us-east-1"


@pytest.fixture()
def dynamo_store():
    """Create a DynamoPlayStore backed by moto-mocked DynamoDB."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name=REGION)
        # Create games table with GSI
        client.create_table(
            TableName=GAMES_TABLE,
            KeySchema=[{"AttributeName": "game_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "game_id", "AttributeType": "S"},
                {"AttributeName": "user_sub", "AttributeType": "S"},
                {"AttributeName": "updated_at", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "user_sub-updated_at-index",
                    "KeySchema": [
                        {"AttributeName": "user_sub", "KeyType": "HASH"},
                        {"AttributeName": "updated_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        # Create users table
        client.create_table(
            TableName=USERS_TABLE,
            KeySchema=[{"AttributeName": "username", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "username", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        store = DynamoPlayStore(GAMES_TABLE, USERS_TABLE, region=REGION)
        yield store


def _make_record(game_id: str = "abc123def456", user_sub: str = "TestUser") -> dict:
    return {
        "game_id": game_id,
        "user_sub": user_sub,
        "status": "completed",
        "updated_at": "2026-01-01T00:00:00Z",
        "steps": [{"action": "take"}],
        "num_players": 2,
    }


class TestPutGameIfNotExists:
    """Tests for put_game_if_not_exists conditional write."""

    def test_returns_true_when_game_does_not_exist(self, dynamo_store: DynamoPlayStore) -> None:
        record = _make_record()
        result = dynamo_store.put_game_if_not_exists(record)
        assert result is True

    def test_record_is_persisted_after_successful_write(self, dynamo_store: DynamoPlayStore) -> None:
        record = _make_record()
        dynamo_store.put_game_if_not_exists(record)
        loaded = dynamo_store.load_game(record["game_id"])
        assert loaded is not None
        assert loaded["game_id"] == record["game_id"]
        assert loaded["user_sub"] == record["user_sub"]
        assert loaded["steps"] == record["steps"]

    def test_returns_false_when_game_already_exists(self, dynamo_store: DynamoPlayStore) -> None:
        record = _make_record()
        # First write succeeds
        assert dynamo_store.put_game_if_not_exists(record) is True
        # Second write with same game_id returns False
        assert dynamo_store.put_game_if_not_exists(record) is False

    def test_does_not_overwrite_existing_record(self, dynamo_store: DynamoPlayStore) -> None:
        original = _make_record()
        original["steps"] = [{"action": "original"}]
        dynamo_store.put_game_if_not_exists(original)

        # Try to write a different record with the same game_id
        modified = _make_record()
        modified["steps"] = [{"action": "modified"}]
        dynamo_store.put_game_if_not_exists(modified)

        # Original should be preserved
        loaded = dynamo_store.load_game(original["game_id"])
        assert loaded["steps"] == [{"action": "original"}]

    def test_different_game_ids_both_succeed(self, dynamo_store: DynamoPlayStore) -> None:
        record_a = _make_record(game_id="aaaaaaaaaaaa")
        record_b = _make_record(game_id="bbbbbbbbbbbb")
        assert dynamo_store.put_game_if_not_exists(record_a) is True
        assert dynamo_store.put_game_if_not_exists(record_b) is True

    def test_preserves_updated_at_from_record(self, dynamo_store: DynamoPlayStore) -> None:
        record = _make_record()
        record["updated_at"] = "2025-06-15T12:30:00Z"
        dynamo_store.put_game_if_not_exists(record)
        loaded = dynamo_store.load_game(record["game_id"])
        assert loaded["updated_at"] == "2025-06-15T12:30:00Z"
