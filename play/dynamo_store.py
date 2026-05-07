"""DynamoDB-backed implementation of the play store interface."""

from __future__ import annotations

import json
import time
from typing import Any, Iterable

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from play.store import GameStatus


class DynamoPlayStore:
    """DynamoDB-backed store replacing JsonPlayStore for cloud deployment.

    Implements the same interface as JsonPlayStore so that PlayService
    works identically with either backend.
    """

    def __init__(
        self,
        games_table_name: str,
        users_table_name: str,
        region: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if region is not None:
            kwargs["region_name"] = region
        self._dynamodb = boto3.resource("dynamodb", **kwargs)
        self._games_table = self._dynamodb.Table(games_table_name)
        self._users_table = self._dynamodb.Table(users_table_name)

    # ── Game record CRUD ──────────────────────────────────────────────

    def load_game(self, game_id: str) -> dict[str, Any] | None:
        """Load a game record by game_id. Returns None if not found."""
        resp = self._games_table.get_item(Key={"game_id": game_id})
        item = resp.get("Item")
        if item is None:
            return None
        data = item["data"]
        return data if isinstance(data, dict) else json.loads(data)

    def save_game(self, record: dict[str, Any]) -> None:
        """Save a game record using unconditional PutItem (last-writer-wins)."""
        game_id = str(record["game_id"])
        updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record = {**record, "updated_at": updated_at}
        self._games_table.put_item(
            Item={
                "game_id": game_id,
                "user_sub": record.get("user_sub", ""),
                "updated_at": updated_at,
                "status": record.get("status", ""),
                "data": json.dumps(record),
            }
        )

    def list_games_for_user(
        self,
        username: str,
        status: Iterable[GameStatus] | None = None,
    ) -> list[dict[str, Any]]:
        """Query games for a user via the GSI, optionally filtering by status."""
        resp = self._games_table.query(
            IndexName="user_sub-updated_at-index",
            KeyConditionExpression=Key("user_sub").eq(username),
            ScanIndexForward=False,  # newest first
        )
        items = resp.get("Items", [])
        want = set(status) if status is not None else None
        results: list[dict[str, Any]] = []
        for item in items:
            if want is not None and item.get("status") not in want:
                continue
            data = item["data"]
            results.append(data if isinstance(data, dict) else json.loads(data))
        return results

    def put_game_if_not_exists(self, record: dict[str, Any]) -> bool:
        """Conditionally write a game record only if game_id does not already exist.

        Used by the sync CLI for deduplication — avoids overwriting existing
        cloud records when uploading local games.

        Returns True if the record was written, False if it already existed.
        """
        game_id = str(record["game_id"])
        updated_at = record.get(
            "updated_at",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        try:
            self._games_table.put_item(
                Item={
                    "game_id": game_id,
                    "user_sub": record.get("user_sub", ""),
                    "updated_at": updated_at,
                    "status": record.get("status", ""),
                    "data": json.dumps(record),
                },
                ConditionExpression="attribute_not_exists(game_id)",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    # ── User rating blob CRUD ─────────────────────────────────────────

    def load_user_rating_blob(self, username: str) -> dict[str, Any] | None:
        """Load a user's rating blob by username. Returns None if not found."""
        resp = self._users_table.get_item(Key={"username": username})
        item = resp.get("Item")
        if item is None:
            return None
        data = item["data"]
        # Handle both JSON string (normal) and raw dict (legacy/DynamoDB map)
        if isinstance(data, dict):
            return data
        return json.loads(data)

    def save_user_rating_blob(self, username: str, data: dict[str, Any]) -> None:
        """Save a user's rating blob using unconditional PutItem."""
        self._users_table.put_item(
            Item={
                "username": username,
                "data": json.dumps(data),
            }
        )

    def list_all_user_rating_blobs(self) -> list[dict[str, Any]]:
        """Scan the users table and return all rating blobs (deserialized)."""
        results: list[dict[str, Any]] = []
        resp = self._users_table.scan()
        for item in resp.get("Items", []):
            data = item["data"]
            results.append(data if isinstance(data, dict) else json.loads(data))
        # Handle pagination for large tables
        while "LastEvaluatedKey" in resp:
            resp = self._users_table.scan(
                ExclusiveStartKey=resp["LastEvaluatedKey"]
            )
            for item in resp.get("Items", []):
                data = item["data"]
                results.append(data if isinstance(data, dict) else json.loads(data))
        return results

    # ── Delete operations ─────────────────────────────────────────────

    def delete_game(self, game_id: str) -> bool:
        """Delete a game record. Returns True if deleted, False if not found."""
        try:
            self._games_table.delete_item(
                Key={"game_id": game_id},
                ConditionExpression="attribute_exists(game_id)",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def delete_user_data(self, username: str) -> int:
        """Delete user rating blob and all their games. Returns count of deleted games."""
        # Delete the user's rating blob
        self._users_table.delete_item(Key={"username": username})

        # Find and delete all games belonging to this user
        resp = self._games_table.query(
            IndexName="user_sub-updated_at-index",
            KeyConditionExpression=Key("user_sub").eq(username),
        )
        items = resp.get("Items", [])
        # Handle pagination
        while "LastEvaluatedKey" in resp:
            resp = self._games_table.query(
                IndexName="user_sub-updated_at-index",
                KeyConditionExpression=Key("user_sub").eq(username),
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            items.extend(resp.get("Items", []))

        deleted_count = 0
        for item in items:
            game_id = item["game_id"]
            self._games_table.delete_item(Key={"game_id": game_id})
            deleted_count += 1

        return deleted_count
