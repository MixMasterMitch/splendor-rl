"""Local HTTP server for interactive Splendor.

Thin wrapper around the Lambda handler logic so local dev and production
behave identically. Uses the prod DynamoDB tables directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from play.lambda_handler import _dispatch, _json_response


def _workspace_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parent.parent


def _get_table_names_from_stack(stack_name: str, region: str) -> tuple[str, str]:
    """Resolve DynamoDB table names from the CloudFormation stack."""
    try:
        result = subprocess.run(
            [
                "aws", "cloudformation", "describe-stack-resources",
                "--stack-name", stack_name,
                "--region", region,
                "--output", "json",
            ],
            capture_output=True, text=True, check=True,
        )
        resources = json.loads(result.stdout).get("StackResources", [])
        games_table = ""
        users_table = ""
        for r in resources:
            logical_id = r.get("LogicalResourceId", "")
            if logical_id.startswith("GamesTable"):
                games_table = r["PhysicalResourceId"]
            elif logical_id.startswith("UsersTable"):
                users_table = r["PhysicalResourceId"]
        if not games_table or not users_table:
            print(f"ERROR: Could not find table names in stack {stack_name}", file=sys.stderr)
            sys.exit(1)
        return games_table, users_table
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: Failed to query CloudFormation: {e}", file=sys.stderr)
        sys.exit(1)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write(f"[play_server] {self.address_string()} - {format % args}\n")

    def _send_lambda_response(self, resp: dict[str, Any]) -> None:
        status = resp.get("statusCode", 200)
        body = resp.get("body", "")
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        for k, v in resp.get("headers", {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Splendor-Username",
        )
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return ""
        return self.rfile.read(length).decode("utf-8")

    def _header_map(self) -> dict[str, str]:
        return {k.lower(): v for k, v in self.headers.items()}

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        # Parse query params
        from urllib.parse import parse_qs
        qp_raw = parse_qs(parsed.query)
        query_params = {k: v[0] for k, v in qp_raw.items() if v}

        headers = self._header_map()
        body = self._read_body() if method in ("POST", "PUT", "DELETE") else ""

        resp = _dispatch(method, path, headers, query_params, body)
        self._send_lambda_response(resp)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_lambda_response(_json_response(200, {"ok": True}))

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle("DELETE")


def serve(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"play_server listening on http://{host}:{port}")
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Splendor play server (local dev)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--stack-name", default="SplendorStack")
    parser.add_argument(
        "--games-table", default=None,
        help="DynamoDB games table name (auto-resolved from stack if omitted)",
    )
    parser.add_argument(
        "--users-table", default=None,
        help="DynamoDB users table name (auto-resolved from stack if omitted)",
    )
    args = parser.parse_args(argv)

    # Resolve table names
    if args.games_table and args.users_table:
        games_table = args.games_table
        users_table = args.users_table
    else:
        print(f"Resolving table names from stack '{args.stack_name}'...")
        games_table, users_table = _get_table_names_from_stack(args.stack_name, args.region)

    # Set env vars that lambda_handler reads
    os.environ.setdefault("GAMES_TABLE", games_table)
    os.environ.setdefault("USERS_TABLE", users_table)
    os.environ.setdefault("AWS_REGION", args.region)

    print(f"  GAMES_TABLE: {games_table}")
    print(f"  USERS_TABLE: {users_table}")
    print(f"  AWS_REGION:  {args.region}")

    serve(args.host, args.port)


if __name__ == "__main__":
    main()
