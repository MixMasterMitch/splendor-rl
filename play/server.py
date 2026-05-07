"""HTTP server for interactive Splendor (local dev adapter for ``PlayService``)."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from play import auth as AU
from play.service import PlayService
from play.store import JsonPlayStore


def _workspace_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


class _Ctx:
    def __init__(self, svc: PlayService) -> None:
        self.service = svc


class _Handler(BaseHTTPRequestHandler):
    server_context: _Ctx

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write(
            f"[play_server] {self.address_string()} - {format % args}\n"
        )

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Splendor-Username",
        )
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"invalid JSON body: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("JSON body must be an object")
        return obj

    def _header_map(self) -> dict[str, str]:
        return {k: v for k, v in self.headers.items()}

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._dispatch_get()
        except PermissionError as e:
            self._send_json(403, {"error": str(e)})
        except KeyError as e:
            self._send_json(404, {"error": f"not found: {e}"})
        except ValueError as e:
            msg = str(e)
            status = 401 if "missing X-Splendor-Username" in msg else 400
            self._send_json(status, {"error": msg})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._dispatch_post()
        except PermissionError as e:
            self._send_json(403, {"error": str(e)})
        except KeyError as e:
            self._send_json(404, {"error": f"not found: {e}"})
        except ValueError as e:
            msg = str(e)
            status = 401 if "missing X-Splendor-Username" in msg else 400
            self._send_json(status, {"error": msg})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})

    def _dispatch_get(self) -> None:
        u = urlparse(self.path)
        p = u.path
        svc = self.server_context.service
        qp = parse_qs(u.query)
        qp1 = {k: v[0] for k, v in qp.items() if v}

        if p == "/api/health":
            self._send_json(200, {"ok": True})
            return
        if p == "/api/agents":
            self._send_json(200, svc.list_models())
            return
        identity = AU.identity_from_headers(self._header_map())
        if p == "/api/me":
            self._send_json(200, svc.me(identity))
            return
        if p == "/api/leaderboard":
            self._send_json(200, svc.leaderboard())
            return
        if p == "/api/games":
            st = qp1.get("status")
            summaries = svc.list_games_summary(identity, st)
            self._send_json(200, summaries)
            return
        parts = p.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "games":
            game_id = parts[2]
            view = svc.get_view(identity, game_id)
            self._send_json(200, view)
            return
        self._send_json(404, {"error": f"not found: {p}"})

    def _dispatch_post(self) -> None:
        u = urlparse(self.path)
        p = u.path
        svc = self.server_context.service
        identity = AU.identity_from_headers(self._header_map())
        body = self._read_json_body()
        if p == "/api/games":
            session = svc.create_game(identity, body)
            with session.lock:
                self._send_json(201, session.view())
            return
        parts = p.strip("/").split("/")
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "games":
            game_id = parts[2]
            verb = parts[3]
            if verb == "action":
                action = body.get("action")
                if not isinstance(action, int):
                    raise ValueError("body must include integer 'action'")
                session = svc.apply_human_action(identity, game_id, action)
                with session.lock:
                    self._send_json(200, session.view())
                return
            if verb == "step-ai":
                session = svc.step_ai(identity, game_id)
                with session.lock:
                    self._send_json(200, session.view())
                return
        self._send_json(404, {"error": f"not found: {p}"})


def serve(ctx: _Ctx, host: str, port: int) -> None:
    handler_cls = type("BoundHandler", (_Handler,), {"server_context": ctx})
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    print(f"play_server listening on http://{host}:{port}")
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Splendor play server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument(
        "--store-root",
        default=None,
        help="JSON store directory. Defaults to play/play_data",
    )
    args = parser.parse_args(argv)

    ws = _workspace_root()
    if args.store_root:
        sr = pathlib.Path(args.store_root)
        store_root = sr if sr.is_absolute() else ws / sr
    else:
        store_root = ws / "play/play_data"

    store = JsonPlayStore(store_root)
    svc = PlayService(workspace_root=ws, play_store=store, device=args.device)
    print(f"workspace: {ws}")
    print(f"play store: {store_root}")
    serve(_Ctx(svc), args.host, args.port)


if __name__ == "__main__":
    main()
