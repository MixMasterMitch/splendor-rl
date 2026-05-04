# Splendor play webapp

React + Vite UI for human-vs-bot Splendor backed by `play_server`.

## Run the API

From the repo root:

```bash
python -m play.server --port 8765
```

Game JSON and rating files persist under `play/play_data/` (override with `--store-root`).

## Run the UI

```bash
cd replay_webapp
pnpm install    # once
pnpm dev
```

Vite proxies `/api/play/*` to the play server (see `PLAY_SERVER`, default `http://127.0.0.1:8765`).

The browser stores a username in `localStorage` and sends it on each request as `X-Splendor-Username`. There is no sign-in.

```bash
PLAY_SERVER=http://127.0.0.1:8765 pnpm dev
```

## Endpoints proxied via `/api/play`

| Method | Route | Notes |
| --- | --- | --- |
| GET | `/models` | public |
| GET | `/health` | public |
| GET | `/me` | requires `X-Splendor-Username` |
| GET | `/leaderboard` | requires `X-Splendor-Username` |
| GET | `/games?status=all` etc | scoped to username |
| POST | `/games` | start a new game when none is in-flight |
| GET | `/games/<id>` | resume |
| POST | `/games/<id>/action` | `{action:int}` |

Only one unfinished game may exist per username at a time. Games must be played to completion.

3+ player setups exclude checkpoint `kind=net` entries (UI and API).
