import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PlaySetup } from "./components/PlaySetup";
import { PlayGame } from "./components/PlayGame";
import type {
  GameListItem,
  LeaderboardResponse,
  ModelInfo,
  NewGameRequest,
  PlayView,
  UserMe,
} from "./play_types";

const PLAY_API = "/api/play";
const ACTIVE_GAME_KEY = "splendor.play.active_game_id";
const USERNAME_KEY = "splendor.play.username";

/** Mirrors server rule: ASCII letters, digits, underscore, hyphen; max length 32. */
const USERNAME_RE = /^[a-zA-Z0-9_-]{1,32}$/;

function isInFlightGame(g: GameListItem): boolean {
  return g.status === "human_turn" || g.status === "ai_thinking";
}

function gamesTableOrder(list: GameListItem[]): GameListItem[] {
  const inflight = list.filter(isInFlightGame);
  const past = list.filter((x) => !isInFlightGame(x));
  const byUpdated = (a: GameListItem, b: GameListItem) =>
    String(b.updated_at ?? "").localeCompare(String(a.updated_at ?? ""));
  return [...inflight.sort(byUpdated), ...past.sort(byUpdated)];
}

function formatGameStatus(status: string): string {
  if (status === "human_turn") return "Your turn";
  if (status === "ai_thinking") return "AI thinking";
  if (status === "completed") return "Completed";
  if (status === "aborted") return "Aborted";
  return status;
}

function readActiveGameId(): string | null {
  try {
    return window.localStorage.getItem(ACTIVE_GAME_KEY);
  } catch {
    return null;
  }
}

function writeActiveGameId(id: string | null): void {
  try {
    if (id) window.localStorage.setItem(ACTIVE_GAME_KEY, id);
    else window.localStorage.removeItem(ACTIVE_GAME_KEY);
  } catch {
    // Ignore storage failures.
  }
}

function readStoredUsername(): string | null {
  try {
    const raw = window.localStorage.getItem(USERNAME_KEY);
    const t = raw?.trim();
    if (!t || !USERNAME_RE.test(t)) return null;
    return t;
  } catch {
    return null;
  }
}

function writeStoredUsername(u: string): void {
  try {
    window.localStorage.setItem(USERNAME_KEY, u);
  } catch {
    // Ignore storage failures.
  }
}

function authHeaders(username: string): HeadersInit {
  return { "X-Splendor-Username": username };
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

async function getJsonAuth<T>(url: string, username: string): Promise<T> {
  const res = await fetch(url, { headers: authHeaders(username) });
  const txt = await res.text();
  if (!res.ok) {
    let msg: string = txt;
    try {
      const parsed = JSON.parse(txt) as { error?: string };
      if (parsed?.error) msg = parsed.error;
    } catch {
      // ignore
    }
    throw new Error(msg || `${res.status}`);
  }
  return JSON.parse(txt) as T;
}

async function postJsonAuth<T>(url: string, username: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(username) },
    body: JSON.stringify(body),
  });
  const txt = await res.text();
  if (!res.ok) {
    let msg: string = txt;
    try {
      const parsed = JSON.parse(txt) as { error?: string };
      if (parsed?.error) msg = parsed.error;
    } catch {
      // ignore
    }
    throw new Error(msg || `${res.status}`);
  }
  return JSON.parse(txt) as T;
}

type Panel = "lobby" | "leaderboard";

function UsernameGate({ onChosen }: { onChosen: (u: string) => void }) {
  const [draft, setDraft] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const submit = (): void => {
    const t = draft.trim();
    if (!USERNAME_RE.test(t)) {
      setErr("Use 1-32 characters: letters, digits, underscore (_), or hyphen (-).");
      return;
    }
    writeStoredUsername(t);
    setErr(null);
    onChosen(t);
  };

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        background: "#0a1830",
      }}
    >
      <div
        style={{
          maxWidth: 420,
          width: "100%",
          padding: 24,
          background: "#0f3460",
          borderRadius: 8,
          border: "1px solid #2a4a7f",
        }}
      >
        <h2 style={{ marginTop: 0, color: "#e8c848" }}>Choose a username</h2>
        <input
          autoFocus
          value={draft}
          onChange={(e) => {
            setDraft(e.target.value);
            setErr(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="e.g. alice"
          style={{
            width: "100%",
            boxSizing: "border-box",
            padding: "10px 12px",
            fontSize: 16,
            borderRadius: 4,
            border: "1px solid #2a4a7f",
            background: "#0a1830",
            color: "#e0e6f0",
            marginBottom: 12,
          }}
        />
        {err && <div style={{ color: "#ffb0b0", fontSize: 13, marginBottom: 12 }}>{err}</div>}
        <button
          type="button"
          onClick={submit}
          style={{
            width: "100%",
            padding: "10px 16px",
            fontSize: 15,
            fontWeight: "bold",
            background: "#e8c848",
            color: "#0a1830",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
          }}
        >
          Continue
        </button>
      </div>
    </div>
  );
}

function PlayAppLoggedIn({ username }: { username: string }) {
  const [panel, setPanel] = useState<Panel>("lobby");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [meUser, setMeUser] = useState<UserMe | null>(null);
  const [meError, setMeError] = useState<string | null>(null);

  const [games, setGames] = useState<GameListItem[]>([]);
  const [gamesError, setGamesError] = useState<string | null>(null);

  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);

  const [view, setView] = useState<PlayView | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const resumeDoneForUser = useRef<string | null>(null);

  const refreshMe = useCallback(async () => {
    try {
      const m = await getJsonAuth<UserMe>(`${PLAY_API}/me`, username);
      setMeUser(m);
      setMeError(null);
    } catch (e: unknown) {
      setMeUser(null);
      setMeError(String(e));
    }
  }, [username]);

  const refreshGames = useCallback(async () => {
    try {
      const rows = await getJsonAuth<GameListItem[]>(
        `${PLAY_API}/games?status=all`,
        username,
      );
      setGames(rows);
      setGamesError(null);
    } catch (e: unknown) {
      setGamesError(String(e));
    }
  }, [username]);

  const refreshLeaderboard = useCallback(async () => {
    try {
      const lb = await getJsonAuth<LeaderboardResponse>(`${PLAY_API}/leaderboard`, username);
      setLeaderboard(lb);
    } catch {
      setLeaderboard(null);
    }
  }, [username]);

  const updateView = useCallback(
    (next: PlayView | null) => {
      setView(next);
      if (next && next.status !== "ended") {
        writeActiveGameId(next.game_id);
      } else {
        writeActiveGameId(null);
      }
      void refreshGames();
    },
    [refreshGames],
  );

  useEffect(() => {
    setLoadingModels(true);
    void getJson<ModelInfo[]>(`${PLAY_API}/models`)
      .then((m) => {
        setModels(m);
        setLoadingModels(false);
      })
      .catch((e: unknown) => {
        setModelsError(String(e));
        setLoadingModels(false);
      });
  }, []);

  useEffect(() => {
    void refreshMe();
    void refreshGames();
    void refreshLeaderboard();
  }, [refreshGames, refreshLeaderboard, refreshMe]);

  useEffect(() => {
    if (resumeDoneForUser.current === username) return;
    resumeDoneForUser.current = username;
    const persisted = readActiveGameId();
    if (!persisted) return;
    void getJsonAuth<PlayView>(`${PLAY_API}/games/${persisted}`, username)
      .then((v) => {
        setPanel("lobby");
        setShowSetup(false);
        setView(v);
      })
      .catch(() => {
        writeActiveGameId(null);
      });
  }, [username]);

  const handleLoadGame = useCallback(
    async (gid: string) => {
      setBusy(true);
      setActionError(null);
      try {
        const v = await getJsonAuth<PlayView>(`${PLAY_API}/games/${gid}`, username);
        updateView(v);
        setShowSetup(false);
        setPanel("lobby");
      } catch (e: unknown) {
        setActionError(String(e));
      } finally {
        setBusy(false);
      }
    },
    [username, updateView],
  );

  const handleStart = useCallback(
    async (req: NewGameRequest) => {
      setStarting(true);
      setStartError(null);
      setActionError(null);
      try {
        const v = await postJsonAuth<PlayView>(`${PLAY_API}/games`, username, req);
        updateView(v);
        setShowSetup(false);
      } catch (e: unknown) {
        setStartError(String(e));
      } finally {
        setStarting(false);
      }
    },
    [username, updateView],
  );

  const handleAction = useCallback(
    async (action: number) => {
      if (!view) return;
      setBusy(true);
      setActionError(null);
      try {
        const v = await postJsonAuth<PlayView>(
          `${PLAY_API}/games/${view.game_id}/action`,
          username,
          { action },
        );
        updateView(v);
        if (v.status === "ended" && v.elo_update) {
          void refreshMe();
          void refreshLeaderboard();
        }
      } catch (e: unknown) {
        setActionError(String(e));
      } finally {
        setBusy(false);
      }
    },
    [username, view, refreshMe, refreshLeaderboard, updateView],
  );

  const handleNew = useCallback(() => {
    updateView(null);
    setShowSetup(false);
    setStartError(null);
    setActionError(null);
    void refreshGames();
  }, [refreshGames, updateView]);

  const goLobby = useCallback(() => {
    writeActiveGameId(null);
    setView(null);
    setShowSetup(false);
    setPanel("lobby");
    void refreshGames();
  }, [refreshGames]);

  const activeGames = games.filter(
    (g) => g.status === "human_turn" || g.status === "ai_thinking",
  );
  const hasInFlightGame = activeGames.length > 0;

  const tableGames = useMemo(() => gamesTableOrder(games), [games]);

  useEffect(() => {
    if (hasInFlightGame) setShowSetup(false);
  }, [hasInFlightGame]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 12px",
          background: "#0a1830",
          borderBottom: "1px solid #2a4a7f",
          flexShrink: 0,
        }}
      >
        {(["lobby", "leaderboard"] as Panel[]).map((p) => {
          const isLeaderboardTab = p === "leaderboard";
          const tabActive = panel === p && isLeaderboardTab;
          return (
            <button
              key={p}
              type="button"
              onClick={() => setPanel(p)}
              style={{
                background: tabActive ? "#e8c848" : "#0f3460",
                color: tabActive ? "#0a1830" : "#e0e6f0",
                border: "1px solid #2a4a7f",
                padding: "4px 12px",
                borderRadius: 4,
                fontSize: 12,
                cursor: "pointer",
                textTransform: "capitalize",
              }}
            >
              {p === "leaderboard" ? "Leaderboard" : "Lobby"}
            </button>
          );
        })}
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 12, color: "#7a94b8" }}>
          {meUser?.username ?? username}
          {meUser?.rating != null ? ` (${meUser.rating.toFixed(0)} Elo)` : ""}
        </span>
      </div>

      {meError && (
        <div style={{ padding: "6px 16px", background: "#3b1f1f", color: "#fff" }}>
          /me error: {meError} · is play_server running? · username header must match server rules.
        </div>
      )}
      {gamesError && (
        <div style={{ padding: "6px 16px", background: "#3b1f1f", color: "#fff" }}>
          games list error: {gamesError}
        </div>
      )}
      {actionError && !view && (
        <div style={{ padding: "6px 16px", background: "#3b1f1f", color: "#fff" }}>
          {actionError}
        </div>
      )}

      {view ? (
        <PlayGame
          view={view}
          busy={busy}
          actionError={actionError}
          onAction={handleAction}
          onNew={handleNew}
          onLobby={goLobby}
        />
      ) : panel === "leaderboard" ? (
        <div style={{ overflow: "auto", padding: 16, flex: 1 }}>
          <h2 style={{ color: "#e8c848", marginTop: 0 }}>Leaderboard</h2>
          {!leaderboard && (
            <div style={{ color: "#7a94b8" }}>Could not load leaderboard.</div>
          )}
          <table style={{ borderCollapse: "collapse", width: "100%", maxWidth: 900 }}>
            <thead>
              <tr style={{ color: "#7a94b8", textAlign: "left" }}>
                <th>#</th>
                <th>Who</th>
                <th>Kind</th>
                <th>Rating</th>
                <th>Games</th>
              </tr>
            </thead>
            <tbody>
              {(leaderboard?.combined ?? []).map((row, i) => (
                <tr key={`${row.entity_id ?? row.model_id ?? row.label}-${i}`}>
                  <td style={{ padding: "4px 8px", color: "#e0e6f0" }}>{i + 1}</td>
                  <td style={{ padding: "4px 8px", color: "#e0e6f0" }}>{row.label}</td>
                  <td style={{ padding: "4px 8px", color: "#e0e6f0" }}>{row.kind}</td>
                  <td style={{ padding: "4px 8px", color: "#e8c848" }}>
                    {(row.rating ?? row.elo).toFixed(0)}
                  </td>
                  <td style={{ padding: "4px 8px", color: "#e0e6f0" }}>{row.games}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ overflow: "auto", flex: 1, padding: 16 }}>
          <h3 style={{ color: "#e8c848", marginTop: 0, marginBottom: 28 }}>Games</h3>
          {!hasInFlightGame ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
              {!showSetup ? (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setStartError(null);
                    setShowSetup(true);
                  }}
                  style={{
                    padding: "6px 14px",
                    fontSize: 13,
                    fontWeight: 600,
                    background: "#e8c848",
                    color: "#0a1830",
                    border: "1px solid #c9ae3d",
                    borderRadius: 4,
                    cursor: busy ? "not-allowed" : "pointer",
                    opacity: busy ? 0.55 : 1,
                  }}
                >
                  New game
                </button>
              ) : (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setStartError(null);
                    setShowSetup(false);
                  }}
                  style={{
                    padding: "6px 14px",
                    fontSize: 13,
                    fontWeight: 600,
                    background: "#1a3055",
                    color: "#e0e6f0",
                    border: "1px solid #2a4a7f",
                    borderRadius: 4,
                    cursor: busy ? "not-allowed" : "pointer",
                    opacity: busy ? 0.55 : 1,
                  }}
                >
                  Cancel
                </button>
              )}
            </div>
          ) : null}
          {showSetup && !hasInFlightGame ? (
            <div style={{ marginBottom: 20, maxWidth: 720 }}>
              <PlaySetup
                models={models}
                loadingModels={loadingModels}
                modelsError={modelsError}
                starting={starting}
                startError={startError}
                onStart={handleStart}
                humanRating={meUser?.rating ?? meUser?.elo ?? null}
              />
            </div>
          ) : null}
          <table
            style={{
              borderCollapse: "collapse",
              width: "100%",
              maxWidth: 960,
            }}
          >
            <thead>
              <tr style={{ color: "#7a94b8", textAlign: "left", fontSize: 12 }}>
                <th style={{ padding: "6px 8px" }}>Status</th>
                <th style={{ padding: "6px 8px" }}>Game ID</th>
                <th style={{ padding: "6px 8px" }}>Players</th>
                <th style={{ padding: "6px 8px" }}>Steps</th>
                <th style={{ padding: "6px 8px" }}>Updated</th>
              </tr>
            </thead>
            <tbody>
              {tableGames.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ padding: "12px 8px", color: "#7a94b8" }}>
                    No games yet.
                  </td>
                </tr>
              ) : (
                tableGames.map((g) => {
                  const current = isInFlightGame(g);
                  return (
                    <tr
                      key={g.game_id}
                      onClick={() => {
                        void handleLoadGame(g.game_id);
                      }}
                      style={{
                        cursor: "pointer",
                        background: current ? "rgba(232, 200, 72, 0.14)" : "transparent",
                        borderLeft: current ? "3px solid #e8c848" : "3px solid transparent",
                        boxShadow: current ? "inset 0 0 0 1px rgba(232, 200, 72, 0.25)" : "none",
                      }}
                    >
                      <td
                        style={{
                          padding: "8px",
                          color: current ? "#f0dc7a" : "#e0e6f0",
                          fontWeight: current ? 600 : 400,
                        }}
                      >
                        {current ? `${formatGameStatus(g.status)} (current)` : formatGameStatus(g.status)}
                      </td>
                      <td style={{ padding: "8px", color: "#e0e6f0", fontFamily: "monospace", fontSize: 12 }}>
                        {g.game_id}
                      </td>
                      <td style={{ padding: "8px", color: "#e0e6f0" }}>{g.num_players}</td>
                      <td style={{ padding: "8px", color: "#e0e6f0" }}>{g.step_count}</td>
                      <td style={{ padding: "8px", color: "#7a94b8", fontSize: 12 }}>
                        {g.updated_at ?? "-"}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function PlayApp() {
  const [username, setUsername] = useState<string | null>(() => readStoredUsername());

  if (username === null) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
        <UsernameGate onChosen={(u) => setUsername(u)} />
      </div>
    );
  }

  return <PlayAppLoggedIn username={username} />;
}
