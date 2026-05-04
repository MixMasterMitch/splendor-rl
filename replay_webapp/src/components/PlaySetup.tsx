import { useEffect, useMemo, useState } from "react";
import { ModelInfo, NewGameRequest } from "../play_types";

interface PlaySetupProps {
  models: ModelInfo[];
  loadingModels: boolean;
  modelsError: string | null;
  starting: boolean;
  startError: string | null;
  onStart: (req: NewGameRequest) => void;
  humanRating: number | null;
}

const NET_MCTS_SIMS = 64;

export function PlaySetup({
  models,
  loadingModels,
  modelsError,
  starting,
  startError,
  onStart,
  humanRating,
}: PlaySetupProps) {
  const [numPlayers, setNumPlayers] = useState(2);
  const [humanSeat, setHumanSeat] = useState(0);
  const [seatModels, setSeatModels] = useState<Record<number, string>>({});

  const sortedModels = useMemo(() => {
    return [...models].sort((a, b) => {
      if (a.kind !== b.kind) {
        const order: Record<string, number> = {
          net: 0,
          heuristic_opus: 1,
          heuristic: 2,
          random: 3,
          human: 9,
        };
        return (order[a.kind] ?? 9) - (order[b.kind] ?? 9);
      }
      const aR = a.rating ?? a.elo ?? 0;
      const bR = b.rating ?? b.elo ?? 0;
      return bR - aR;
    });
  }, [models]);

  const pickableModels = useMemo(() => {
    return sortedModels.filter((m) => {
      if (m.kind === "human") return false;
      if (numPlayers > 2 && m.kind === "net") return false;
      return true;
    });
  }, [sortedModels, numPlayers]);

  const validIds = useMemo(
    () => new Set(pickableModels.map((m) => m.id)),
    [pickableModels],
  );

  // Default seat assignments. Re-runs both when player counts change *and*
  // whenever the available model list changes, since on first mount we are
  // called before `models` has loaded.
  useEffect(() => {
    if (humanSeat >= numPlayers) setHumanSeat(0);
    if (pickableModels.length === 0) return;
    const defaultModel =
      pickableModels.find((m) => m.kind === "net") ??
      pickableModels.find((m) => m.kind === "heuristic_opus") ??
      pickableModels.find((m) => m.kind === "heuristic") ??
      pickableModels[0];
    setSeatModels((prev) => {
      const next: Record<number, string> = {};
      let changed = false;
      for (let s = 0; s < numPlayers; s++) {
        if (s === humanSeat) continue;
        const existing = prev[s];
        const valid = existing && validIds.has(existing);
        next[s] = valid ? existing : (defaultModel?.id ?? "");
        if (next[s] !== existing) changed = true;
      }
      // Drop any stale entries from removed seats.
      for (const k of Object.keys(prev)) {
        if (!(k in next)) {
          changed = true;
          break;
        }
      }
      return changed ? next : prev;
    });
  }, [numPlayers, humanSeat, pickableModels, validIds]);

  const canStart =
    !starting &&
    !loadingModels &&
    pickableModels.length > 0 &&
    Array.from({ length: numPlayers }, (_, i) => i)
      .filter((s) => s !== humanSeat)
      .every((s) => seatModels[s] && validIds.has(seatModels[s]));

  function handleStart() {
    const opponents: Record<number, string> = {};
    for (let s = 0; s < numPlayers; s++) {
      if (s === humanSeat) continue;
      opponents[s] = seatModels[s];
    }
    const req: NewGameRequest = {
      num_players: numPlayers,
      human_seat: humanSeat,
      opponents,
      num_sims: NET_MCTS_SIMS,
    };
    onStart(req);
  }

  return (
    <div
      style={{
        maxWidth: 720,
        margin: "32px auto",
        padding: 24,
        background: "#0a1830",
        border: "1px solid #2a4a7f",
        borderRadius: 8,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h2 style={{ margin: 0, color: "#e8c848" }}>New game</h2>
        <span style={{ color: "#7a94b8", fontSize: 13 }}>
          Your rating: {humanRating != null ? humanRating.toFixed(0) : "-"}
        </span>
      </div>

      {modelsError && (
        <div style={{ color: "#ef5350" }}>Models error: {modelsError}</div>
      )}

      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 12, color: "#7a94b8" }}>Number of players</span>
        <select
          value={numPlayers}
          onChange={(e) => setNumPlayers(Number(e.target.value))}
        >
          {[2, 3, 4].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>

      {numPlayers > 2 && (
        <div style={{ fontSize: 12, color: "#7a94b8" }}>
          3 or 4 player games exclude trained net checkpoints; use built-in bots
          only.
        </div>
      )}

      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 12, color: "#7a94b8" }}>Your seat</span>
        <select
          value={humanSeat}
          onChange={(e) => setHumanSeat(Number(e.target.value))}
        >
          {Array.from({ length: numPlayers }, (_, i) => (
            <option key={i} value={i}>
              Seat {i}
            </option>
          ))}
        </select>
      </label>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontSize: 12, color: "#7a94b8" }}>Opponents</span>
        {Array.from({ length: numPlayers }, (_, i) => i)
          .filter((s) => s !== humanSeat)
          .map((s) => (
            <label
              key={s}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
              }}
            >
              <span style={{ minWidth: 60 }}>Seat {s}</span>
              <select
                value={seatModels[s] ?? ""}
                onChange={(e) =>
                  setSeatModels((prev) => ({ ...prev, [s]: e.target.value }))
                }
                style={{ flex: 1 }}
                disabled={loadingModels}
              >
                {pickableModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
          ))}
      </div>

      {startError && <div style={{ color: "#ef5350" }}>{startError}</div>}

      <button
        disabled={!canStart}
        onClick={handleStart}
        style={{
          background: canStart ? "#0f3460" : "#1a2a4a",
          border: "1px solid #2a4a7f",
          color: "#e0e6f0",
          padding: "10px 16px",
          borderRadius: 4,
          fontSize: 14,
          cursor: canStart ? "pointer" : "not-allowed",
        }}
      >
        {starting ? "Starting..." : "Start game"}
      </button>
    </div>
  );
}
