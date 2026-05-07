import { useEffect, useRef, useState } from "react";
import { Board } from "./Board";
import { ActionLog } from "./ActionLog";
import { PlayView, PlaySeatInfo } from "../play_types";
import { ReplayStep, PlayerInfo } from "../types";

interface PlayGameProps {
  view: PlayView;
  busy: boolean;
  actionError: string | null;
  onAction: (action: number) => void;
  onNew: () => void;
  placed?: boolean;
  wins?: number;
}

function seatLabel(p: PlaySeatInfo): string {
  if (p.kind === "human") return `You (seat ${p.seat + 1})`;
  return `${p.label} (seat ${p.seat + 1})`;
}

function ratingDeltaText(view: PlayView, placed?: boolean, wins?: number): string | null {
  if (!view.elo_update) return null;
  if (!placed) {
    const w = wins ?? 0;
    return `Placement match (${w}/5 wins)`;
  }
  const u = view.elo_update;
  const oldR = u.old_rating ?? u.old_elo;
  const newR = u.new_rating ?? u.new_elo;
  const sign = u.delta >= 0 ? "+" : "";
  return `Rating: ${oldR.toFixed(0)} → ${newR.toFixed(0)} (${sign}${u.delta.toFixed(1)})`;
}

export function PlayGame({
  view,
  busy,
  actionError,
  onAction,
  onNew,
  placed,
  wins,
}: PlayGameProps) {
  const logEndRef = useRef<HTMLDivElement | null>(null);
  const isEnded = view.status === "ended";

  // Replay mode: when game is ended, allow scrubbing through steps.
  // viewStep 0 = initial state
  // viewStep 1..N = selecting step N (shows state BEFORE that action, highlights what it touches)
  // viewStep N+1 = "Final State" (shows final snapshot, no highlights, green winner scores)
  const finalStateIdx = view.steps.length + 1;
  const [viewStep, setViewStep] = useState<number>(isEnded ? finalStateIdx : view.steps.length);

  // When game ends, jump to final state. While in progress, track latest step.
  useEffect(() => {
    if (isEnded) {
      setViewStep(finalStateIdx);
    } else {
      setViewStep(view.steps.length);
    }
  }, [view.steps.length, isEnded, finalStateIdx]);

  useEffect(() => {
    if (!isEnded) {
      logEndRef.current?.scrollIntoView({ block: "end" });
    }
  }, [view.steps.length, isEnded]);

  // Determine which snapshot to show based on viewStep.
  const isFinalState = viewStep >= finalStateIdx;
  const displaySnapshot = (() => {
    if (isFinalState) return view.snapshot;
    if (!isEnded) {
      return view.snapshot;
    }
    if (viewStep === 0) return view.initial_state;
    if (viewStep === 1) return view.initial_state;
    return view.steps[viewStep - 2].state_after;
  })();

  const snapshot = displaySnapshot;
  const humanSeat = view.human_seat;

  const playerInfos: PlayerInfo[] = view.players.map((p) => ({
    name: p.kind === "human" ? "You" : p.label,
    kind: p.kind,
  }));

  const replaySteps: ReplayStep[] = view.steps.map((s) => ({
    step: s.step,
    player: s.player,
    phase: s.phase,
    action: s.action,
    action_name: s.action_name,
    action_detail: s.action_detail,
    legal_actions: s.legal_actions,
    state_after: s.state_after,
  }));

  const winners = view.winners ?? [];
  const isHumanTurn = view.status === "human_turn";
  const ratingLine = ratingDeltaText(view, placed, wins);

  // Highlight the action that is selected at the current viewStep (replay mode only).
  const highlightedAction = (() => {
    if (!isEnded) return undefined; // During live play, board is interactive — no replay highlights
    if (viewStep >= 1 && viewStep <= view.steps.length) {
      return view.steps[viewStep - 1].action_detail;
    }
    return undefined;
  })();

  const handleJump = (stepIdx: number) => {
    if (isEnded) {
      setViewStep(Math.max(0, Math.min(stepIdx, finalStateIdx)));
    }
  };

  const showWinners = isEnded && isFinalState;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Status bar */}
      <div
        style={{
          padding: "8px 16px",
          background: "#0a1830",
          borderBottom: "1px solid #2a4a7f",
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 12, color: "#7a94b8" }}>
          Game {view.game_id} | seed {view.seed} | {view.num_players}p
        </span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 13 }}>
          {view.players.map((p) => (
            <span
              key={p.seat}
              style={{
                color: view.current_player === p.seat ? "#e8c848" : "#e0e6f0",
                marginRight: 12,
                fontWeight: view.current_player === p.seat ? "bold" : "normal",
              }}
            >
              {seatLabel(p)}
              {p.rating != null ? ` [${p.rating.toFixed(0)}]` : ""}
            </span>
          ))}
        </span>
        {!isEnded ? null : (
          <button
            onClick={onNew}
            style={{
              background: "#0f3460",
              color: "#e0e6f0",
              border: "1px solid #2a4a7f",
              padding: "4px 12px",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            New game
          </button>
        )}
      </div>

      {isEnded && (
        <div
          style={{
            padding: "10px 16px",
            background: "#0a1830",
            borderBottom: "1px solid #2a4a7f",
            color: "#4caf50",
            fontWeight: "bold",
            display: "flex",
            gap: 12,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          {view.aborted ? (
            <span style={{ color: "#ef5350" }}>Game aborted (no rating change)</span>
          ) : (
            <>
              <span>
                {winners.includes(humanSeat)
                  ? "You win!"
                  : `Winner: ${winners
                        .map((s) => view.players[s]?.label ?? `seat ${s + 1}`)
                        .join(", ")}`}
              </span>
              {ratingLine && <span style={{ color: "#e8c848" }}>{ratingLine}</span>}
            </>
          )}
        </div>
      )}

      {actionError && (
        <div style={{ padding: "6px 16px", background: "#3b1f1f", color: "#fff" }}>
          {actionError}
        </div>
      )}

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Board (now handles all interaction) */}
        <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
          <Board
            snapshot={snapshot}
            cardDb={view.cards}
            nobleDb={view.nobles}
            playerInfos={playerInfos}
            winners={showWinners ? winners : []}
            highlightedAction={highlightedAction}
            interactive={isHumanTurn}
            legalActions={isHumanTurn ? (view.legal_actions ?? []) : []}
            humanSeat={humanSeat}
            busy={busy}
            onAction={onAction}
            aiThinking={view.status === "ai_thinking" || busy}
          />
        </div>

        {/* Right column: Action log only */}
        <div
          style={{
            width: 280,
            borderLeft: "1px solid #2a4a7f",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            flexShrink: 0,
          }}
        >
          <ActionLog
            steps={replaySteps}
            currentStep={viewStep}
            totalSteps={view.steps.length}
            playerInfos={playerInfos}
            onJump={handleJump}
            cardDb={view.cards}
            initialState={view.initial_state}
            isEnded={isEnded}
            winners={winners}
          />
          <div ref={logEndRef} />
        </div>
      </div>

      {/* Timeline slider (only shown when game is ended) */}
      {isEnded && view.steps.length > 0 && (
        <div
          style={{
            padding: "8px 16px",
            background: "#0a1830",
            borderTop: "1px solid #2a4a7f",
            display: "flex",
            alignItems: "center",
            gap: 10,
            flexShrink: 0,
          }}
        >
          <button
            type="button"
            onClick={() => handleJump(0)}
            style={{
              background: "none",
              border: "none",
              color: "#7a94b8",
              cursor: "pointer",
              fontSize: 14,
              padding: "2px 6px",
            }}
            title="Go to start"
          >
            ⏮
          </button>
          <button
            type="button"
            onClick={() => handleJump(Math.max(0, viewStep - 1))}
            style={{
              background: "none",
              border: "none",
              color: "#7a94b8",
              cursor: "pointer",
              fontSize: 14,
              padding: "2px 6px",
            }}
            title="Previous step"
          >
            ◀
          </button>
          <input
            type="range"
            min={0}
            max={finalStateIdx}
            value={viewStep}
            onChange={(e) => handleJump(Number(e.target.value))}
            style={{ flex: 1, accentColor: "#e8c848" }}
          />
          <button
            type="button"
            onClick={() => handleJump(Math.min(finalStateIdx, viewStep + 1))}
            style={{
              background: "none",
              border: "none",
              color: "#7a94b8",
              cursor: "pointer",
              fontSize: 14,
              padding: "2px 6px",
            }}
            title="Next step"
          >
            ▶
          </button>
          <button
            type="button"
            onClick={() => handleJump(finalStateIdx)}
            style={{
              background: "none",
              border: "none",
              color: "#7a94b8",
              cursor: "pointer",
              fontSize: 14,
              padding: "2px 6px",
            }}
            title="Go to final state"
          >
            ⏭
          </button>
          <span style={{ color: "#7a94b8", fontSize: 11, minWidth: 80, textAlign: "right" }}>
            {isFinalState ? "Final" : viewStep === 0 ? "Start" : `${viewStep} / ${view.steps.length}`}
          </span>
        </div>
      )}
    </div>
  );
}
