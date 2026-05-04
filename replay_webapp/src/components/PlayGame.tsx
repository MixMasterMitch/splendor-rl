import { useEffect, useRef } from "react";
import { Board } from "./Board";
import { LegalActionsPanel } from "./LegalActionsPanel";
import { ActionLog } from "./ActionLog";
import { PlayView, PlaySeatInfo } from "../play_types";
import { ReplayStep, PlayerInfo } from "../types";

interface PlayGameProps {
  view: PlayView;
  busy: boolean;
  actionError: string | null;
  onAction: (action: number) => void;
  onNew: () => void;
  onLobby?: () => void;
}

function seatLabel(p: PlaySeatInfo): string {
  if (p.kind === "human") return `You (seat ${p.seat})`;
  return `${p.label} (seat ${p.seat})`;
}

function ratingDeltaText(view: PlayView): string | null {
  if (!view.elo_update) return null;
  const u = view.elo_update;
  const oldR = u.old_rating ?? u.old_elo;
  const newR = u.new_rating ?? u.new_elo;
  const sign = u.delta >= 0 ? "+" : "";
  return `Rating: ${oldR.toFixed(0)} -> ${newR.toFixed(0)} (${sign}${u.delta.toFixed(1)})`;
}

export function PlayGame({
  view,
  busy,
  actionError,
  onAction,
  onNew,
  onLobby,
}: PlayGameProps) {
  const logEndRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "end" });
  }, [view.steps.length]);

  const snapshot = view.snapshot;
  const humanSeat = view.human_seat;
  const youPlayer = view.snapshot.players[humanSeat];
  const reservedIds = youPlayer.reserved;

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
  const isEnded = view.status === "ended";
  const ratingLine = ratingDeltaText(view);

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
        {onLobby && (
          <button
            type="button"
            onClick={onLobby}
            disabled={busy}
            style={{
              background: "#1a3055",
              color: "#e0e6f0",
              border: "1px solid #2a4a7f",
              padding: "4px 10px",
              borderRadius: 4,
              cursor: busy ? "not-allowed" : "pointer",
              fontSize: 12,
            }}
          >
            Lobby
          </button>
        )}
        <span style={{ fontSize: 13 }}>
          {view.players.map((p, i) => (
            <span
              key={p.seat}
              style={{
                color: view.current_player === p.seat ? "#e8c848" : "#e0e6f0",
                marginRight: 12,
                fontWeight: view.current_player === p.seat ? "bold" : "normal",
              }}
            >
              {seatLabel(p)}
              {(p.rating ?? p.elo) != null ? ` [${(p.rating ?? p.elo)!.toFixed(0)}]` : ""}
              {i === view.players.length - 1 ? "" : ""}
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
                {winners.includes(humanSeat) && winners.length === 1
                  ? "You win!"
                  : winners.includes(humanSeat)
                    ? `Tie (you and ${winners.length - 1} other)`
                    : `Winner: ${winners
                        .map((s) => view.players[s]?.label ?? `seat ${s}`)
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
        {/* Board */}
        <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
          <Board
            snapshot={snapshot}
            cardDb={view.cards}
            nobleDb={view.nobles}
            playerInfos={playerInfos}
            winners={winners}
          />
        </div>

        {/* Right column: Actions on top, Action log below */}
        <div
          style={{
            width: 340,
            borderLeft: "1px solid #2a4a7f",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              flex: "0 0 auto",
              maxHeight: "60%",
              overflowY: "auto",
              borderBottom: "1px solid #2a4a7f",
            }}
          >
            {isHumanTurn ? (
              <LegalActionsPanel
                legalActions={view.legal_actions ?? []}
                cards={view.cards}
                nobles={view.nobles}
                grid={snapshot.grid}
                reservedIds={reservedIds}
                noblesOnBoard={snapshot.nobles}
                humanPlayer={youPlayer}
                gemPool={snapshot.gem_pool}
                busy={busy}
                onAction={onAction}
              />
            ) : (
              <div
                style={{
                  padding: 16,
                  fontSize: 13,
                  color: "#7a94b8",
                  fontStyle: "italic",
                }}
              >
                {view.status === "ai_thinking"
                  ? "AI is thinking..."
                  : "Game over."}
              </div>
            )}
          </div>
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <ActionLog
              steps={replaySteps}
              currentStep={view.steps.length}
              playerInfos={playerInfos}
              onJump={() => {}}
            />
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
