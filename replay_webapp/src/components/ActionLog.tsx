import { useEffect, useRef } from "react";
import { ReplayStep, PlayerInfo, CardData, GameSnapshot, GEM_COLORS, GEM_TEXT_COLORS, PHASE_LABELS } from "../types";
import { GemIcon } from "./GemIcon";

interface ActionLogProps {
  steps: ReplayStep[];
  currentStep: number;  // 0 = initial state, 1..N = after step, N+1 = final state
  totalSteps: number;   // steps.length (used to detect "final state" selection)
  playerInfos: PlayerInfo[];
  onJump: (stepIdx: number) => void;
  cardDb: CardData[];
  initialState: GameSnapshot;
  isEnded: boolean;
  winners: number[];
}

const PLAYER_COLORS = ["#e8c848", "#4caf50", "#42a5f5", "#ef5350"];

/** Tiny inline card rectangle showing bonus color + points */
function MiniCard({ card }: { card: CardData | null }) {
  if (!card) {
    // Unknown/hidden card — show a "?" placeholder
    return (
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 24,
          height: 14,
          borderRadius: 3,
          border: "1px solid #2a4a7f",
          background: "#16213e",
          color: "#7a94b8",
          fontSize: 8,
          fontWeight: "bold",
          verticalAlign: "middle",
          marginLeft: 3,
        }}
        title="Hidden card"
      >
        ?
      </span>
    );
  }
  const bg = GEM_COLORS[card.bonus] ?? "#888";
  const fg = GEM_TEXT_COLORS[card.bonus] ?? "#fff";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        minWidth: 28,
        height: 14,
        borderRadius: 3,
        background: bg,
        color: fg,
        fontSize: 8,
        fontWeight: "bold",
        verticalAlign: "middle",
        marginLeft: 3,
        padding: "0 3px",
      }}
      title={`${card.points} pts, ${["White","Blue","Green","Red","Black"][card.bonus]} bonus`}
    >
      {card.points} pts
    </span>
  );
}

/** Tiny inline gem circle */
function MiniGem({ colorIdx }: { colorIdx: number }) {
  const bg = GEM_COLORS[colorIdx] ?? "#888";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 12,
        height: 12,
        borderRadius: "50%",
        background: bg,
        border: "1px solid rgba(255,255,255,0.25)",
        verticalAlign: "middle",
        marginLeft: 2,
      }}
      title={["White","Blue","Green","Red","Black","Gold"][colorIdx]}
    >
      <GemIcon colorIdx={colorIdx} size={8} fill={GEM_TEXT_COLORS[colorIdx] ?? "#fff"} />
    </span>
  );
}

/** Tiny gold/joker token for reserves */
function MiniJoker() {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 12,
        height: 12,
        borderRadius: "50%",
        background: GEM_COLORS[5],
        border: "1px solid rgba(255,255,255,0.25)",
        verticalAlign: "middle",
        marginLeft: 3,
      }}
      title="Gold token"
    >
      <GemIcon colorIdx={5} size={8} fill={GEM_TEXT_COLORS[5] ?? "#333"} />
    </span>
  );
}

function lookupCard(cardDb: CardData[], cardId: number | null): CardData | null {
  if (cardId == null) return null;
  return cardDb.find((c) => c.id === cardId) ?? null;
}

function formatActionInline(
  step: ReplayStep,
  stateBefore: GameSnapshot,
  cardDb: CardData[],
): JSX.Element {
  const d = step.action_detail;
  switch (d.kind) {
    case "take3": {
      return (
        <span>
          Take 3:{" "}
          {(d.colors ?? []).map((c, i) => (
            <MiniGem key={i} colorIdx={c} />
          ))}
        </span>
      );
    }
    case "take2": {
      const c = d.color ?? 0;
      return (
        <span>
          Take 2: <MiniGem colorIdx={c} /><MiniGem colorIdx={c} />
        </span>
      );
    }
    case "reserve_grid": {
      const tier = d.tier ?? 0;
      const slot = d.slot ?? 0;
      const cardId = stateBefore.grid[tier]?.[slot] ?? null;
      const card = lookupCard(cardDb, cardId);
      return (
        <span style={{ display: "inline-flex", alignItems: "center" }}>
          Reserve L{tier + 1}: <MiniCard card={card} /><MiniJoker />
        </span>
      );
    }
    case "reserve_blind": {
      const tier = d.tier ?? 0;
      return (
        <span style={{ display: "inline-flex", alignItems: "center" }}>
          Reserve blind L{tier + 1}<MiniJoker />
        </span>
      );
    }
    case "buy_grid": {
      const tier = d.tier ?? 0;
      const slot = d.slot ?? 0;
      const cardId = stateBefore.grid[tier]?.[slot] ?? null;
      const card = lookupCard(cardDb, cardId);
      return (
        <span style={{ display: "inline-flex", alignItems: "center" }}>
          Buy L{tier + 1}: <MiniCard card={card} />
        </span>
      );
    }
    case "buy_reserved": {
      const rslot = d.slot ?? 0;
      const player = stateBefore.players[step.player];
      const cardId = player?.reserved?.[rslot] ?? null;
      const card = lookupCard(cardDb, cardId);
      return (
        <span style={{ display: "inline-flex", alignItems: "center" }}>
          Buy reserved: <MiniCard card={card} />
        </span>
      );
    }
    case "pass":
      return <span>Pass</span>;
    case "discard": {
      const token = d.token ?? 0;
      return (
        <span>
          Discard <MiniGem colorIdx={token} />
        </span>
      );
    }
    case "pick_noble":
      return <span>Pick noble</span>;
    default:
      return <span>{step.action_name}</span>;
  }
}

export function ActionLog({ steps, currentStep, totalSteps, playerInfos, onJump, cardDb, initialState, isEnded, winners }: ActionLogProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);

  const isFinalState = currentStep === totalSteps + 1;

  // Scroll active item into view when currentStep changes
  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [currentStep]);

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
          fontSize: 12,
          fontWeight: "bold",
          color: "#7a94b8",
          padding: "6px 8px",
          borderBottom: "1px solid #2a4a7f",
          flexShrink: 0,
        }}
      >
        Action Log ({steps.length} steps)
      </div>

      {/* Initial state marker */}
      <div
        onClick={() => onJump(0)}
        ref={currentStep === 0 ? activeRef : undefined}
        style={{
          padding: "4px 8px",
          fontSize: 11,
          cursor: "pointer",
          background: currentStep === 0 ? "rgba(232,200,72,0.15)" : "transparent",
          borderLeft: currentStep === 0 ? "3px solid #e8c848" : "3px solid transparent",
          color: "#7a94b8",
          flexShrink: 0,
        }}
      >
        [Initial state]
      </div>

      <div
        ref={listRef}
        style={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {steps.map((step, i) => {
          const stepIdx = i + 1;  // stepIdx 1..N matches "after step i"
          const isActive = currentStep === stepIdx;
          const playerColor = PLAYER_COLORS[step.player] ?? "#e0e6f0";
          const playerName = playerInfos[step.player]?.name ?? `P${step.player + 1}`;
          const phaseLabel = PHASE_LABELS[step.phase] ?? "";

          // State before this action: initial_state for step 0, otherwise previous step's state_after
          const stateBefore = i === 0 ? initialState : steps[i - 1].state_after;

          return (
            <div
              key={step.step}
              ref={isActive ? activeRef : undefined}
              onClick={() => onJump(stepIdx)}
              style={{
                padding: "4px 8px",
                fontSize: 11,
                cursor: "pointer",
                background: isActive ? "rgba(232,200,72,0.12)" : "transparent",
                borderLeft: isActive ? "3px solid #e8c848" : "3px solid transparent",
                display: "flex",
                flexDirection: "column",
                gap: 1,
                transition: "background 0.1s",
              }}
            >
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span
                  style={{
                    color: "#7a94b8",
                    minWidth: 26,
                    fontSize: 10,
                  }}
                >
                  #{step.step}
                </span>
                <span
                  style={{
                    color: playerColor,
                    fontWeight: "bold",
                    fontSize: 10,
                  }}
                >
                  {playerName}
                </span>
                {phaseLabel && (
                  <span style={{ color: "#7a94b8", fontSize: 9 }}>
                    [{phaseLabel}]
                  </span>
                )}
              </div>
              <div
                style={{
                  color: isActive ? "#e8c848" : "#c0d0e8",
                  paddingLeft: 32,
                  display: "flex",
                  alignItems: "center",
                }}
              >
                {formatActionInline(step, stateBefore, cardDb)}
              </div>
            </div>
          );
        })}

        {/* Final State marker (only when game is ended) */}
        {isEnded && (
          <div
            onClick={() => onJump(totalSteps + 1)}
            ref={isFinalState ? activeRef : undefined}
            style={{
              padding: "4px 8px",
              fontSize: 11,
              cursor: "pointer",
              background: isFinalState ? "rgba(76,175,80,0.15)" : "transparent",
              borderLeft: isFinalState ? "3px solid #4caf50" : "3px solid transparent",
              color: isFinalState ? "#4caf50" : "#7a94b8",
              fontWeight: "bold",
              flexShrink: 0,
              marginTop: 2,
            }}
          >
            [Final State]
            {winners.length > 0 && (
              <span style={{ color: "#4caf50", fontWeight: "normal", marginLeft: 6 }}>
                Winner: {winners.map((w) => playerInfos[w]?.name ?? `P${w}`).join(", ")}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
