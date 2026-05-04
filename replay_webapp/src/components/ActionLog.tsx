import { useEffect, useRef } from "react";
import { ReplayStep, PlayerInfo, PHASE_LABELS } from "../types";

interface ActionLogProps {
  steps: ReplayStep[];
  currentStep: number;  // 0 = initial state, 1 = after step 1, etc.
  playerInfos: PlayerInfo[];
  onJump: (stepIdx: number) => void;
}

const PLAYER_COLORS = ["#e8c848", "#4caf50", "#42a5f5", "#ef5350"];

function formatActionDetail(step: ReplayStep): string {
  const d = step.action_detail;
  switch (d.kind) {
    case "take3": {
      const names = (d.colors ?? []).map((c) => "WBGRK"[c] ?? "?").join(", ");
      return `Take 3: ${names}`;
    }
    case "take2": {
      const name = "WBGRK"[d.color ?? 0] ?? "?";
      return `Take 2: ${name}${name}`;
    }
    case "reserve_grid":
      return `Reserve L${(d.tier ?? 0) + 1} slot ${d.slot ?? 0}`;
    case "reserve_blind":
      return `Reserve blind L${(d.tier ?? 0) + 1}`;
    case "buy_grid":
      return `Buy L${(d.tier ?? 0) + 1} slot ${d.slot ?? 0}`;
    case "buy_reserved":
      return `Buy reserved slot ${d.slot ?? 0}`;
    case "pass":
      return "Pass";
    case "discard": {
      const name = "WBGRKg"[d.token ?? 0] ?? "?";
      return `Discard ${name}`;
    }
    case "pick_noble":
      return `Pick noble (slot ${d.slot ?? 0})`;
    default:
      return step.action_name;
  }
}

export function ActionLog({ steps, currentStep, playerInfos, onJump }: ActionLogProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLDivElement>(null);

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
          const playerName = playerInfos[step.player]?.name ?? `P${step.player}`;
          const phaseLabel = PHASE_LABELS[step.phase] ?? "";

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
                }}
              >
                {formatActionDetail(step)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
