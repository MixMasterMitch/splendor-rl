import { CardData, NobleData, PlayerInfo, PlayerSnapshot, GEM_COLORS, COLOR_NAMES, GEM_TEXT_COLORS } from "../types";
import { CardTile } from "./CardTile";
import { GemIcon } from "./GemIcon";

interface PlayerPanelProps {
  playerIdx: number;
  info: PlayerInfo;
  snapshot: PlayerSnapshot;
  isActive: boolean;
  isCurrentPlayer: boolean;
  currentPhase: number;
  cardDb: CardData[];
  nobleDb: NobleData[];
  isWinner: boolean;
  // Interactive props
  affordableReservedSlots?: Set<number>;
  onReservedClick?: (rslot: number) => void;
  discardableTokens?: Set<number>;
  onTokenDiscard?: (colorIdx: number) => void;
  // Cost display
  getDisplayCosts?: (card: CardData, affordable: boolean) => number[] | undefined;
}

// Each color column is this wide so all 6 fit neatly and panels never reflow.
const COL_W = 36;
// Indices 0-4 are the 5 gem colors; index 5 is Gold.
const GEM_COLS = [0, 1, 2, 3, 4, 5] as const;

export function PlayerPanel({
  playerIdx,
  info,
  snapshot,
  isCurrentPlayer,
  currentPhase,
  cardDb,
  isWinner,
  affordableReservedSlots,
  onReservedClick,
  discardableTokens,
  onTokenDiscard,
  getDisplayCosts,
}: PlayerPanelProps) {
  const phaseLabel = isCurrentPlayer
    ? currentPhase === 0 ? " (Main)" : currentPhase === 1 ? " (Discard)" : " (Noble Pick)"
    : "";

  // Fixed panel width: 6 columns * COL_W + padding (8px each side) + border (2px each side)
  const PANEL_W = GEM_COLS.length * COL_W + 16 + 4;

  return (
    <div
      style={{
        border: `2px solid ${isCurrentPlayer ? "#e8c848" : isWinner ? "#4caf50" : "#2a4a7f"}`,
        borderRadius: 8,
        background: "#16213e",
        padding: 8,
        width: PANEL_W,
        minWidth: PANEL_W,
        maxWidth: PANEL_W,
        flexShrink: 0,
        boxShadow: isCurrentPlayer ? "0 0 10px rgba(232,200,72,0.3)" : undefined,
      }}
    >
      {/* Header: name + PV */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 6,
          gap: 4,
        }}
      >
        <span
          style={{
            fontWeight: "bold",
            fontSize: 12,
            color: isCurrentPlayer ? "#e8c848" : isWinner ? "#4caf50" : "#e0e6f0",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          P{playerIdx}: {info.name}{phaseLabel}
        </span>
        <span
          style={{
            fontWeight: "bold",
            fontSize: 15,
            color: isWinner ? "#4caf50" : "#e0e6f0",
            flexShrink: 0,
          }}
        >
          {snapshot.points} PV
        </span>
      </div>

      {/*
        Color table: 6 columns (W B G R K Gold), 2 rows.
        Row 1: token chips (circular badge with count).
        Row 2: purchased-card bonus count (colored pill, 0 shown dimmed).
        Gold column only has tokens (no bonuses), so row 2 col 5 is empty.
      */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${GEM_COLS.length}, ${COL_W}px)`,
          rowGap: 4,
          marginBottom: 8,
        }}
      >
        {/* Row 1: tokens */}
        {GEM_COLS.map((i) => {
          const count = snapshot.tokens[i] ?? 0;
          const bg = GEM_COLORS[i] ?? "#888";
          const fg = GEM_TEXT_COLORS[i] ?? "#fff";
          const canDiscard = discardableTokens?.has(i) ?? false;
          return (
            <div
              key={`tok-${i}`}
              title={`${COLOR_NAMES[i] ?? "?"} tokens: ${count}${canDiscard ? " (click to discard)" : ""}`}
              onClick={canDiscard ? () => onTokenDiscard?.(i) : undefined}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
                cursor: canDiscard ? "pointer" : undefined,
              }}
            >
              {/* Gem icon */}
              <div
                style={{
                  width: 26,
                  height: 26,
                  borderRadius: "50%",
                  background: bg,
                  border: canDiscard ? "2.5px solid #ef5350" : "2px solid rgba(255,255,255,0.15)",
                  boxShadow: canDiscard ? "0 0 6px rgba(239,83,80,0.5)" : undefined,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  opacity: count === 0 ? 0.25 : 1,
                  flexShrink: 0,
                }}
              >
                <GemIcon colorIdx={i} size={16} fill={fg} />
              </div>
              {/* Count below chip */}
              <span
                style={{
                  fontSize: 11,
                  fontWeight: "bold",
                  color: count === 0 ? "#3a5070" : "#e0e6f0",
                  lineHeight: 1,
                }}
              >
                {count}
              </span>
            </div>
          );
        })}

        {/* Row 2: card bonuses (only 5 colors, gold col is blank) */}
        {GEM_COLS.map((i) => {
          if (i === 5) {
            // Gold has no card bonus
            return <div key={`bon-${i}`} />;
          }
          const count = snapshot.bonuses[i] ?? 0;
          const bg = GEM_COLORS[i] ?? "#888";
          const fg = GEM_TEXT_COLORS[i] ?? "#fff";
          return (
            <div
              key={`bon-${i}`}
              title={`${COLOR_NAMES[i] ?? "?"} card bonuses: ${count}`}
              style={{ display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: count === 0 ? "transparent" : bg,
                  color: count === 0 ? "#3a5070" : fg,
                  borderRadius: 4,
                  border: count === 0 ? "1px solid #2a4a7f" : "1px solid rgba(255,255,255,0.15)",
                  width: 26,
                  height: 18,
                  fontSize: 11,
                  fontWeight: "bold",
                }}
              >
                +{count}
              </span>
            </div>
          );
        })}
      </div>

      {/* Reserved cards */}
      <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
        {snapshot.reserved.map((cardId, r) => {
          const card = cardId != null ? (cardDb[cardId] ?? null) : null;
          const isHidden = !!snapshot.reserved_hidden[r];
          const isAffordable = affordableReservedSlots?.has(r) ?? false;
          return (
            <CardTile
              key={r}
              card={card}
              hidden={isHidden}
              small
              affordable={isAffordable}
              onClick={isAffordable ? () => onReservedClick?.(r) : undefined}
              displayCosts={card && getDisplayCosts ? getDisplayCosts(card, isAffordable) : undefined}
            />
          );
        })}
      </div>

      {/* Nobles */}
      {snapshot.nobles_claimed > 0 && (
        <div style={{ fontSize: 11, color: "#c0a0d0" }}>
          Nobles claimed: {snapshot.nobles_claimed}
        </div>
      )}
    </div>
  );
}
