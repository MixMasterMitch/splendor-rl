import { useState } from "react";
import { CardData, NobleData, GEM_COLORS, GEM_TEXT_COLORS, PlayerSnapshot } from "../types";
import {
  COLOR_LETTERS,
  DecodedAction,
  decodeAction,
} from "../play_actions";
import { GemIcon } from "./GemIcon";

interface LegalActionsPanelProps {
  legalActions: number[];
  cards: CardData[];
  nobles: NobleData[];
  grid: (number | null)[][];
  reservedIds: (number | null)[];
  noblesOnBoard: (number | null)[];
  humanPlayer: PlayerSnapshot;
  gemPool: number[];
  busy: boolean;
  onAction: (action: number) => void;
}

// Compute the actual token cost the human will pay for a card, factoring in
// their bonuses and substituting jokers (gold tokens) for any deficit. The
// engine only enumerates legal buys, so we can assume the card is affordable.
function netBuyCost(
  card: CardData,
  player: PlayerSnapshot,
): { perColor: number[]; jokers: number; total: number } {
  const perColor: number[] = [0, 0, 0, 0, 0];
  let jokers = 0;
  for (let i = 0; i < 5; i++) {
    const needed = Math.max(0, card.cost[i] - (player.bonuses[i] ?? 0));
    if (needed === 0) continue;
    const have = player.tokens[i] ?? 0;
    const fromTokens = Math.min(needed, have);
    perColor[i] = fromTokens;
    jokers += needed - fromTokens;
  }
  const total = perColor.reduce((a, b) => a + b, 0) + jokers;
  return { perColor, jokers, total };
}

const KIND_TITLES: Record<string, string> = {
  take3: "Take 3 different gems",
  take2: "Take 2 of the same color",
  reserve_grid: "Reserve a visible card",
  reserve_blind: "Reserve from deck (blind)",
  buy_grid: "Buy a visible card",
  buy_reserved: "Buy a reserved card",
  pass: "Pass",
  discard: "Discard a token (over 10)",
  pick_noble: "Choose noble",
};

const KIND_ORDER: string[] = [
  "buy_grid",
  "buy_reserved",
  "take3",
  "take2",
  "reserve_grid",
  "reserve_blind",
  "discard",
  "pick_noble",
  "pass",
];

const TIER_LABELS = ["Lvl 1", "Lvl 2", "Lvl 3"];

function ColorPip({ color, size = 14 }: { color: number; size?: number }) {
  return (
    <span
      style={{
        display: "inline-flex",
        width: size,
        height: size,
        borderRadius: "50%",
        background: GEM_COLORS[color] ?? "#888",
        alignItems: "center",
        justifyContent: "center",
        marginRight: 2,
      }}
    >
      <GemIcon
        colorIdx={color}
        size={size - 4}
        fill={GEM_TEXT_COLORS[color] ?? "#fff"}
      />
    </span>
  );
}

function CostBreakdown({
  card,
  player,
}: {
  card: CardData;
  player: PlayerSnapshot;
}) {
  const cost = netBuyCost(card, player);
  if (cost.total === 0) {
    return (
      <span style={{ color: "#4caf50", fontSize: 11 }}>free</span>
    );
  }
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
      {cost.perColor.map((n, i) =>
        n > 0 ? (
          <span key={i} style={{ display: "inline-flex", alignItems: "center" }}>
            {n}
            <ColorPip color={i} size={10} />
          </span>
        ) : null,
      )}
      {cost.jokers > 0 ? (
        <span style={{ display: "inline-flex", alignItems: "center", color: "#e8c848" }}>
          {cost.jokers}
          <ColorPip color={5} size={10} />
        </span>
      ) : null}
    </span>
  );
}

function actionLabel(
  d: DecodedAction,
  cards: CardData[],
  nobles: NobleData[],
  grid: (number | null)[][],
  reserved: (number | null)[],
  noblesOnBoard: (number | null)[],
  humanPlayer: PlayerSnapshot,
  gemPool: number[],
): React.ReactNode {
  switch (d.kind) {
    case "take3": {
      // Engine edge-case: when fewer than 3 piles have tokens, take3 combos
      // that include the available colors are still legal but you only
      // receive tokens for piles with > 0 supply. Filter the display so the
      // button shows what you will actually take.
      const taken = (d.colors ?? []).filter((c) => (gemPool[c] ?? 0) > 0);
      const skipped = (d.colors ?? []).filter((c) => (gemPool[c] ?? 0) <= 0);
      return (
        <span style={{ display: "flex", alignItems: "center", gap: 2 }}>
          {taken.map((c, i) => <ColorPip key={`t${i}`} color={c} />)}
          {skipped.length > 0 && (
            <span
              style={{ color: "#7a94b8", fontSize: 10, marginLeft: 2 }}
              title={`combo would also include ${skipped
                .map((c) => COLOR_LETTERS[c])
                .join(", ")} but those piles are empty`}
            >
              (only {taken.length})
            </span>
          )}
        </span>
      );
    }
    case "take2":
      return (
        <span style={{ display: "flex", alignItems: "center", gap: 2 }}>
          <ColorPip color={d.color!} />
          <ColorPip color={d.color!} />
        </span>
      );
    case "reserve_grid": {
      const cardId = grid[d.tier!]?.[d.slot!];
      const card = cardId != null ? cards[cardId] : null;
      return (
        <span style={{ fontSize: 12 }}>
          {TIER_LABELS[d.tier!]} slot {d.slot! + 1}
          {card ? ` (${card.points}pt, +`: ""}
          {card ? <ColorPip color={card.bonus} size={10} /> : null}
          {card ? ")" : ""}
        </span>
      );
    }
    case "reserve_blind":
      return <span style={{ fontSize: 12 }}>{TIER_LABELS[d.tier!]} deck</span>;
    case "buy_grid": {
      const cardId = grid[d.tier!]?.[d.slot!];
      const card = cardId != null ? cards[cardId] : null;
      if (!card) return <span>{TIER_LABELS[d.tier!]} slot {d.slot! + 1}</span>;
      return (
        <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}>
          <span style={{ minWidth: 38 }}>
            {TIER_LABELS[d.tier!]}.{d.slot! + 1}
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 2 }}>
            <ColorPip color={card.bonus} size={12} />
            <span style={{ color: "#e8c848" }}>{card.points}pt</span>
          </span>
          <span style={{ color: "#7a94b8" }}>pay:</span>
          <CostBreakdown card={card} player={humanPlayer} />
        </span>
      );
    }
    case "buy_reserved": {
      const cardId = reserved[d.rslot!];
      const card = cardId != null ? cards[cardId] : null;
      if (!card) return <span>reserved {d.rslot! + 1}</span>;
      return (
        <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}>
          <span style={{ minWidth: 60 }}>reserved {d.rslot! + 1}</span>
          <ColorPip color={card.bonus} size={12} />
          <span style={{ color: "#e8c848" }}>{card.points}pt</span>
          <span style={{ color: "#7a94b8" }}>pay:</span>
          <CostBreakdown card={card} player={humanPlayer} />
        </span>
      );
    }
    case "pass":
      return <span>Pass</span>;
    case "discard": {
      const tok = d.token!;
      const letter = COLOR_LETTERS[tok];
      return (
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          discard {letter}
          {tok < 5 && <ColorPip color={tok} size={12} />}
        </span>
      );
    }
    case "pick_noble": {
      const nid = noblesOnBoard[d.slot!];
      const n = nid != null ? nobles[nid] : null;
      return <span style={{ fontSize: 12 }}>{n?.name ?? `noble ${d.slot! + 1}`}</span>;
    }
    default:
      return <span>action {d.action}</span>;
  }
}

// Returns true if a decoded token-taking action matches a token selection.
// sel is an ordered list of color indices selected so far (length 1 or 2).
function matchesTokenSelection(d: DecodedAction, sel: number[]): boolean {
  if (sel.length === 0) return true;
  if (d.kind === "take3") {
    return sel.every((c) => (d.colors ?? []).includes(c));
  }
  if (d.kind === "take2") {
    if (sel.length === 1) return d.color === sel[0];
    // Two same colors → take2 of that color
    if (sel.length === 2) return sel[0] === sel[1] && d.color === sel[0];
  }
  return false;
}

export function LegalActionsPanel({
  legalActions,
  cards,
  nobles,
  grid,
  reservedIds,
  noblesOnBoard,
  humanPlayer,
  gemPool,
  busy,
  onAction,
}: LegalActionsPanelProps) {
  const [selectedTokens, setSelectedTokens] = useState<number[]>([]);

  // Wrap onAction to always clear token selection after any action is taken.
  function handleAction(a: number) {
    setSelectedTokens([]);
    onAction(a);
  }

  const grouped: Record<string, DecodedAction[]> = {};
  for (const a of legalActions) {
    const d = decodeAction(a);
    (grouped[d.kind] = grouped[d.kind] ?? []).push(d);
  }
  // Dedupe take3 actions whose combo resolves to the same effective set of
  // taken colors after filtering empty piles. The engine considers all such
  // combos legal in the edge case where fewer than 3 piles have tokens, but
  // they are functionally equivalent moves for the player. We keep the
  // first action int seen for each effective set.
  if (grouped["take3"]) {
    const seen = new Set<string>();
    grouped["take3"] = grouped["take3"].filter((d) => {
      const taken = (d.colors ?? [])
        .filter((c) => (gemPool[c] ?? 0) > 0)
        .slice()
        .sort()
        .join(",");
      if (seen.has(taken)) return false;
      seen.add(taken);
      return true;
    });
  }

  // All available token-taking actions (after dedup) used for chip logic.
  const tokenActions: DecodedAction[] = [
    ...(grouped["take3"] ?? []),
    ...(grouped["take2"] ?? []),
  ];

  // Whether a token chip (color c) can be clicked given the current selection.
  function canClickToken(c: number): boolean {
    // Already selected: clicking deselects or auto-fires — always allowed.
    if (selectedTokens.includes(c)) return true;
    // No tokens of this color available in pool.
    if ((gemPool[c] ?? 0) === 0) return false;
    // Check if adding c leads to at least one matching legal token action.
    const newSel = [...selectedTokens, c];
    return tokenActions.some((d) => matchesTokenSelection(d, newSel));
  }

  function handleTokenClick(c: number) {
    if (selectedTokens.includes(c)) {
      if (selectedTokens.length === 1) {
        // Same token clicked twice → auto-fire take2 if legal, else deselect.
        const action = tokenActions.find(
          (d) => d.kind === "take2" && d.color === c,
        );
        if (action) {
          handleAction(action.action);
        } else {
          setSelectedTokens([]);
        }
      } else {
        // Remove this color from a multi-token selection.
        setSelectedTokens((prev) => prev.filter((x) => x !== c));
      }
      return;
    }

    const newSel = [...selectedTokens, c];

    if (newSel.length === 3) {
      // Three colors selected → auto-fire the matching take3 action.
      const [x, y, z] = newSel;
      const action = tokenActions.find(
        (d) =>
          d.kind === "take3" &&
          (d.colors ?? []).includes(x) &&
          (d.colors ?? []).includes(y) &&
          (d.colors ?? []).includes(z),
      );
      if (action) handleAction(action.action);
      // Clear selection regardless (if somehow not found, reset state).
      setSelectedTokens([]);
      return;
    }

    setSelectedTokens(newSel);
  }

  // Apply token selection filter to take3/take2 groups; all other kinds unaffected.
  const displayGrouped: Record<string, DecodedAction[]> =
    selectedTokens.length === 0
      ? grouped
      : {
          ...grouped,
          take3: (grouped["take3"] ?? []).filter((d) =>
            matchesTokenSelection(d, selectedTokens),
          ),
          take2: (grouped["take2"] ?? []).filter((d) =>
            matchesTokenSelection(d, selectedTokens),
          ),
        };

  const NON_GOLD_COLORS = [0, 1, 2, 3, 4];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        padding: 12,
        overflowY: "auto",
      }}
    >
      {/* Token filter chips */}
      <div>
        <div style={{ fontSize: 11, color: "#7a94b8", marginBottom: 6 }}>
          Tokens
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {NON_GOLD_COLORS.map((c) => {
            const isSelected = selectedTokens.includes(c);
            const isDisabled = busy || !canClickToken(c);
            const count = gemPool[c] ?? 0;
            return (
              <button
                key={c}
                disabled={isDisabled}
                onClick={() => handleTokenClick(c)}
                title={`${COLOR_LETTERS[c]}: ${count} in pool`}
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: "50%",
                  background: isDisabled ? "#1a2535" : GEM_COLORS[c],
                  border: isSelected
                    ? "2.5px solid #e8c848"
                    : "2.5px solid transparent",
                  boxShadow: isSelected
                    ? "0 0 0 1.5px #e8c848, 0 0 6px rgba(232,200,72,0.5)"
                    : "none",
                  cursor: isDisabled ? "not-allowed" : "pointer",
                  opacity: isDisabled ? 0.3 : 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexDirection: "column",
                  gap: 1,
                  padding: 0,
                  transition: "opacity 0.15s, box-shadow 0.15s",
                }}
              >
                <GemIcon
                  colorIdx={c}
                  size={14}
                  fill={isDisabled ? "#4a6a8a" : (GEM_TEXT_COLORS[c] ?? "#fff")}
                />
                <span
                  style={{
                    fontSize: 9,
                    lineHeight: 1,
                    color: isDisabled ? "#4a6a8a" : (GEM_TEXT_COLORS[c] ?? "#fff"),
                    fontWeight: "bold",
                  }}
                >
                  {count}
                </span>
              </button>
            );
          })}
          {selectedTokens.length > 0 && (
            <button
              onClick={() => setSelectedTokens([])}
              style={{
                background: "none",
                border: "1px solid #4a6a8a",
                borderRadius: 4,
                color: "#7a94b8",
                cursor: "pointer",
                fontSize: 11,
                padding: "3px 7px",
                marginLeft: 2,
              }}
            >
              clear
            </button>
          )}
        </div>
      </div>

      <div
        style={{
          fontSize: 13,
          fontWeight: "bold",
          color: "#e8c848",
        }}
      >
        Your move
      </div>
      {KIND_ORDER.map((kind) => {
        const list = displayGrouped[kind];
        if (!list || list.length === 0) return null;
        return (
          <div key={kind}>
            <div style={{ fontSize: 11, color: "#7a94b8", marginBottom: 4 }}>
              {KIND_TITLES[kind] ?? kind} ({list.length})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {list.map((d) => (
                <button
                  key={d.action}
                  disabled={busy}
                  onClick={() => handleAction(d.action)}
                  title={`action ${d.action}`}
                  style={{
                    background: busy ? "#1a2a4a" : "#0f3460",
                    border: "1px solid #2a4a7f",
                    color: "#e0e6f0",
                    borderRadius: 4,
                    padding: "4px 6px",
                    cursor: busy ? "not-allowed" : "pointer",
                    fontSize: 12,
                  }}
                >
                  {actionLabel(
                    d,
                    cards,
                    nobles,
                    grid,
                    reservedIds,
                    noblesOnBoard,
                    humanPlayer,
                    gemPool,
                  )}
                </button>
              ))}
            </div>
          </div>
        );
      })}
      {legalActions.length === 0 && (
        <div style={{ color: "#7a94b8", fontSize: 12 }}>
          No legal actions (game is paused).
        </div>
      )}
    </div>
  );
}
