import { useState, useCallback, useEffect } from "react";
import { CardData, GameSnapshot, NobleData, PlayerInfo, GEM_COLORS, GEM_TEXT_COLORS } from "../types";
import { CardTile } from "./CardTile";
import { GemIcon } from "./GemIcon";
import { NobleTile } from "./NobleTile";
import { PlayerPanel } from "./PlayerPanel";
import { DecodedAction, decodeAction } from "../play_actions";

interface BoardProps {
  snapshot: GameSnapshot;
  cardDb: CardData[];
  nobleDb: NobleData[];
  playerInfos: PlayerInfo[];
  winners: number[];
  highlightedAction?: {
    kind: string;
    tier?: number;
    slot?: number;
    colors?: number[];
    color?: number;
  };
  // Interactive mode props (only provided during human's turn)
  interactive?: boolean;
  legalActions?: number[];
  humanSeat?: number;
  busy?: boolean;
  onAction?: (action: number) => void;
  // Show "AI is thinking" indicator in the status area
  aiThinking?: boolean;
}

const TIER_NAMES = ["Level III (top)", "Level II", "Level I (bottom)"];

// Returns true if a decoded token-taking action matches a token selection.
function matchesTokenSelection(d: DecodedAction, sel: number[]): boolean {
  if (sel.length === 0) return true;
  if (d.kind === "take3") {
    return sel.every((c) => (d.colors ?? []).includes(c));
  }
  if (d.kind === "take2") {
    if (sel.length === 1) return d.color === sel[0];
    if (sel.length === 2) return sel[0] === sel[1] && d.color === sel[0];
  }
  return false;
}

export function Board({ snapshot, cardDb, nobleDb, playerInfos, winners, highlightedAction, interactive = false, legalActions = [], humanSeat = 0, busy = false, onAction, aiThinking = false }: BoardProps) {
  // Grid is stored as tier 0=L1, 1=L2, 2=L3. Display tier 2 on top.
  const displayTiers = [2, 1, 0];

  // Token selection state (only used in interactive mode)
  const [selectedTokens, setSelectedTokens] = useState<number[]>([]);
  // Reserve mode: user clicked gold token, now selecting a card to reserve
  const [reserveMode, setReserveMode] = useState(false);
  // Card cost display mode
  const [costMode, setCostMode] = useState<"true" | "effective" | "remaining">("true");

  // Reset selection when legal actions change (new turn)
  useEffect(() => {
    setSelectedTokens([]);
    setReserveMode(false);
  }, [legalActions]);

  // Decode all legal actions
  const decoded: DecodedAction[] = interactive ? legalActions.map(decodeAction) : [];

  // Group by kind
  const byKind = (kind: string) => decoded.filter((d) => d.kind === kind);

  const tokenActions = [...byKind("take3"), ...byKind("take2")];
  const buyGridActions = byKind("buy_grid");
  const buyReservedActions = byKind("buy_reserved");
  const reserveGridActions = byKind("reserve_grid");
  const reserveBlindActions = byKind("reserve_blind");
  const pickNobleActions = byKind("pick_noble");
  const discardActions = byKind("discard");
  const passActions = byKind("pass");

  // Determine which grid cards are affordable (have a buy_grid action)
  const affordableGrid = new Set<string>();
  for (const d of buyGridActions) {
    affordableGrid.add(`${d.tier}-${d.slot}`);
  }

  // Determine which reserved cards are affordable (have a buy_reserved action)
  const affordableReserved = new Set<number>();
  for (const d of buyReservedActions) {
    affordableReserved.add(d.rslot!);
  }

  // Determine which grid cards can be reserved
  const reservableGrid = new Set<string>();
  for (const d of reserveGridActions) {
    reservableGrid.add(`${d.tier}-${d.slot}`);
  }

  // Determine which tiers can be blind-reserved
  const reservableBlindTiers = new Set<number>();
  for (const d of reserveBlindActions) {
    reservableBlindTiers.add(d.tier!);
  }

  // Determine which nobles can be picked
  const pickableNobles = new Set<number>();
  for (const d of pickNobleActions) {
    pickableNobles.add(d.slot!);
  }

  // Whether a token chip (color c) can be clicked given the current selection.
  const canClickToken = useCallback((c: number): boolean => {
    if (!interactive || busy) return false;
    if (reserveMode) return false;
    // Gold token click → enter reserve mode
    if (c === 5) return reserveGridActions.length > 0 || reserveBlindActions.length > 0;
    if (selectedTokens.includes(c)) return true;
    if ((snapshot.gem_pool[c] ?? 0) === 0) return false;
    const newSel = [...selectedTokens, c];
    return tokenActions.some((d) => matchesTokenSelection(d, newSel));
  }, [interactive, busy, reserveMode, selectedTokens, snapshot.gem_pool, tokenActions, reserveGridActions, reserveBlindActions]);

  function fireAction(a: number) {
    setSelectedTokens([]);
    setReserveMode(false);
    onAction?.(a);
  }

  function handleTokenClick(c: number) {
    if (!interactive || busy) return;

    // Gold token → enter reserve mode
    if (c === 5) {
      if (reserveGridActions.length > 0 || reserveBlindActions.length > 0) {
        setReserveMode(true);
        setSelectedTokens([]);
      }
      return;
    }

    if (reserveMode) return; // Can't pick gems in reserve mode

    if (selectedTokens.includes(c)) {
      if (selectedTokens.length === 1) {
        // Same token clicked twice → auto-fire take2 if legal, else deselect.
        const action = tokenActions.find((d) => d.kind === "take2" && d.color === c);
        if (action) {
          fireAction(action.action);
        } else {
          setSelectedTokens([]);
        }
      } else {
        setSelectedTokens((prev) => prev.filter((x) => x !== c));
      }
      return;
    }

    const newSel = [...selectedTokens, c];

    if (newSel.length === 3) {
      const [x, y, z] = newSel;
      const action = tokenActions.find(
        (d) =>
          d.kind === "take3" &&
          (d.colors ?? []).includes(x) &&
          (d.colors ?? []).includes(y) &&
          (d.colors ?? []).includes(z),
      );
      if (action) fireAction(action.action);
      setSelectedTokens([]);
      return;
    }

    setSelectedTokens(newSel);
  }

  function handleCardClick(tier: number, slot: number) {
    if (!interactive || busy) return;

    if (reserveMode) {
      // Reserve this card
      const action = reserveGridActions.find((d) => d.tier === tier && d.slot === slot);
      if (action) fireAction(action.action);
      return;
    }

    // Buy this card
    const action = buyGridActions.find((d) => d.tier === tier && d.slot === slot);
    if (action) fireAction(action.action);
  }

  function handleReservedCardClick(rslot: number) {
    if (!interactive || busy) return;
    const action = buyReservedActions.find((d) => d.rslot === rslot);
    if (action) fireAction(action.action);
  }

  function handleDeckClick(tier: number) {
    if (!interactive || busy || !reserveMode) return;
    const action = reserveBlindActions.find((d) => d.tier === tier);
    if (action) fireAction(action.action);
  }

  function handleNobleClick(nobleSlot: number) {
    if (!interactive || busy) return;
    const action = pickNobleActions.find((d) => d.slot === nobleSlot);
    if (action) fireAction(action.action);
  }

  function handleDiscard(colorIdx: number) {
    if (!interactive || busy) return;
    const action = discardActions.find((d) => d.token === colorIdx);
    if (action) fireAction(action.action);
  }

  function handlePass() {
    if (!interactive || busy) return;
    if (passActions.length > 0) fireAction(passActions[0].action);
  }

  const isDiscardPhase = discardActions.length > 0;
  const isNoblePick = pickNobleActions.length > 0;

  // Compute display costs for a card based on the cost mode
  const humanPlayerSnapshot = snapshot.players[humanSeat];
  function getDisplayCosts(card: CardData, affordable: boolean = false): number[] | undefined {
    if (costMode === "true") return undefined; // use card's raw cost
    const bonuses = humanPlayerSnapshot?.bonuses ?? [0, 0, 0, 0, 0];
    const tokens = humanPlayerSnapshot?.tokens ?? [0, 0, 0, 0, 0, 0];
    if (costMode === "effective") {
      if (affordable) {
        // Show exact tokens spent including jokers explicitly
        const perColor = [0, 0, 0, 0, 0];
        let jokers = 0;
        for (let i = 0; i < 5; i++) {
          const needed = Math.max(0, card.cost[i] - (bonuses[i] ?? 0));
          const have = tokens[i] ?? 0;
          const fromTokens = Math.min(needed, have);
          perColor[i] = fromTokens;
          jokers += needed - fromTokens;
        }
        return [...perColor, jokers]; // length 6: [W,B,G,R,K,Gold]
      }
      // Not affordable: just show cost minus bonuses
      return card.cost.map((c, i) => Math.max(0, c - (bonuses[i] ?? 0)));
    }
    // "remaining": tokens still needed beyond what the player has
    if (affordable) {
      // Card is affordable — no remaining tokens needed
      return [0, 0, 0, 0, 0];
    }
    return card.cost.map((c, i) => {
      const effective = Math.max(0, c - (bonuses[i] ?? 0));
      return Math.max(0, effective - (tokens[i] ?? 0));
    });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Interactive status bar — always rendered with fixed height to prevent layout shift */}
      {interactive && !busy && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", minHeight: 24 }}>
          {isDiscardPhase ? (
            <span style={{ fontSize: 12, color: "#e8c848", fontWeight: "bold" }}>
              Discard a token (over 10) — click a token in your panel below
            </span>
          ) : isNoblePick ? (
            <span style={{ fontSize: 12, color: "#e8c848", fontWeight: "bold" }}>
              Pick a noble — click one of the highlighted nobles
            </span>
          ) : reserveMode ? (
            <span style={{ fontSize: 12, color: "#e8c848", fontWeight: "bold" }}>
              Reserve mode — click a card or deck to reserve
            </span>
          ) : selectedTokens.length > 0 ? (
            <span style={{ fontSize: 12, color: "#e8c848", fontWeight: "bold" }}>
              Selecting tokens ({selectedTokens.length}/3)...
            </span>
          ) : (
            <span style={{ fontSize: 12, color: "#e8c848", fontWeight: "bold" }}>
              Your turn
            </span>
          )}
          {(selectedTokens.length > 0 || reserveMode) && (
            <button
              onClick={() => { setSelectedTokens([]); setReserveMode(false); }}
              style={{
                background: "none",
                border: "1px solid #4a6a8a",
                borderRadius: 4,
                color: "#7a94b8",
                cursor: "pointer",
                fontSize: 11,
                padding: "3px 8px",
              }}
            >
              Clear
            </button>
          )}
          {passActions.length > 0 && !reserveMode && selectedTokens.length === 0 && (
            <button
              onClick={handlePass}
              disabled={busy}
              style={{
                background: "#0f3460",
                border: "1px solid #2a4a7f",
                borderRadius: 4,
                color: "#e0e6f0",
                cursor: busy ? "not-allowed" : "pointer",
                fontSize: 11,
                padding: "3px 8px",
              }}
            >
              Pass
            </button>
          )}
        </div>
      )}

      {/* AI thinking status — shown in the same area as "Your turn" to avoid layout shift */}
      {((!interactive && aiThinking) || (interactive && busy)) && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, minHeight: 24 }}>
          <span className="thinking-dots" aria-hidden="true">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
          </span>
          <span style={{ fontSize: 12, color: "#e8c848", fontWeight: "bold" }}>
            AI is thinking…
          </span>
          <style>{`
            .thinking-dots {
              display: inline-flex;
              align-items: center;
              gap: 3px;
            }
            .thinking-dot {
              width: 5px;
              height: 5px;
              border-radius: 50%;
              background: #e8c848;
              animation: thinking-pulse 1.4s ease-in-out infinite;
            }
            .thinking-dot:nth-child(2) {
              animation-delay: 0.2s;
            }
            .thinking-dot:nth-child(3) {
              animation-delay: 0.4s;
            }
            @keyframes thinking-pulse {
              0%, 100% { opacity: 0.3; transform: scale(0.8); }
              50% { opacity: 1; transform: scale(1.2); }
            }
          `}</style>
        </div>
      )}

      {/* Card cost display mode toggle */}
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span style={{ fontSize: 11, color: "#7a94b8", marginRight: 4 }}>Costs:</span>
        {([
          { key: "true" as const, label: "True" },
          { key: "effective" as const, label: "Effective" },
          { key: "remaining" as const, label: "Remaining" },
        ]).map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setCostMode(key)}
            style={{
              background: costMode === key ? "#0f3460" : "transparent",
              border: costMode === key ? "1px solid #e8c848" : "1px solid #2a4a7f",
              borderRadius: 4,
              color: costMode === key ? "#e8c848" : "#7a94b8",
              cursor: "pointer",
              fontSize: 11,
              padding: "2px 8px",
              fontWeight: costMode === key ? "bold" : "normal",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Nobles row */}
      <div>
        <div style={{ fontSize: 11, color: "#7a94b8", marginBottom: 4 }}>Nobles</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {snapshot.nobles.map((nid, ns) => {
            const noble = nid != null ? (nobleDb[nid] ?? null) : null;
            const canPick = interactive && pickableNobles.has(ns);
            const nobleDisplayReqs = noble && costMode === "remaining"
              ? noble.requirement.map((req, i) => Math.max(0, req - (humanPlayerSnapshot?.bonuses[i] ?? 0)))
              : undefined;
            return (
              <NobleTile
                key={ns}
                noble={noble}
                highlight={canPick}
                onClick={canPick ? () => handleNobleClick(ns) : undefined}
                displayRequirements={nobleDisplayReqs}
              />
            );
          })}
        </div>
      </div>

      {/* Card grid */}
      {displayTiers.map((tier) => {
        const deckCount = snapshot.deck_counts[tier] ?? 0;
        const canBlindReserve = interactive && reserveMode && reservableBlindTiers.has(tier) && deckCount > 0;
        return (
          <div key={tier}>
            <div
              style={{
                fontSize: 11,
                color: "#7a94b8",
                marginBottom: 4,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              {TIER_NAMES[2 - tier]}
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {/* Deck tile (face-down card) */}
              <div
                onClick={canBlindReserve ? () => handleDeckClick(tier) : undefined}
                style={{
                  width: 72,
                  height: 96,
                  borderRadius: 6,
                  border: `2px solid ${canBlindReserve ? "#e8c848" : "rgba(255,255,255,0.1)"}`,
                  background: deckCount > 0
                    ? "repeating-linear-gradient(45deg,#0f1f3a,#0f1f3a 4px,#16213e 4px,#16213e 8px)"
                    : "transparent",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  cursor: canBlindReserve ? "pointer" : undefined,
                  boxShadow: canBlindReserve ? "0 0 8px #e8c848" : undefined,
                  opacity: deckCount === 0 ? 0.3 : 1,
                }}
                title={deckCount > 0 ? `Deck: ${deckCount} cards${canBlindReserve ? " (click to blind reserve)" : ""}` : "Empty deck"}
              >
                <span style={{ fontSize: 11, color: "#7a94b8", fontWeight: "bold" }}>
                  {deckCount}
                </span>
              </div>
              {/* Grid cards */}
              {(snapshot.grid[tier] ?? []).map((cardId, slot) => {
                const card = cardId != null ? (cardDb[cardId] ?? null) : null;
                const isHighlighted =
                  (highlightedAction?.kind === "buy_grid" &&
                    highlightedAction.tier === tier &&
                    highlightedAction.slot === slot) ||
                  (highlightedAction?.kind === "reserve_grid" &&
                    highlightedAction.tier === tier &&
                    highlightedAction.slot === slot);
                const isAffordable = interactive && affordableGrid.has(`${tier}-${slot}`) && !reserveMode;
                const canReserve = interactive && reserveMode && reservableGrid.has(`${tier}-${slot}`);
                const isClickable = isAffordable || canReserve;
                return (
                  <CardTile
                    key={slot}
                    card={card}
                    highlight={isHighlighted || canReserve}
                    affordable={isAffordable && !isHighlighted}
                    onClick={isClickable ? () => handleCardClick(tier, slot) : undefined}
                    displayCosts={card ? getDisplayCosts(card, isAffordable) : undefined}
                  />
                );
              })}
            </div>
          </div>
        );
      })}

      {/* Gem pool */}
      <div>
        <div style={{ fontSize: 11, color: "#7a94b8", marginBottom: 4 }}>Gem Pool</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          {snapshot.gem_pool.map((count, i) => {
            const isHighlighted =
              (highlightedAction?.kind === "take3" && (highlightedAction.colors ?? []).includes(i)) ||
              (highlightedAction?.kind === "take2" && highlightedAction.color === i);
            const isSelected = interactive && selectedTokens.includes(i);
            const isGoldReserve = interactive && i === 5 && (reserveGridActions.length > 0 || reserveBlindActions.length > 0);
            const isGoldActive = interactive && i === 5 && reserveMode;
            const clickable = interactive && (canClickToken(i) || isGoldReserve);
            const isDisabled = interactive && !clickable && !isSelected;
            return (
              <div
                key={i}
                onClick={clickable ? () => handleTokenClick(i) : undefined}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 2,
                  cursor: clickable ? "pointer" : undefined,
                }}
              >
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    background: GEM_COLORS[i] ?? "#888",
                    border: isSelected || isGoldActive
                      ? "2.5px solid #e8c848"
                      : isHighlighted
                        ? "2px solid #e8c848"
                        : "2px solid rgba(255,255,255,0.2)",
                    boxShadow: isSelected || isGoldActive
                      ? "0 0 0 1.5px #e8c848, 0 0 6px rgba(232,200,72,0.5)"
                      : isHighlighted
                        ? "0 0 8px #e8c848"
                        : undefined,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: count === 0 && !isSelected ? 0.3 : interactive && isDisabled ? 0.4 : 1,
                    transition: "opacity 0.15s, box-shadow 0.15s",
                  }}
                >
                  <GemIcon colorIdx={i} size={18} fill={GEM_TEXT_COLORS[i] ?? "#fff"} />
                </div>
                <span style={{ fontSize: 11, color: count === 0 ? "#7a94b8" : "#e0e6f0" }}>
                  {count}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Player panels — human first, then others in seat order */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {(() => {
          const numPlayers = snapshot.players.length;
          const order: number[] = [];
          for (let offset = 0; offset < numPlayers; offset++) {
            order.push((humanSeat + offset) % numPlayers);
          }
          return order.map((i) => {
            const player = snapshot.players[i];
            const isHuman = i === humanSeat;
            return (
              <PlayerPanel
                key={i}
                playerIdx={i}
                info={playerInfos[i] ?? { name: `P${i}`, kind: "unknown" }}
                snapshot={player}
                isActive={true}
                isCurrentPlayer={snapshot.current_player === i}
                currentPhase={snapshot.phase}
                cardDb={cardDb}
                nobleDb={nobleDb}
                isWinner={winners.includes(i)}
                affordableReservedSlots={isHuman && interactive ? affordableReserved : undefined}
                onReservedClick={isHuman && interactive ? handleReservedCardClick : undefined}
                discardableTokens={isHuman && interactive && isDiscardPhase ? new Set(discardActions.map((d) => d.token!)) : undefined}
                onTokenDiscard={isHuman && interactive && isDiscardPhase ? handleDiscard : undefined}
                getDisplayCosts={isHuman ? getDisplayCosts : undefined}
              />
            );
          });
        })()}
      </div>
    </div>
  );
}
