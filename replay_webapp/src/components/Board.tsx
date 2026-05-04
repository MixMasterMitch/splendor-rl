import { CardData, GameSnapshot, NobleData, PlayerInfo, GEM_COLORS, GEM_TEXT_COLORS } from "../types";
import { CardTile } from "./CardTile";
import { GemIcon } from "./GemIcon";
import { NobleTile } from "./NobleTile";
import { PlayerPanel } from "./PlayerPanel";

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
  };
}

const TIER_NAMES = ["Level III (top)", "Level II", "Level I (bottom)"];

export function Board({ snapshot, cardDb, nobleDb, playerInfos, winners, highlightedAction }: BoardProps) {
  // Grid is stored as tier 0=L1, 1=L2, 2=L3. Display tier 2 on top.
  const displayTiers = [2, 1, 0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Nobles row */}
      <div>
        <div style={{ fontSize: 11, color: "#7a94b8", marginBottom: 4 }}>Nobles</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {snapshot.nobles.map((nid, ns) => {
            const noble = nid != null ? (nobleDb[nid] ?? null) : null;
            return <NobleTile key={ns} noble={noble} />;
          })}
        </div>
      </div>

      {/* Card grid */}
      {displayTiers.map((tier) => {
        const deckCount = snapshot.deck_counts[tier] ?? 0;
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
              <span
                style={{
                  background: "#0f3460",
                  borderRadius: 4,
                  padding: "1px 6px",
                  fontSize: 10,
                }}
              >
                deck: {deckCount}
              </span>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {(snapshot.grid[tier] ?? []).map((cardId, slot) => {
                const card = cardId != null ? (cardDb[cardId] ?? null) : null;
                const isHighlighted =
                  highlightedAction?.kind === "buy_grid" &&
                  highlightedAction.tier === tier &&
                  highlightedAction.slot === slot;
                const isReserveHighlighted =
                  highlightedAction?.kind === "reserve_grid" &&
                  highlightedAction.tier === tier &&
                  highlightedAction.slot === slot;
                return (
                  <CardTile
                    key={slot}
                    card={card}
                    highlight={isHighlighted || isReserveHighlighted}
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
          {snapshot.gem_pool.map((count, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
              }}
            >
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: "50%",
                  background: GEM_COLORS[i] ?? "#888",
                  border: "2px solid rgba(255,255,255,0.2)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  opacity: count === 0 ? 0.3 : 1,
                }}
              >
                <GemIcon colorIdx={i} size={18} fill={GEM_TEXT_COLORS[i] ?? "#fff"} />
              </div>
              <span style={{ fontSize: 11, color: count === 0 ? "#7a94b8" : "#e0e6f0" }}>
                {count}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Player panels */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {snapshot.players.map((player, i) => (
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
          />
        ))}
      </div>
    </div>
  );
}
