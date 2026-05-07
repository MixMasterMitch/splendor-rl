import { CardData, GEM_COLORS, GEM_TEXT_COLORS } from "../types";
import { GemIcon } from "./GemIcon";

interface CardTileProps {
  card: CardData | null;   // null = empty slot
  hidden?: boolean;        // true = face-down reserve (show contents dimmed to observer)
  small?: boolean;
  highlight?: boolean;
  affordable?: boolean;    // true = card can be bought by the human (subtle emphasis)
  onClick?: () => void;    // clickable card (buy or reserve)
  displayCosts?: number[]; // optional override for displayed costs (length 5)
}

export function CardTile({ card, hidden = false, small = false, highlight = false, affordable = false, onClick, displayCosts }: CardTileProps) {
  const w = small ? 52 : 72;
  const h = small ? 70 : 96;

  if (!card) {
    if (hidden) {
      // Face-down placeholder: no card details visible, just a card-back tile
      return (
        <div
          onClick={onClick}
          style={{
            width: w,
            height: h,
            borderRadius: 6,
            border: `2px solid ${highlight ? "#e8c848" : "rgba(255,255,255,0.1)"}`,
            background:
              "repeating-linear-gradient(45deg,#0f1f3a,#0f1f3a 4px,#16213e 4px,#16213e 8px)",
            flexShrink: 0,
            position: "relative",
            cursor: onClick ? "pointer" : undefined,
            boxShadow: highlight ? "0 0 8px #e8c848" : undefined,
          }}
          title="Hidden reserved card"
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              pointerEvents: "none",
            }}
          >
            <svg
              width={small ? 28 : 36}
              height={small ? 28 : 36}
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              style={{ opacity: 0.45 }}
            >
              <path
                d="M1 12C1 12 5 5 12 5C19 5 23 12 23 12C23 12 19 19 12 19C5 19 1 12 1 12Z"
                stroke="#a8c8ff"
                strokeWidth="1.8"
                fill="none"
              />
              <circle cx="12" cy="12" r="3" stroke="#a8c8ff" strokeWidth="1.6" fill="none" />
              <line x1="3" y1="3" x2="21" y2="21" stroke="#a8c8ff" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </div>
        </div>
      );
    }
    return (
      <div
        onClick={onClick}
        style={{
          width: w,
          height: h,
          borderRadius: 6,
          border: `1px dashed ${highlight ? "#e8c848" : "#2a4a7f"}`,
          background: "transparent",
          flexShrink: 0,
          cursor: onClick ? "pointer" : undefined,
          boxShadow: highlight ? "0 0 8px #e8c848" : undefined,
        }}
      />
    );
  }

  const bonusColor = GEM_COLORS[card.bonus] ?? "#888";
  const bonusFg = GEM_TEXT_COLORS[card.bonus] ?? "#fff";
  const costsToShow = displayCosts ?? card.cost;
  const nonZeroCosts = costsToShow
    .map((c, i) => ({ count: c, color: i }))
    .filter((x) => x.count > 0);

  return (
    <div
      onClick={onClick}
      style={{
        width: w,
        height: h,
        borderRadius: 6,
        border: `2px solid ${highlight ? "#e8c848" : affordable ? "#4caf50" : "rgba(255,255,255,0.1)"}`,
        background: "#16213e",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        flexShrink: 0,
        position: "relative",
        boxShadow: highlight ? "0 0 8px #e8c848" : affordable ? "0 0 6px rgba(76,175,80,0.4)" : undefined,
        opacity: hidden ? 0.75 : 1,
        cursor: onClick ? "pointer" : undefined,
      }}
    >
      {/* Single header: fixed height, gem icon left, PV right */}
      <div
        style={{
          background: bonusColor,
          height: small ? 18 : 22,
          padding: "0 4px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexShrink: 0,
        }}
      >
        <GemIcon colorIdx={card.bonus} size={small ? 11 : 14} fill={bonusFg} />
        <span
          style={{
            fontWeight: "bold",
            fontSize: small ? 11 : 14,
            color: bonusFg,
            lineHeight: 1,
          }}
        >
          {card.points > 0 ? card.points : ""}
        </span>
      </div>

      {/* Cost grid */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          padding: "2px 3px",
          gap: 1,
        }}
      >
        {nonZeroCosts.map(({ count, color }) => (
          <div
            key={color}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 2,
            }}
          >
            <div
              style={{
                width: small ? 10 : 13,
                height: small ? 10 : 13,
                borderRadius: "50%",
                background: GEM_COLORS[color] ?? "#888",
                border: "1px solid rgba(255,255,255,0.2)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <GemIcon
                colorIdx={color}
                size={small ? 7 : 9}
                fill={GEM_TEXT_COLORS[color] ?? "#fff"}
              />
            </div>
            <span style={{ fontSize: small ? 9 : 11, color: "#e0e6f0" }}>
              {count}
            </span>
          </div>
        ))}
      </div>

      {/* Hidden overlay: eye-slash centered in the card body (below header) */}
      {hidden && (
        <div
          style={{
            position: "absolute",
            top: small ? 18 : 22,
            left: 0,
            right: 0,
            bottom: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            pointerEvents: "none",
          }}
          title="Hidden from other players"
        >
          <svg
            width={small ? 28 : 36}
            height={small ? 28 : 36}
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            style={{ opacity: 0.28 }}
          >
            <path
              d="M1 12C1 12 5 5 12 5C19 5 23 12 23 12C23 12 19 19 12 19C5 19 1 12 1 12Z"
              stroke="#a8c8ff"
              strokeWidth="1.8"
              fill="none"
            />
            <circle cx="12" cy="12" r="3" stroke="#a8c8ff" strokeWidth="1.6" fill="none" />
            <line x1="3" y1="3" x2="21" y2="21" stroke="#a8c8ff" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </div>
      )}
    </div>
  );
}
