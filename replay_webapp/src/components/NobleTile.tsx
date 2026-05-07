import { NobleData, GEM_COLORS, GEM_TEXT_COLORS } from "../types";
import { GemIcon } from "./GemIcon";

interface NobleTileProps {
  noble: NobleData | null;
  highlight?: boolean;
  onClick?: () => void;
  displayRequirements?: number[]; // optional override for displayed requirements (length 5)
}

export function NobleTile({ noble, highlight = false, onClick, displayRequirements }: NobleTileProps) {
  if (!noble) {
    return (
      <div
        style={{
          width: 120,
          minHeight: 48,
          borderRadius: 6,
          border: "1px dashed #2a4a7f",
          flexShrink: 0,
        }}
      />
    );
  }

  const reqs = displayRequirements ?? noble.requirement;
  const nonZeroReqs = reqs
    .map((c, i) => ({ count: c, color: i }))
    .filter((x) => x.count > 0);

  return (
    <div
      onClick={onClick}
      style={{
        width: 120,
        borderRadius: 6,
        border: `2px solid ${highlight ? "#e8c848" : "#9a4a7c"}`,
        background: "#2a1a3e",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        flexShrink: 0,
        cursor: onClick ? "pointer" : undefined,
        boxShadow: highlight ? "0 0 8px #e8c848" : undefined,
      }}
    >
      {/* Header: name left, PV right */}
      <div
        style={{
          background: "#9a4a7c",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "3px 6px",
          gap: 4,
        }}
      >
        <span
          style={{
            fontSize: 10,
            color: "#fff",
            lineHeight: 1.2,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            flex: 1,
          }}
        >
          {noble.name}
        </span>
        <span style={{ fontWeight: "bold", fontSize: 13, color: "#fff", lineHeight: 1, flexShrink: 0 }}>
          {noble.points}
        </span>
      </div>

      {/* Body: requirements in a single row */}
      <div
        style={{
          padding: "4px 6px",
          display: "flex",
          gap: 6,
          alignItems: "center",
          flexWrap: "nowrap",
        }}
      >
        {nonZeroReqs.map(({ count, color }) => (
          <div key={color} style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <div
              style={{
                width: 14,
                height: 14,
                borderRadius: "50%",
                background: GEM_COLORS[color] ?? "#888",
                border: "1px solid rgba(255,255,255,0.2)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <GemIcon colorIdx={color} size={9} fill={GEM_TEXT_COLORS[color] ?? "#fff"} />
            </div>
            <span style={{ fontSize: 11, color: "#e0e6f0", fontWeight: "bold" }}>{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
