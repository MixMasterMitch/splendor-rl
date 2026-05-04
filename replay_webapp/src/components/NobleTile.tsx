import { NobleData, GEM_COLORS, GEM_TEXT_COLORS } from "../types";
import { GemIcon } from "./GemIcon";

interface NobleTileProps {
  noble: NobleData | null;
  size?: number;
}

export function NobleTile({ noble, size = 84 }: NobleTileProps) {
  if (!noble) {
    return (
      <div
        style={{
          width: size,
          minHeight: size,
          borderRadius: 6,
          border: "1px dashed #2a4a7f",
          flexShrink: 0,
        }}
      />
    );
  }

  const nonZeroReqs = noble.requirement
    .map((c, i) => ({ count: c, color: i }))
    .filter((x) => x.count > 0);

  // Split requirements into two columns for compact display.
  const col1 = nonZeroReqs.slice(0, Math.ceil(nonZeroReqs.length / 2));
  const col2 = nonZeroReqs.slice(Math.ceil(nonZeroReqs.length / 2));

  const ReqItem = ({ count, color }: { count: number; color: number }) => (
    <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
      <div
        style={{
          width: 15,
          height: 15,
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
  );

  return (
    <div
      style={{
        width: size,
        borderRadius: 6,
        border: "2px solid #9a4a7c",
        background: "#2a1a3e",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        flexShrink: 0,
      }}
    >
      {/* Header: name left, PV right */}
      <div
        style={{
          background: "#9a4a7c",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          padding: "3px 5px",
          gap: 4,
        }}
      >
        <span
          style={{
            fontSize: 9,
            color: "#fff",
            lineHeight: 1.25,
            wordBreak: "break-word",
            flex: 1,
          }}
        >
          {noble.name}
        </span>
        <span style={{ fontWeight: "bold", fontSize: 13, color: "#fff", lineHeight: 1, flexShrink: 0 }}>
          {noble.points}
        </span>
      </div>

      {/* Body: requirements in two columns */}
      <div
        style={{
          padding: "5px 5px",
          display: "flex",
          gap: 6,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {col1.map(({ count, color }) => (
            <ReqItem key={color} count={count} color={color} />
          ))}
        </div>
        {col2.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {col2.map(({ count, color }) => (
              <ReqItem key={color} count={count} color={color} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
