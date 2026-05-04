import { GEM_COLORS, GEM_TEXT_COLORS, COLOR_NAMES } from "../types";

interface TokenProps {
  colorIdx: number;  // 0-5
  count: number;
  size?: number;
}

export function Token({ colorIdx, count, size = 28 }: TokenProps) {
  const bg = GEM_COLORS[colorIdx] ?? "#888";
  const fg = GEM_TEXT_COLORS[colorIdx] ?? "#fff";
  const label = (COLOR_NAMES[colorIdx] ?? "?")[0];
  return (
    <span
      title={`${COLOR_NAMES[colorIdx] ?? "?"}: ${count}`}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: size,
        height: size,
        borderRadius: "50%",
        background: bg,
        color: fg,
        fontWeight: "bold",
        fontSize: size * 0.4,
        border: "2px solid rgba(255,255,255,0.15)",
        position: "relative",
        flexShrink: 0,
      }}
    >
      {label}
      {count > 0 && (
        <span
          style={{
            position: "absolute",
            bottom: -4,
            right: -4,
            background: "#1a1a2e",
            color: "#e0e6f0",
            borderRadius: "50%",
            width: size * 0.55,
            height: size * 0.55,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: size * 0.32,
            fontWeight: "bold",
            border: "1px solid rgba(255,255,255,0.2)",
          }}
        >
          {count}
        </span>
      )}
    </span>
  );
}
