import PlayApp from "./PlayApp";

export default function App() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          background: "#0f3460",
          borderBottom: "1px solid #2a4a7f",
          padding: "8px 16px",
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontWeight: "bold",
            fontSize: 16,
            color: "#e8c848",
            whiteSpace: "nowrap",
          }}
        >
          Splendor RL
        </span>
      </div>
      <div style={{ flex: 1, overflow: "hidden" }}>
        <PlayApp />
      </div>
    </div>
  );
}
