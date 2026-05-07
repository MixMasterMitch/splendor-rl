/**
 * Loading spinner: a gold eight-pointed star (joker token) that rocks
 * back and forth ±45 degrees, with "Loading..." text beneath.
 */

interface LoadingSpinnerProps {
  size?: number;
  text?: string;
}

export function LoadingSpinner({ size = 40, text = "Loading..." }: LoadingSpinnerProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        padding: 24,
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 20 20"
        fill="none"
        style={{ animation: "rock 1.2s ease-in-out infinite" }}
      >
        <polygon
          points="10,1 11.8,7.5 18,6.5 13.2,11 16,17.5 10,14 4,17.5 6.8,11 2,6.5 8.2,7.5"
          fill="#e8c848"
          stroke="#c9ae3d"
          strokeWidth="0.5"
        />
      </svg>
      {text && (
        <span style={{ color: "#7a94b8", fontSize: 13, fontWeight: 500 }}>{text}</span>
      )}
      <style>{`
        @keyframes rock {
          0%, 100% { transform: rotate(-45deg); }
          50% { transform: rotate(45deg); }
        }
      `}</style>
    </div>
  );
}
