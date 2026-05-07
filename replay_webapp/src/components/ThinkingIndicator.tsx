import { useEffect, useState } from "react";

interface ThinkingIndicatorProps {
  startTime: number;
}

/**
 * Animated thinking indicator shown while the LLM agent is processing its move.
 * Displays pulsing dots, "AI is thinking…" text, and an elapsed time counter
 * updated every 100ms.
 */
export function ThinkingIndicator({ startTime }: ThinkingIndicatorProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setElapsed(Date.now() - startTime), 100);
    return () => clearInterval(interval);
  }, [startTime]);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 16px",
        background: "#0a1830",
        borderBottom: "1px solid #2a4a7f",
      }}
    >
      <span className="thinking-dots" aria-hidden="true">
        <span className="thinking-dot" />
        <span className="thinking-dot" />
        <span className="thinking-dot" />
      </span>
      <span style={{ color: "#e8c848", fontSize: 13, fontWeight: 500 }}>
        AI is thinking… ({(elapsed / 1000).toFixed(1)}s)
      </span>
      <style>{`
        .thinking-dots {
          display: inline-flex;
          align-items: center;
          gap: 4px;
        }
        .thinking-dot {
          width: 6px;
          height: 6px;
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
  );
}
