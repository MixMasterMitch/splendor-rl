interface StepSliderProps {
  stepIdx: number;        // current index (0 = initial, 1..N = after step N)
  maxStep: number;        // total number of steps (= steps.length)
  playing: boolean;
  playSpeed: number;      // ms per step
  onStep: (idx: number) => void;
  onPlayPause: () => void;
  onSpeedChange: (ms: number) => void;
}

const SPEED_OPTIONS = [
  { label: "0.25x", ms: 4000 },
  { label: "0.5x",  ms: 2000 },
  { label: "1x",    ms: 1000 },
  { label: "2x",    ms: 500  },
  { label: "4x",    ms: 250  },
];

export function StepSlider({
  stepIdx,
  maxStep,
  playing,
  playSpeed,
  onStep,
  onPlayPause,
  onSpeedChange,
}: StepSliderProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 12px",
        background: "#0f3460",
        borderTop: "1px solid #2a4a7f",
        flexShrink: 0,
      }}
    >
      {/* Prev */}
      <button
        onClick={() => onStep(Math.max(0, stepIdx - 1))}
        disabled={stepIdx <= 0}
        title="Previous step (Left Arrow)"
        style={{ minWidth: 30, display: "flex", alignItems: "center", justifyContent: "center" }}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <polygon points="12,1 3,7 12,13" />
        </svg>
      </button>

      {/* Play/Pause */}
      <button
        onClick={onPlayPause}
        disabled={maxStep === 0}
        title="Play/Pause (Space)"
        style={{ minWidth: 30, display: "flex", alignItems: "center", justifyContent: "center" }}
      >
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
            <rect x="2" y="1" width="4" height="12" rx="1" />
            <rect x="8" y="1" width="4" height="12" rx="1" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
            <polygon points="2,1 13,7 2,13" />
          </svg>
        )}
      </button>

      {/* Next */}
      <button
        onClick={() => onStep(Math.min(maxStep, stepIdx + 1))}
        disabled={stepIdx >= maxStep}
        title="Next step (Right Arrow)"
        style={{ minWidth: 30, display: "flex", alignItems: "center", justifyContent: "center" }}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <polygon points="2,1 11,7 2,13" />
        </svg>
      </button>

      {/* Slider */}
      <input
        type="range"
        min={0}
        max={maxStep}
        value={stepIdx}
        onChange={(e) => onStep(Number(e.target.value))}
        style={{ flex: 1, minWidth: 80 }}
        title={`Step ${stepIdx} / ${maxStep}`}
      />

      {/* Step counter */}
      <span
        style={{
          fontSize: 12,
          color: "#7a94b8",
          minWidth: 64,
          textAlign: "right",
          whiteSpace: "nowrap",
        }}
      >
        {stepIdx} / {maxStep}
      </span>

      {/* Speed selector */}
      <select
        value={playSpeed}
        onChange={(e) => onSpeedChange(Number(e.target.value))}
        title="Playback speed"
        style={{ fontSize: 12 }}
      >
        {SPEED_OPTIONS.map((opt) => (
          <option key={opt.ms} value={opt.ms}>
            {opt.label}
          </option>
        ))}
      </select>

      {/* First / Last */}
      <button
        onClick={() => onStep(0)}
        disabled={stepIdx <= 0}
        title="First step (Home)"
        style={{ minWidth: 34, display: "flex", alignItems: "center", justifyContent: "center", gap: 1 }}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <rect x="1" y="1" width="2.5" height="12" rx="1" />
          <polygon points="13,1 5,7 13,13" />
        </svg>
      </button>
      <button
        onClick={() => onStep(maxStep)}
        disabled={stepIdx >= maxStep}
        title="Last step (End)"
        style={{ minWidth: 34, display: "flex", alignItems: "center", justifyContent: "center", gap: 1 }}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
          <polygon points="1,1 9,7 1,13" />
          <rect x="10.5" y="1" width="2.5" height="12" rx="1" />
        </svg>
      </button>
    </div>
  );
}
