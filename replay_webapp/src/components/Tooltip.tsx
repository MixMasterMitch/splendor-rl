import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  /** Position relative to the target element */
  position?: "right" | "top" | "bottom" | "left";
  /** Extra style on the wrapper span */
  style?: CSSProperties;
}

/**
 * A tooltip component that shows a styled popup with a caret/arrow
 * positioned relative to the target element.
 */
export function Tooltip({ content, children, position = "right", style }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  const wrapperRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hideTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = useCallback(() => {
    if (hideTimeout.current) {
      clearTimeout(hideTimeout.current);
      hideTimeout.current = null;
    }
    setVisible(true);
  }, []);

  const hide = useCallback(() => {
    hideTimeout.current = setTimeout(() => setVisible(false), 120);
  }, []);

  useEffect(() => {
    if (!visible || !wrapperRef.current) {
      setCoords(null);
      return;
    }
    const rect = wrapperRef.current.getBoundingClientRect();
    const gap = 8;
    let top: number;
    let left: number;

    switch (position) {
      case "right":
        top = rect.top + rect.height / 2;
        left = rect.right + gap;
        break;
      case "left":
        top = rect.top + rect.height / 2;
        left = rect.left - gap;
        break;
      case "top":
        top = rect.top - gap;
        left = rect.left + rect.width / 2;
        break;
      case "bottom":
        top = rect.bottom + gap;
        left = rect.left + rect.width / 2;
        break;
    }
    setCoords({ top, left });
  }, [visible, position]);

  // Determine transform to center the tooltip relative to the anchor point
  const getTransform = (): string => {
    switch (position) {
      case "right":
        return "translateY(-50%)";
      case "left":
        return "translate(-100%, -50%)";
      case "top":
        return "translate(-50%, -100%)";
      case "bottom":
        return "translateX(-50%)";
    }
  };

  const getCaretStyle = (): CSSProperties => {
    const base: CSSProperties = {
      position: "absolute",
      width: 0,
      height: 0,
    };
    const size = 6;
    switch (position) {
      case "right":
        return {
          ...base,
          left: -size,
          top: "50%",
          transform: "translateY(-50%)",
          borderTop: `${size}px solid transparent`,
          borderBottom: `${size}px solid transparent`,
          borderRight: `${size}px solid #1e2a3e`,
        };
      case "left":
        return {
          ...base,
          right: -size,
          top: "50%",
          transform: "translateY(-50%)",
          borderTop: `${size}px solid transparent`,
          borderBottom: `${size}px solid transparent`,
          borderLeft: `${size}px solid #1e2a3e`,
        };
      case "top":
        return {
          ...base,
          bottom: -size,
          left: "50%",
          transform: "translateX(-50%)",
          borderLeft: `${size}px solid transparent`,
          borderRight: `${size}px solid transparent`,
          borderTop: `${size}px solid #1e2a3e`,
        };
      case "bottom":
        return {
          ...base,
          top: -size,
          left: "50%",
          transform: "translateX(-50%)",
          borderLeft: `${size}px solid transparent`,
          borderRight: `${size}px solid transparent`,
          borderBottom: `${size}px solid #1e2a3e`,
        };
    }
  };

  return (
    <span
      ref={wrapperRef}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      style={{ display: "inline-block", ...style }}
      tabIndex={0}
    >
      {children}
      {visible && coords && content && (
        <div
          ref={tooltipRef}
          onMouseEnter={show}
          onMouseLeave={hide}
          style={{
            position: "fixed",
            top: coords.top,
            left: coords.left,
            transform: getTransform(),
            zIndex: 9999,
            background: "#1e2a3e",
            border: "1px solid #3a4f6f",
            borderRadius: 6,
            padding: "8px 12px",
            color: "#e0e6f0",
            fontSize: 12,
            lineHeight: 1.5,
            whiteSpace: "nowrap",
            boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
            pointerEvents: "auto",
          }}
          role="tooltip"
        >
          <div style={getCaretStyle()} />
          {content}
        </div>
      )}
    </span>
  );
}
