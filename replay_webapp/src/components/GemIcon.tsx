/**
 * SVG gem icons, one per color index (0-5).
 *
 *   0 White  - Diamond (four-pointed rotated square)
 *   1 Blue   - Teardrop / water drop (sapphire)
 *   2 Green  - Beveled rectangle (emerald cut)
 *   3 Red    - Oval (ruby cabochon)
 *   4 Black  - Hexagon (onyx)
 *   5 Gold   - Eight-pointed star (wild)
 */

interface GemIconProps {
  colorIdx: number;
  size?: number;
  fill?: string;
  stroke?: string;
}

export function GemIcon({ colorIdx, size = 20, fill = "currentColor", stroke = "none" }: GemIconProps) {
  const s = size;

  switch (colorIdx) {
    // White: Diamond (rotated square with facet lines)
    case 0:
      return (
        <svg width={s} height={s} viewBox="0 0 20 20" fill="none">
          <polygon
            points="10,1 19,10 10,19 1,10"
            fill={fill}
            stroke={stroke}
            strokeWidth="1"
          />
          <line x1="10" y1="1" x2="10" y2="19" stroke="rgba(255,255,255,0.3)" strokeWidth="0.8" />
          <line x1="1" y1="10" x2="19" y2="10" stroke="rgba(255,255,255,0.3)" strokeWidth="0.8" />
          <line x1="10" y1="1" x2="19" y2="10" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8" />
          <line x1="10" y1="1" x2="1" y2="10" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8" />
        </svg>
      );

    // Blue: Water drop (sapphire)
    case 1:
      return (
        <svg width={s} height={s} viewBox="0 0 20 20" fill="none">
          <path
            d="M10 2 C10 2, 17 10, 17 13.5 A7 7 0 0 1 3 13.5 C3 10, 10 2, 10 2Z"
            fill={fill}
            stroke={stroke}
            strokeWidth="1"
          />
          <ellipse cx="7.5" cy="12" rx="1.5" ry="2.5" fill="rgba(255,255,255,0.25)" />
        </svg>
      );

    // Green: Beveled rectangle (emerald cut)
    case 2:
      return (
        <svg width={s} height={s} viewBox="0 0 20 20" fill="none">
          <polygon
            points="5,2 15,2 18,6 18,14 15,18 5,18 2,14 2,6"
            fill={fill}
            stroke={stroke}
            strokeWidth="1"
          />
          <polygon
            points="6,5 14,5 16,8 16,12 14,15 6,15 4,12 4,8"
            fill="none"
            stroke="rgba(255,255,255,0.25)"
            strokeWidth="0.8"
          />
        </svg>
      );

    // Red: Oval / ruby cabochon
    case 3:
      return (
        <svg width={s} height={s} viewBox="0 0 20 20" fill="none">
          <ellipse cx="10" cy="10" rx="8" ry="6" fill={fill} stroke={stroke} strokeWidth="1" />
          <ellipse cx="7.5" cy="8" rx="2" ry="1.5" fill="rgba(255,255,255,0.3)" />
        </svg>
      );

    // Black: Hexagon (onyx)
    case 4:
      return (
        <svg width={s} height={s} viewBox="0 0 20 20" fill="none">
          <polygon
            points="10,1 18,5.5 18,14.5 10,19 2,14.5 2,5.5"
            fill={fill}
            stroke={stroke}
            strokeWidth="1"
          />
          <polygon
            points="10,5 15,7.7 15,13 10,16 5,13 5,7.7"
            fill="none"
            stroke="rgba(255,255,255,0.2)"
            strokeWidth="0.8"
          />
        </svg>
      );

    // Gold: Eight-pointed star
    case 5:
      return (
        <svg width={s} height={s} viewBox="0 0 20 20" fill="none">
          <polygon
            points="10,1 11.8,7.5 18,6.5 13.2,11 16,17.5 10,14 4,17.5 6.8,11 2,6.5 8.2,7.5"
            fill={fill}
            stroke={stroke}
            strokeWidth="0.5"
          />
        </svg>
      );

    default:
      return (
        <svg width={s} height={s} viewBox="0 0 20 20">
          <circle cx="10" cy="10" r="8" fill={fill} stroke={stroke} strokeWidth="1" />
        </svg>
      );
  }
}
