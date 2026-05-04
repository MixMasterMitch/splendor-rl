// Shared TypeScript types that mirror the Python replay JSON schema.

// ---------------------------------------------------------------------------
// Card / Noble static data
// ---------------------------------------------------------------------------

export interface CardData {
  id: number;
  level: number;    // 1 | 2 | 3
  bonus: number;    // color index 0-4 (W,B,G,R,K)
  points: number;
  cost: number[];   // length 5: [W,B,G,R,K]
}

export interface NobleData {
  id: number;
  name: string;
  points: number;
  requirement: number[];  // length 5: [W,B,G,R,K]
}

// ---------------------------------------------------------------------------
// Replay schema
// ---------------------------------------------------------------------------

export interface PlayerInfo {
  name: string;
  kind: string;
  checkpoint?: string;
  num_sims?: number;
  seed?: number;
}

export interface PlayerSnapshot {
  tokens: number[];           // length 6: W,B,G,R,K,Gold
  bonuses: number[];          // length 5: W,B,G,R,K
  reserved: (number | null)[];      // length 3
  reserved_hidden: boolean[];       // length 3
  points: number;
  nobles_claimed: number;
}

export interface GameSnapshot {
  gem_pool: number[];                 // length 6
  grid: (number | null)[][];         // [3][4]
  deck_counts: number[];             // length 3
  nobles: (number | null)[];         // length 5
  current_player: number;
  phase: number;
  turn_count: number;
  ended: boolean;
  players: PlayerSnapshot[];
}

export interface ActionDetail {
  kind: string;
  colors?: number[];
  color?: number;
  tier?: number;
  slot?: number;
  token?: number;
  raw?: number;
}

export interface ReplayStep {
  step: number;
  player: number;
  phase: number;
  action: number;
  action_name: string;
  action_detail: ActionDetail;
  legal_actions: number[];
  state_after: GameSnapshot;
}

export interface FinalScore {
  points: number;
  cards: number;
  nobles: number;
}

export interface Replay {
  version: number;
  seed: number;
  num_players: number;
  players: PlayerInfo[];
  starting_player: number;
  cards: CardData[];
  nobles: NobleData[];
  initial_state: GameSnapshot;
  steps: ReplayStep[];
  final_scores: FinalScore[];
  winners: number[];
}

// Lightweight listing entry returned by /api/replays
export interface ReplayMeta {
  name: string;
  num_players: number | null;
  players: PlayerInfo[] | null;
  winners: number[] | null;
  final_scores: FinalScore[] | null;
  step_count: number | null;
  mtime: number;
}

// ---------------------------------------------------------------------------
// UI constants
// ---------------------------------------------------------------------------

// Color index -> display color name
export const COLOR_NAMES = ["White", "Blue", "Green", "Red", "Black", "Gold"] as const;

// CSS color values per gem color (index 0-5)
export const GEM_COLORS: string[] = [
  "#e8e0d0",  // White (diamond)
  "#3a7dc9",  // Blue  (sapphire)
  "#3cae5c",  // Green (emerald)
  "#d94f3d",  // Red   (ruby)
  "#2b2b2b",  // Black (onyx)
  "#e8c848",  // Gold
];

export const GEM_TEXT_COLORS: string[] = [
  "#333",     // on white
  "#fff",     // on blue
  "#fff",     // on green
  "#fff",     // on red
  "#fff",     // on black
  "#333",     // on gold
];

export const PHASE_LABELS: Record<number, string> = {
  0: "Main",
  1: "Discard",
  2: "Noble",
};
