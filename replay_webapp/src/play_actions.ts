// Decode action ints into structured data for rendering action buttons.
// Mirrors the layout in agent/env/actions.py.

export const TAKE3_BASE = 0;
export const TAKE3_COUNT = 10;
export const TAKE2_BASE = 10;
export const TAKE2_COUNT = 5;
export const RESERVE_GRID_BASE = 15;
export const RESERVE_GRID_COUNT = 12;
export const RESERVE_BLIND_BASE = 27;
export const RESERVE_BLIND_COUNT = 3;
export const BUY_GRID_BASE = 30;
export const BUY_GRID_COUNT = 12;
export const BUY_RESERVED_BASE = 42;
export const BUY_RESERVED_COUNT = 3;
export const PASS_ACTION = 45;
export const DISCARD_BASE = 46;
export const DISCARD_COUNT = 6;
export const PICK_NOBLE_BASE = 52;
export const PICK_NOBLE_COUNT = 5;
export const NUM_GRID_SLOTS = 4;

const COMBOS3: [number, number, number][] = [];
for (let i = 0; i < 5; i++)
  for (let j = i + 1; j < 5; j++)
    for (let k = j + 1; k < 5; k++) COMBOS3.push([i, j, k]);

export type ActionKind =
  | "take3"
  | "take2"
  | "reserve_grid"
  | "reserve_blind"
  | "buy_grid"
  | "buy_reserved"
  | "pass"
  | "discard"
  | "pick_noble"
  | "unknown";

export interface DecodedAction {
  action: number;
  kind: ActionKind;
  colors?: number[];
  color?: number;
  tier?: number;
  slot?: number;
  rslot?: number;
  token?: number;
}

export function decodeAction(a: number): DecodedAction {
  if (a >= TAKE3_BASE && a < TAKE3_BASE + TAKE3_COUNT) {
    return { action: a, kind: "take3", colors: COMBOS3[a - TAKE3_BASE] };
  }
  if (a >= TAKE2_BASE && a < TAKE2_BASE + TAKE2_COUNT) {
    return { action: a, kind: "take2", color: a - TAKE2_BASE };
  }
  if (a >= RESERVE_GRID_BASE && a < RESERVE_GRID_BASE + RESERVE_GRID_COUNT) {
    const x = a - RESERVE_GRID_BASE;
    return {
      action: a,
      kind: "reserve_grid",
      tier: Math.floor(x / NUM_GRID_SLOTS),
      slot: x % NUM_GRID_SLOTS,
    };
  }
  if (a >= RESERVE_BLIND_BASE && a < RESERVE_BLIND_BASE + RESERVE_BLIND_COUNT) {
    return { action: a, kind: "reserve_blind", tier: a - RESERVE_BLIND_BASE };
  }
  if (a >= BUY_GRID_BASE && a < BUY_GRID_BASE + BUY_GRID_COUNT) {
    const x = a - BUY_GRID_BASE;
    return {
      action: a,
      kind: "buy_grid",
      tier: Math.floor(x / NUM_GRID_SLOTS),
      slot: x % NUM_GRID_SLOTS,
    };
  }
  if (a >= BUY_RESERVED_BASE && a < BUY_RESERVED_BASE + BUY_RESERVED_COUNT) {
    return { action: a, kind: "buy_reserved", rslot: a - BUY_RESERVED_BASE };
  }
  if (a === PASS_ACTION) return { action: a, kind: "pass" };
  if (a >= DISCARD_BASE && a < DISCARD_BASE + DISCARD_COUNT) {
    return { action: a, kind: "discard", token: a - DISCARD_BASE };
  }
  if (a >= PICK_NOBLE_BASE && a < PICK_NOBLE_BASE + PICK_NOBLE_COUNT) {
    return { action: a, kind: "pick_noble", slot: a - PICK_NOBLE_BASE };
  }
  return { action: a, kind: "unknown" };
}

export const COLOR_LETTERS = ["W", "B", "G", "R", "K", "g"]; // last is gold
