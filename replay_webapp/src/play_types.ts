// Types for the interactive play server (proxied at /api/play/*).

import { CardData, GameSnapshot, NobleData } from "./types";

export type BotKind =
  | "random"
  | "heuristic"
  | "heuristic_opus"
  | "net"
  | "human";

export interface ModelInfo {
  id: string;
  label: string;
  kind: BotKind;
  run: string | null;
  tag: string | null;
  ckpt: string | null;
  rating: number;
  elo: number;
  games: number;
  hidden?: number | null;
  arch?: string | null;
  score_hint?: number | null;
  winrate_vs_heuristic?: number | null;
}

export interface HumanRatingHistoryOpponent {
  seat: number;
  entity_id: string;
  model_id: string;
  label: string;
  opp_rating: number;
  score: number;
}

export interface HumanRatingHistoryEntry {
  timestamp: string;
  seed: number;
  human_seat: number;
  human_rank: number;
  ranks: number[];
  final_scores: { points: number; cards: number; nobles: number }[];
  old_rating: number;
  new_rating: number;
  delta: number;
  opponents: HumanRatingHistoryOpponent[];
  legacy?: boolean;
}

export interface UserMe {
  username: string;
  rating_system?: string;
  rating: number;
  elo: number;
  games: number;
  anchors?: Record<string, number>;
  history?: HumanRatingHistoryEntry[];
  results?: Record<string, unknown>[];
}

export interface GameListItem {
  game_id: string;
  num_players: number;
  human_seat: number;
  seed: number;
  status: string;
  step_count: number;
  updated_at?: string;
}

export interface LeaderboardCombinedEntry {
  kind: string;
  entity_id?: string;
  label: string;
  rating: number;
  elo: number;
  games: number;
  username?: string;
  model_id?: string;
  bot_kind?: string;
}

export interface LeaderboardResponse {
  agents: LeaderboardCombinedEntry[];
  humans: LeaderboardCombinedEntry[];
  combined: LeaderboardCombinedEntry[];
}

/** @deprecated use UserMe instead */
export type HumanElo = UserMe;

export interface PlaySeatInfo {
  seat: number;
  kind: BotKind;
  label: string;
  model_id: string | null;
  rating: number | null;
  elo: number | null;
}

export interface PlayStep {
  step: number;
  player: number;
  phase: number;
  action: number;
  action_name: string;
  action_detail: {
    kind: string;
    colors?: number[];
    color?: number;
    tier?: number;
    slot?: number;
    token?: number;
    raw?: number;
  };
  legal_actions: number[];
  state_after: GameSnapshot;
}

export interface EloUpdateResult {
  old_elo: number;
  new_elo: number;
  old_rating: number;
  new_rating: number;
  delta: number;
  games: number;
  per_opponent: HumanRatingHistoryOpponent[];
}

export type PlayStatus = "human_turn" | "ai_thinking" | "ended";

export interface PlayView {
  game_id: string;
  num_players: number;
  human_seat: number;
  players: PlaySeatInfo[];
  initial_state: GameSnapshot;
  cards: CardData[];
  nobles: NobleData[];
  steps: PlayStep[];
  snapshot: GameSnapshot;
  legal_actions: number[] | null;
  current_player: number | null;
  phase: number | null;
  status: PlayStatus;
  winners: number[] | null;
  final_scores: { points: number; cards: number; nobles: number }[] | null;
  seed: number;
  elo_update: EloUpdateResult | null;
  aborted: boolean;
}

export interface NewGameRequest {
  num_players: number;
  human_seat: number;
  opponents: Record<number, string>;
  seed?: number;
  num_sims?: number;
}
