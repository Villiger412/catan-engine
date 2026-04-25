export interface HexData {
  id: number
  resource: 'wood' | 'brick' | 'wheat' | 'ore' | 'sheep' | 'desert'
  number: number
  q: number
  r: number
}

export interface PortData {
  type: string  // "3:1" | "2:1:brick" | "2:1:wood" | etc.
  v1: number
  v2: number
}

export interface BoardData {
  hexes: HexData[]
  ports: PortData[]
}

export interface GamePosition {
  settlements: number[][]  // settlements[player] = vertex IDs
  cities: number[][]       // cities[player] = vertex IDs
  roads: number[][]        // roads[player]    = edge IDs
  resources: number[][]    // resources[player] = [brick, wood, wheat, ore, sheep]
  unplayed_dev: number[][] // unplayed_dev[player] = [knight, vp, road_building, year_of_plenty, monopoly]
  knights_played: number[] // per player
  vp_hidden: number[]      // per player (face-down VP cards)
  robber_hex?: number
  current_player: number
}

export interface SimulateRequest {
  n_simulations: number
  policy: 'rule_based' | 'random' | 'mcts' | 'mcts_rule' | string
  antithetic: boolean
  seed: number
  target_margin?: number
  method: string
  position?: GamePosition
  coalition_pressure: number
}

export interface MethodInfo {
  id: string
  name: string
  description: string
}

export interface SimResult {
  probabilities: number[]
  ci_lower: number[]
  ci_upper: number[]
  simulations_run: number
  elapsed_ms: number
  games_per_sec: number
  avg_turns: number
  draws: number
  policy: string
  max_margin: number
}

export const PLAYER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'] as const
export const PLAYER_NAMES = ['Red', 'Blue', 'Green', 'Orange'] as const

export interface SeatStats {
  win_rate: number
  avg_vp: number
  avg_cities: number
  avg_settlements: number
  avg_roads: number
  avg_knights: number
  lr_rate: number
  la_rate: number
}

export interface RecordsResult {
  games_run: number
  elapsed_ms: number
  games_per_sec: number
  policy: string
  turns_min: number
  turns_p25: number
  turns_median: number
  turns_mean: number
  turns_p75: number
  turns_max: number
  seat_stats: SeatStats[]
}

export interface RecordsRequest {
  n_games: number
  policy: string
  antithetic: boolean
  seed: number
  coalition_pressure: number
  board?: BoardData
  position?: GamePosition
}
