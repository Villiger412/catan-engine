// Per-policy timing profile: elapsed_ms ≈ overhead_ms + ms_per_game × n_simulations.
// Numbers measured on this dev box (Windows, beginner board, multi-threaded Rayon).
// They are rough — throughput varies with board, coalition_pressure, and contention.
const PROFILES: Record<string, { overhead: number; msPerGame: number }> = {
  rule_based: { overhead: 500, msPerGame: 0.15 },
  random:     { overhead: 200, msPerGame: 0.33 },
  mcts:       { overhead: 400, msPerGame: 45  },
  mcts_rule:  { overhead: 400, msPerGame: 90  },
}

function profileFor(policy: string | undefined | null): { overhead: number; msPerGame: number } {
  // Null-safe: treat missing policy as the default (rule_based).
  if (typeof policy !== 'string' || policy.length === 0) return PROFILES.rule_based
  // Accept the Rust alias "rule-based" (see src/lib.rs parse_policy).
  const normalized = policy === 'rule-based' ? 'rule_based' : policy
  if (normalized.startsWith('mcts_rule')) {
    const n = parseInt(normalized.slice('mcts_rule_'.length), 10)
    const base = PROFILES.mcts_rule
    if (!Number.isFinite(n)) return base
    return { overhead: base.overhead, msPerGame: base.msPerGame * (n / 200) }
  }
  if (normalized.startsWith('mcts')) {
    const n = parseInt(normalized.slice('mcts_'.length), 10)
    const base = PROFILES.mcts
    if (!Number.isFinite(n)) return base
    return { overhead: base.overhead, msPerGame: base.msPerGame * (n / 200) }
  }
  return PROFILES[normalized] ?? PROFILES.rule_based
}

/** Calibration snapshot from a previous run.
 *
 * `turnsPerSec = avg_turns × games_per_sec` is a mostly position-agnostic
 * throughput metric: simulation cost is roughly linear in turns played, so
 * turns/sec tends to stay stable even as position progress changes game length.
 * `progress` is the positionProgress at which the calibration was measured,
 * used to translate observed avg_turns into expected avg_turns at a new
 * progress level.
 */
export type Calibration = {
  policy: string
  msPerGame: number
  avgTurns: number
  turnsPerSec: number
  progress: number
  simulationsRun: number
}

/** Auto-mode convergence constant — overridden when we have calibration data.
 * Seeded from early manual measurements on the beginner board; log `stats.n`
 * on converged runs and tune this if the typical value drifts. */
export const AUTO_MODE_TYPICAL_N = 4000

/** Estimated wall-clock time in ms for (policy, n_simulations).
 *
 * `positionProgress` is an optional 0..1 hint of how far into the game the
 * loaded position is (0 = fresh game, ~1 = near terminal). Simulation cost is
 * roughly linear in remaining turns, so we scale by `(1 - progress)` with a
 * floor of 0.15 to avoid predicting implausibly short runs.
 *
 * `calibration` lets a prior run's measured throughput override the static
 * profile — see the `Calibration` type. We only trust it when the policy
 * matches (different policies have wildly different per-game costs).
 */
export function estimateMs(
  policy: string,
  nSimulations: number,
  targetMargin?: number,
  positionProgress: number = 0,
  calibration?: Calibration | null,
): number {
  const p = profileFor(policy)
  const effectiveN = targetMargin !== undefined ? AUTO_MODE_TYPICAL_N : nSimulations
  const progress = Math.min(Math.max(positionProgress, 0), 0.85)
  const remainingFrac = Math.max(0.15, 1 - progress)

  // Self-correcting path: translate the prior run's turns/sec into an expected
  // ms-per-game at the *current* progress. turnsPerSec ≈ constant throughput,
  // so expected_avg_turns_now × 1000 / turnsPerSec gives the new ms/game.
  if (calibration && calibration.policy === policy && calibration.turnsPerSec > 0) {
    const oldRemaining = Math.max(0.15, 1 - Math.min(Math.max(calibration.progress, 0), 0.85))
    const expectedAvgTurnsNow = calibration.avgTurns * (remainingFrac / oldRemaining)
    const calibratedMsPerGame = (expectedAvgTurnsNow * 1000) / calibration.turnsPerSec
    return p.overhead + calibratedMsPerGame * effectiveN
  }

  return p.overhead + p.msPerGame * effectiveN * remainingFrac
}

/** Rough game-progress estimate from a GamePosition (0 = fresh, up to 0.85).
 *
 * Uses total visible VP across all four players. After initial placement each
 * player has 2 settlements = 8 VP total; a typical endgame sees ~18–22 total VP
 * (winner at 10, others trailing). We map 8 → 0 and 22 → 0.85, clamped.
 */
export function positionProgress(pos: {
  settlements?: number[][]
  cities?: number[][]
  vp_hidden?: number[]
} | null | undefined): number {
  if (!pos) return 0
  let vp = 0
  for (let p = 0; p < 4; p++) {
    vp += pos.settlements?.[p]?.length ?? 0
    vp += (pos.cities?.[p]?.length ?? 0) * 2
    vp += pos.vp_hidden?.[p] ?? 0
  }
  return Math.min(Math.max((vp - 8) / 14, 0), 0.85)
}

/** Human-readable short form, e.g. "instant", "~1.2s", "~25s", "~2m 10s". */
export function formatEta(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return 'instant'
  if (ms < 500) return 'instant'
  if (ms < 10_000) return `~${(ms / 1000).toFixed(1)}s`
  if (ms < 60_000) return `~${Math.round(ms / 1000)}s`
  const m = Math.floor(ms / 60_000)
  const s = Math.round((ms % 60_000) / 1000)
  return s === 0 ? `~${m}m` : `~${m}m ${s}s`
}

/** Longer form for tooltips, e.g. "~1.2 seconds", "~2 minutes 10 seconds". */
export function formatEtaLong(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return 'runs instantly'
  if (ms < 500) return 'runs instantly'
  if (ms < 10_000) return `~${(ms / 1000).toFixed(1)} seconds`
  if (ms < 60_000) return `~${Math.round(ms / 1000)} seconds`
  const m = Math.floor(ms / 60_000)
  const s = Math.round((ms % 60_000) / 1000)
  return s === 0 ? `~${m} minute${m === 1 ? '' : 's'}` : `~${m}m ${s}s`
}

/** Live elapsed-time counter format: "0:12", "1:05", "10:30". */
export function formatElapsed(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) ms = 0
  const totalSec = Math.floor(ms / 1000)
  const m = Math.floor(totalSec / 60)
  const s = totalSec % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}
