import type { RecordsResult } from '../types'
import { PLAYER_NAMES } from '../types'

/** Build a CSV string from a RecordsResult. One header row + 4 seat rows. */
export function buildRecordsCsv(r: RecordsResult): string {
  const cell = (v: string | number) => {
    const s = String(v)
    return s.includes(',') || s.includes('"') || s.includes('\n')
      ? `"${s.replace(/"/g, '""')}"`
      : s
  }

  const headers = [
    'seat', 'player',
    'win_rate', 'avg_vp', 'avg_cities', 'avg_settlements',
    'avg_roads', 'avg_knights', 'lr_rate', 'la_rate',
    'turns_min', 'turns_p25', 'turns_median', 'turns_mean', 'turns_p75', 'turns_max',
    'games_run', 'policy', 'elapsed_ms', 'games_per_sec',
  ]

  const rows = r.seat_stats.map((s, i) => [
    i,
    PLAYER_NAMES[i],
    s.win_rate.toFixed(6),
    s.avg_vp.toFixed(4),
    s.avg_cities.toFixed(4),
    s.avg_settlements.toFixed(4),
    s.avg_roads.toFixed(4),
    s.avg_knights.toFixed(4),
    s.lr_rate.toFixed(6),
    s.la_rate.toFixed(6),
    r.turns_min,
    r.turns_p25.toFixed(1),
    r.turns_median.toFixed(1),
    r.turns_mean.toFixed(2),
    r.turns_p75.toFixed(1),
    r.turns_max,
    r.games_run,
    r.policy,
    r.elapsed_ms.toFixed(1),
    r.games_per_sec.toFixed(1),
  ])

  return [headers, ...rows].map(row => row.map(cell).join(',')).join('\n')
}

export function downloadCsv(content: string, filename: string) {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
