import type { RecordsResult } from '../types'
import { PLAYER_COLORS, PLAYER_NAMES } from '../types'
import { buildRecordsCsv, downloadCsv } from '../lib/csv'

interface Props {
  result: RecordsResult | null
  loading: boolean
}

const ROWS: { key: keyof import('../types').SeatStats; label: string; fmt: (v: number) => string }[] = [
  { key: 'win_rate',        label: 'Win Rate',    fmt: v => `${(v * 100).toFixed(1)}%` },
  { key: 'avg_vp',          label: 'Avg VP',      fmt: v => v.toFixed(2) },
  { key: 'avg_cities',      label: 'Avg Cities',  fmt: v => v.toFixed(2) },
  { key: 'avg_settlements', label: 'Avg Settles', fmt: v => v.toFixed(2) },
  { key: 'avg_roads',       label: 'Avg Roads',   fmt: v => v.toFixed(1) },
  { key: 'avg_knights',     label: 'Avg Knights', fmt: v => v.toFixed(2) },
  { key: 'lr_rate',         label: 'LR Rate',     fmt: v => `${(v * 100).toFixed(1)}%` },
  { key: 'la_rate',         label: 'LA Rate',     fmt: v => `${(v * 100).toFixed(1)}%` },
]

export default function RecordsPanel({ result, loading }: Props) {
  const handleDownload = () => {
    if (!result) return
    const csv = buildRecordsCsv(result)
    const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-')
    downloadCsv(csv, `catan-stats-${ts}.csv`)
  }

  return (
    <div className="win-panel records-panel">
      <div className="panel-title">
        Game Statistics
        {result && (
          <button className="records-csv-btn" onClick={handleDownload} title="Download results as CSV">
            ⬇ CSV
          </button>
        )}
      </div>

      {result && (
        <>
          {/* Turn distribution */}
          <div className="records-section-label">Turn Length Distribution</div>
          <div className="turns-dist">
            <TurnCell label="Min"    value={result.turns_min.toString()} />
            <TurnCell label="P25"    value={result.turns_p25.toFixed(0)} />
            <TurnCell label="Median" value={result.turns_median.toFixed(0)} highlight />
            <TurnCell label="Mean"   value={result.turns_mean.toFixed(1)} />
            <TurnCell label="P75"    value={result.turns_p75.toFixed(0)} />
            <TurnCell label="Max"    value={result.turns_max.toString()} />
          </div>

          {/* Per-seat stats table */}
          <div className="records-section-label" style={{ marginTop: 14 }}>Per-Seat Averages</div>
          <div className="records-table-wrap">
            <table className="records-table">
              <thead>
                <tr>
                  <th className="records-th-row" />
                  {PLAYER_NAMES.map((name, i) => (
                    <th key={i} className="records-th-col" style={{ color: PLAYER_COLORS[i] }}>
                      <span className="records-player-dot" style={{ background: PLAYER_COLORS[i] }} />
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ROWS.map(row => (
                  <tr key={row.key} className="records-tr">
                    <td className="records-td-label">{row.label}</td>
                    {result.seat_stats.map((s, i) => {
                      const val = s[row.key]
                      const isWinRate = row.key === 'win_rate'
                      const maxVal = isWinRate ? Math.max(...result.seat_stats.map(x => x.win_rate)) : null
                      const isLeader = isWinRate && val === maxVal
                      return (
                        <td
                          key={i}
                          className={`records-td-val${isLeader ? ' records-leader' : ''}`}
                          style={isLeader ? { color: PLAYER_COLORS[i] } : undefined}
                        >
                          {row.fmt(val)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Footer: two main cells + throughput line */}
          <div className="records-footer">
            <div className="records-footer-main">
              <div className="stat-cell">
                <div className="stat-label">Games</div>
                <div className="stat-value">{result.games_run.toLocaleString()}</div>
              </div>
              <div className="stat-cell">
                <div className="stat-label">Policy</div>
                <div className="stat-value">{result.policy.replace(/_/g, '-')}</div>
              </div>
            </div>
            <div className="records-footer-perf">
              {result.elapsed_ms.toFixed(0)} ms · {(result.games_per_sec / 1000).toFixed(1)}K games/s
            </div>
          </div>
        </>
      )}

      {!result && !loading && (
        <div className="idle-hint">Run a simulation to see per-game statistics</div>
      )}

      {loading && (
        <div className="loading-hint">
          <span className="spinner" />
          Simulating…
        </div>
      )}
    </div>
  )
}

function TurnCell({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`turn-cell${highlight ? ' turn-cell-hi' : ''}`}>
      <div className="turn-cell-label">{label}</div>
      <div className="turn-cell-value">{value}</div>
    </div>
  )
}
