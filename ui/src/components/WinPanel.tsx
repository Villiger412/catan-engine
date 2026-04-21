import { useEffect, useRef, useState } from 'react'
import type { SimResult } from '../types'
import { PLAYER_COLORS, PLAYER_NAMES } from '../types'

interface Props {
  result: SimResult | null
  loading: boolean
}

export default function WinPanel({ result, loading }: Props) {
  const INIT = [0.25, 0.25, 0.25, 0.25]
  const displayedRef = useRef(INIT)
  const [displayed, setDisplayed] = useState(INIT)

  useEffect(() => {
    if (!result) return
    const target = result.probabilities
    // Always start from the actual current displayed values (not stale state)
    const start = displayedRef.current.slice()
    const startTime = performance.now()
    const duration = 900

    let raf: number
    function animate(now: number) {
      const t = Math.min((now - startTime) / duration, 1)
      const ease = 1 - Math.pow(1 - t, 3)
      const next = start.map((s, i) => s + (target[i] - s) * ease)
      displayedRef.current = next
      setDisplayed(next)
      if (t < 1) raf = requestAnimationFrame(animate)
    }
    raf = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(raf)
  }, [result])

  const isReady = result !== null

  return (
    <div className="win-panel">
      <div className="panel-title">Win Probabilities</div>

      <div className="player-list">
        {PLAYER_NAMES.map((name, i) => {
          const prob = displayed[i]
          const lo = result?.ci_lower[i] ?? 0
          const hi = result?.ci_upper[i] ?? 0
          const color = PLAYER_COLORS[i]
          const pct = (prob * 100).toFixed(1)

          return (
            <div key={i} className="player-row">
              {/* Avatar */}
              <div className="player-avatar" style={{ background: color }}>
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none">
                  <circle cx="12" cy="8" r="4" fill="white" />
                  <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" fill="white" />
                </svg>
              </div>

              {/* Name + bar */}
              <div className="player-bar-wrap">
                <div className="player-name-row">
                  <span className="player-name" style={{ color }}>P{i + 1} · {name}</span>
                  <span className="player-pct" style={{ color: isReady ? '#fff' : '#666' }}>
                    {isReady ? `${pct}%` : '—'}
                  </span>
                </div>

                {/* Bar track */}
                <div className="bar-track">
                  {/* CI range (lighter shade, underneath) */}
                  {isReady && (
                    <div
                      className="bar-ci"
                      style={{
                        left: `${lo * 100}%`,
                        width: `${(hi - lo) * 100}%`,
                        background: color + '55',
                      }}
                    />
                  )}
                  {/* Main probability bar */}
                  <div
                    className="bar-fill"
                    style={{
                      width: `${prob * 100}%`,
                      background: `linear-gradient(90deg, ${color}cc, ${color})`,
                      boxShadow: isReady ? `0 0 8px ${color}88` : 'none',
                    }}
                  />
                  {/* Loading shimmer */}
                  {loading && <div className="bar-shimmer" />}
                </div>

                {/* CI text */}
                {isReady && (
                  <div className="bar-ci-label">
                    95% CI: [{(lo * 100).toFixed(1)}%, {(hi * 100).toFixed(1)}%]
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Stats footer */}
      {isReady && (
        <div className="stats-grid">
          <StatCell label="Simulations" value={result.simulations_run.toLocaleString()} />
          <StatCell label="Time" value={`${result.elapsed_ms.toFixed(0)} ms`} />
          <StatCell label="Speed" value={`${(result.games_per_sec / 1000).toFixed(1)}K g/s`} />
          <StatCell label="Avg turns" value={result.avg_turns.toFixed(1)} />
          <StatCell
            label="Turns/s"
            value={`${((result.avg_turns * result.games_per_sec) / 1000).toFixed(1)}K`}
          />
          <StatCell label="CI margin" value={`±${(result.max_margin * 100).toFixed(2)}%`} />
          <StatCell label="Policy" value={result.policy.replace('_', '-')} />
        </div>
      )}

      {!isReady && !loading && (
        <div className="idle-hint">
          Run a simulation to see win probabilities
        </div>
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

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-cell">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  )
}
