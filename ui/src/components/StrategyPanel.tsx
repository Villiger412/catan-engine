import { useState } from 'react'
import type { RecordsResult, SimulateRequest } from '../types'

const TEMPLATE = `from catan_research import Strategy

class MyStrategy(Strategy):
    """
    Override hooks below — anything not overridden falls through to rule_based.

    Available ctx helpers:
      ctx.vertex_production_value(vertex_id) -> float
      ctx.vertex_resource_diversity(vertex_id) -> int
      ctx.player_vp(seat) -> int
      ctx.player_resources(seat) -> dict
      ctx.current_player() -> int
      ctx.turn_number() -> int
      ctx.is_port(vertex_id) -> bool
    """

    def score_setup_vertex(self, ctx, vertex_id):
        # Prefer high-production vertices (default rule_based behaviour)
        return ctx.vertex_production_value(vertex_id)

    # def score_setup_road(self, ctx, edge_id): ...
    # def choose_robber_hex(self, ctx, candidates): ...
    # def rank_main_action(self, ctx, actions): ...
`

interface Props {
  config: SimulateRequest
  isCustomBoard: boolean
  board: unknown
  hasPosition: boolean
  position: unknown
  onResult: (r: RecordsResult) => void
}

export default function StrategyPanel({ config, isCustomBoard, board, hasPosition, position, onResult }: Props) {
  const [code, setCode] = useState(TEMPLATE)
  const [className, setClassName] = useState('MyStrategy')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const body: Record<string, unknown> = {
        code,
        class_name: className,
        n_games: config.n_simulations,
        seed: config.seed,
        antithetic: config.antithetic,
        coalition_pressure: config.coalition_pressure,
      }
      if (isCustomBoard) body.board = board
      if (hasPosition) body.position = position

      const res = await fetch('/api/run-strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? `${res.status}`)
      onResult(data as RecordsResult)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="win-panel strategy-panel">
      <div className="panel-title">Python Strategy</div>

      <textarea
        className="strategy-editor"
        value={code}
        onChange={e => setCode(e.target.value)}
        spellCheck={false}
        aria-label="Strategy code editor"
      />

      <div className="strategy-class-row">
        <span className="strategy-class-label">Class</span>
        <input
          className="strategy-class-input"
          value={className}
          onChange={e => setClassName(e.target.value)}
          placeholder="MyStrategy"
          aria-label="Strategy class name"
        />
        <button
          className="strategy-run-btn"
          onClick={run}
          disabled={loading || !className.trim()}
        >
          {loading ? '⏳ Running…' : '▶ Run'}
        </button>
      </div>

      {error && <div className="strategy-error">{error}</div>}
    </div>
  )
}
