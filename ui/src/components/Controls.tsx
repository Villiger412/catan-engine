import { useEffect, useState } from 'react'
import type { GamePosition, MethodInfo, SimulateRequest } from '../types'
import type { Calibration } from '../lib/timing'
import { estimateMs, formatEta, formatEtaLong, formatElapsed, positionProgress } from '../lib/timing'

export type SimMode = 'winprob' | 'records'

interface Props {
  config: SimulateRequest
  methods: MethodInfo[]
  onChange: (c: Partial<SimulateRequest>) => void
  onRun: () => void
  loading: boolean
  position?: GamePosition | null
  calibration?: Calibration | null
  simMode?: SimMode
  onSimModeChange?: (m: SimMode) => void
}

function policyLabel(id: string): string {
  if (id === 'rule_based') return 'Rule-Based'
  if (id === 'random') return 'Random'
  if (id.startsWith('mcts_rule')) return 'MCTS+Rule'
  if (id.startsWith('mcts')) return 'MCTS'
  return id
}

const SIM_PRESETS = [
  { label: '1K',   value: 1_000,  tag: 'Fast',    config: { n_simulations: 1_000,  target_margin: undefined } },
  { label: '5K',   value: 5_000,  tag: 'Default', config: { n_simulations: 5_000,  target_margin: undefined } },
  { label: '10K',  value: 10_000, tag: 'Precise', config: { n_simulations: 10_000, target_margin: undefined } },
  { label: 'Auto', value: 0,      tag: '±2% CI',  config: { n_simulations: 5_000,  target_margin: 0.02      } },
] as const

export default function Controls({ config, methods, onChange, onRun, loading, position, calibration, simMode = 'winprob', onSimModeChange }: Props) {
  const isAuto = config.target_margin !== undefined

  const progress = positionProgress(position ?? null)
  const etaMs = estimateMs(config.policy, config.n_simulations, config.target_margin, progress, calibration)
  const etaShort = formatEta(etaMs)
  const etaLong  = formatEtaLong(etaMs)

  // Live elapsed-time counter while a run is in flight, so long MCTS runs don't
  // look frozen. Ticks once a second; cleared when loading flips back to false.
  const [elapsedMs, setElapsedMs] = useState(0)
  useEffect(() => {
    if (!loading) { setElapsedMs(0); return }
    const start = performance.now()
    setElapsedMs(0)
    const id = window.setInterval(() => setElapsedMs(performance.now() - start), 500)
    return () => window.clearInterval(id)
  }, [loading])

  return (
    <div className="controls-panel">
      <div className="panel-title">Simulation</div>

      {/* Preset buttons */}
      <div className="preset-label">Simulations</div>
      <div className="preset-row">
        {SIM_PRESETS.map(p => {
          const active = p.value === 0
            ? isAuto
            : !isAuto && config.n_simulations === p.value
          const presetMs = estimateMs(
            config.policy,
            p.config.n_simulations ?? config.n_simulations,
            p.config.target_margin,
            progress,
            calibration,
          )
          return (
            <button
              key={p.label}
              className={`preset-btn ${active ? 'active' : ''}`}
              onClick={() => onChange(p.config)}
              title={`${p.tag} — ${formatEtaLong(presetMs)} with ${policyLabel(config.policy)}`}
            >
              <span className="preset-num">{p.label}</span>
              <span className="preset-tag">{p.tag}</span>
              <span className="preset-eta">{formatEta(presetMs)}</span>
            </button>
          )
        })}
      </div>

      {/* Policy toggle */}
      <div className="preset-label">Policy</div>
      <div className="policy-row">
        {([
          { id: 'rule_based', label: '⚙️ Rule-Based', title: 'Heuristic expert policy — fastest.' },
          { id: 'random',     label: '🎲 Random',     title: 'Uniform-random — baseline for variance checks.' },
          { id: 'mcts',       label: '🌲 MCTS',       title: 'Flat-UCB MCTS with random rollouts (Szita & Chaslot).' },
          { id: 'mcts_rule',  label: '🌳 MCTS+Rule',  title: 'MCTS with rule-based rollouts — coalition vs leader emerges in the tree. Slower.' },
        ] as const).map(p => {
          const pMs = estimateMs(p.id, config.n_simulations, config.target_margin, progress, calibration)
          return (
            <button
              key={p.id}
              className={`policy-btn ${config.policy === p.id ? 'active' : ''}`}
              onClick={() => onChange({ policy: p.id })}
              title={`${p.title}  ·  ${formatEtaLong(pMs)} at current sim count`}
            >
              <span>{p.label}</span>
              <span className="policy-eta">{formatEta(pMs)}</span>
            </button>
          )
        })}
      </div>

      {/* Coalition pressure — only affects rule_based and mcts_rule */}
      {(config.policy === 'rule_based' || config.policy === 'mcts_rule') && (
        <>
          <div className="preset-label">
            Coalition pressure
            <span className="preset-label-value">{config.coalition_pressure.toFixed(2)}</span>
          </div>
          <div className="coalition-row">
            <input
              type="range"
              min={0}
              max={2}
              step={0.25}
              value={config.coalition_pressure}
              onChange={e => onChange({ coalition_pressure: parseFloat(e.target.value) })}
              className="coalition-slider"
              title="0 = independent best-response; 1 = default; 2 = strong focus-fire against the VP leader. Multi-player GTO win-prob is the band you get by sweeping this."
            />
            <div className="coalition-ticks">
              <span>selfish</span><span>default</span><span>focus-fire</span>
            </div>
          </div>
        </>
      )}

      {/* Estimator method — only shown when multiple methods are available */}
      {methods.length > 1 && (
        <>
          <div className="preset-label">Method</div>
          <div className="policy-row">
            {methods.map(m => {
              // We don't yet have per-method timing profiles; until a second
              // estimator ships with measured throughput, assume each method
              // matches the current policy+n cost. Swap in a per-method factor
              // here once we can measure a second estimator's ms/game.
              const mMs = estimateMs(config.policy, config.n_simulations, config.target_margin, progress, calibration)
              return (
                <button
                  key={m.id}
                  className={`policy-btn ${config.method === m.id ? 'active' : ''}`}
                  onClick={() => onChange({ method: m.id })}
                  title={`${m.description}  ·  ${formatEtaLong(mMs)} at current sim count`}
                >
                  <span>{m.name}</span>
                  <span className="policy-eta">{formatEta(mMs)}</span>
                </button>
              )
            })}
          </div>
        </>
      )}

      {/* Output mode */}
      {onSimModeChange && (
        <>
          <div className="preset-label">Output</div>
          <div className="sim-mode-row">
            <button
              className={`sim-mode-btn${simMode === 'winprob' ? ' active' : ''}`}
              onClick={() => onSimModeChange('winprob')}
              title="Aggregate win probabilities with 95% CI bars"
            >
              📊 Win Prob
            </button>
            <button
              className={`sim-mode-btn${simMode === 'records' ? ' active' : ''}`}
              onClick={() => onSimModeChange('records')}
              title="Per-game statistics: turn distribution, seat averages, LR/LA rates"
            >
              📋 Game Stats
            </button>
          </div>
        </>
      )}

      {/* Options row */}
      <div className="options-row">
        <label className="toggle-label">
          <input
            type="checkbox"
            checked={config.antithetic}
            onChange={e => onChange({ antithetic: e.target.checked })}
            className="toggle-input"
          />
          <span className="toggle-text">Antithetic variates</span>
        </label>
      </div>

      {/* Run button */}
      <button
        className={`run-btn ${loading ? 'loading' : ''}`}
        onClick={onRun}
        disabled={loading}
        title={loading
          ? `Running… ${formatElapsed(elapsedMs)} elapsed (est ${etaLong})`
          : `Estimated run time: ${etaLong}`}
      >
        {loading ? (
          <>
            <span className="spinner" /> Simulating…
            <span className="run-eta">{formatElapsed(elapsedMs)} / {etaShort.replace(/^~/, '')}</span>
          </>
        ) : (
          <><span className="run-icon">▶</span> Run Simulation <span className="run-eta">({etaShort})</span></>
        )}
      </button>

      {/* Keyboard hint */}
      <div className="hint-text">Press Enter to run</div>
    </div>
  )
}
