import type { MethodInfo, SimulateRequest } from '../types'

interface Props {
  config: SimulateRequest
  methods: MethodInfo[]
  onChange: (c: Partial<SimulateRequest>) => void
  onRun: () => void
  loading: boolean
}

const SIM_PRESETS = [
  { label: '1K',   value: 1_000,  tag: 'Fast',    config: { n_simulations: 1_000,  target_margin: undefined } },
  { label: '5K',   value: 5_000,  tag: 'Default', config: { n_simulations: 5_000,  target_margin: undefined } },
  { label: '10K',  value: 10_000, tag: 'Precise', config: { n_simulations: 10_000, target_margin: undefined } },
  { label: 'Auto', value: 0,      tag: '±2% CI',  config: { n_simulations: 5_000,  target_margin: 0.02      } },
] as const

export default function Controls({ config, methods, onChange, onRun, loading }: Props) {
  const isAuto = config.target_margin !== undefined

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
          return (
            <button
              key={p.label}
              className={`preset-btn ${active ? 'active' : ''}`}
              onClick={() => onChange(p.config)}
            >
              <span className="preset-num">{p.label}</span>
              <span className="preset-tag">{p.tag}</span>
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
        ] as const).map(p => (
          <button
            key={p.id}
            className={`policy-btn ${config.policy === p.id ? 'active' : ''}`}
            onClick={() => onChange({ policy: p.id })}
            title={p.title}
          >
            {p.label}
          </button>
        ))}
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
            {methods.map(m => (
              <button
                key={m.id}
                className={`policy-btn ${config.method === m.id ? 'active' : ''}`}
                onClick={() => onChange({ method: m.id })}
                title={m.description}
              >
                {m.name}
              </button>
            ))}
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
      >
        {loading ? (
          <><span className="spinner" /> Simulating…</>
        ) : (
          <><span className="run-icon">▶</span> Run Simulation</>
        )}
      </button>

      {/* Keyboard hint */}
      <div className="hint-text">Press Enter to run</div>
    </div>
  )
}
