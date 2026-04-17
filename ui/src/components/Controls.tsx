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
        {(['rule_based', 'random'] as const).map(p => (
          <button
            key={p}
            className={`policy-btn ${config.policy === p ? 'active' : ''}`}
            onClick={() => onChange({ policy: p })}
          >
            {p === 'rule_based' ? '⚙️ Rule-Based' : '🎲 Random'}
          </button>
        ))}
      </div>

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
