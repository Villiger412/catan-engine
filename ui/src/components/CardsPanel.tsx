import type { GamePosition } from '../types'
import { PLAYER_COLORS, PLAYER_NAMES } from '../types'

interface Props {
  position: GamePosition
  onChange: (p: GamePosition) => void
  /** If set, show only that player's row. */
  focusPlayer?: number | null
}

const RESOURCE_LABELS = ['🧱', '🌲', '🌾', '⛰️', '🐑'] as const
const DEV_LABELS      = ['⚔️', '⭐', '🛣️', '🌽', '💰'] as const
const RESOURCE_TITLES = ['Brick', 'Wood', 'Wheat', 'Ore', 'Sheep'] as const
const DEV_TITLES      = ['Knight', 'VP', 'Road Building', 'Year of Plenty', 'Monopoly'] as const

function Stepper({
  value, title, onDelta,
}: { value: number; title: string; onDelta: (d: number) => void }) {
  return (
    <div className="card-stepper" title={title}>
      <button className="stepper-btn" onClick={() => onDelta(-1)} disabled={value <= 0}>−</button>
      <span className="stepper-val">{value}</span>
      <button className="stepper-btn" onClick={() => onDelta(+1)}>+</button>
    </div>
  )
}

export default function CardsPanel({ position, onChange, focusPlayer = null }: Props) {
  const visiblePlayers = focusPlayer !== null && focusPlayer >= 0
    ? [focusPlayer]
    : [0, 1, 2, 3]
  const patch = (fn: (p: GamePosition) => GamePosition) => onChange(fn(position))

  function bumpResource(player: number, i: number, d: number) {
    patch(p => {
      const resources = p.resources.map(arr => [...arr])
      resources[player][i] = Math.max(0, resources[player][i] + d)
      return { ...p, resources }
    })
  }

  function bumpDev(player: number, i: number, d: number) {
    patch(p => {
      const unplayed_dev = p.unplayed_dev.map(arr => [...arr])
      unplayed_dev[player][i] = Math.max(0, unplayed_dev[player][i] + d)
      return { ...p, unplayed_dev }
    })
  }

  function bumpKnights(player: number, d: number) {
    patch(p => {
      const knights_played = [...p.knights_played]
      knights_played[player] = Math.max(0, knights_played[player] + d)
      return { ...p, knights_played }
    })
  }

  function bumpVpHidden(player: number, d: number) {
    patch(p => {
      const vp_hidden = [...p.vp_hidden]
      vp_hidden[player] = Math.max(0, vp_hidden[player] + d)
      return { ...p, vp_hidden }
    })
  }

  return (
    <div className="cards-panel">
      {focusPlayer === null && <div className="panel-title">Cards & Hands</div>}
      <div className="cards-hint">
        Seed the engine with known or assumed hand contents. Leave everything
        at 0 for the "pieces only" baseline.
      </div>
      {focusPlayer === null && (
        <div className="current-player-row">
          <span className="current-player-label">On move:</span>
          {[0, 1, 2, 3].map(p => (
            <button
              key={p}
              className={`current-player-btn ${position.current_player === p ? 'active' : ''}`}
              onClick={() => onChange({ ...position, current_player: p })}
              style={{
                borderColor: PLAYER_COLORS[p],
                background: position.current_player === p ? PLAYER_COLORS[p] : 'transparent',
                color: position.current_player === p ? '#fff' : PLAYER_COLORS[p],
              }}
              title={`${PLAYER_NAMES[p]} is next to act`}
            >
              {PLAYER_NAMES[p]}
            </button>
          ))}
        </div>
      )}
      {visiblePlayers.map(p => (
        <div key={p} className="cards-player-row" style={{ borderColor: PLAYER_COLORS[p] }}>
          <div className="cards-player-name" style={{ color: PLAYER_COLORS[p] }}>
            {PLAYER_NAMES[p]}
          </div>

          <div className="cards-group">
            <span className="cards-group-label">Resources</span>
            <div className="cards-group-steppers">
              {RESOURCE_LABELS.map((icon, i) => (
                <div key={i} className="card-col">
                  <span className="card-icon" title={RESOURCE_TITLES[i]}>{icon}</span>
                  <Stepper
                    value={position.resources[p][i]}
                    title={RESOURCE_TITLES[i]}
                    onDelta={d => bumpResource(p, i, d)}
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="cards-group">
            <span className="cards-group-label">Unplayed dev</span>
            <div className="cards-group-steppers">
              {DEV_LABELS.map((icon, i) => (
                <div key={i} className="card-col">
                  <span className="card-icon" title={DEV_TITLES[i]}>{icon}</span>
                  <Stepper
                    value={position.unplayed_dev[p][i]}
                    title={DEV_TITLES[i]}
                    onDelta={d => bumpDev(p, i, d)}
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="cards-group">
            <span className="cards-group-label">Extras</span>
            <div className="cards-group-steppers">
              <div className="card-col">
                <span className="card-icon" title="Knights already played (counts toward Largest Army)">🛡️</span>
                <Stepper
                  value={position.knights_played[p]}
                  title="Knights played"
                  onDelta={d => bumpKnights(p, d)}
                />
              </div>
              <div className="card-col">
                <span className="card-icon" title="Face-down VP cards (hidden)">⭐</span>
                <Stepper
                  value={position.vp_hidden[p]}
                  title="Hidden VP"
                  onDelta={d => bumpVpHidden(p, d)}
                />
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
