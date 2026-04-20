import type { GamePosition } from '../types'
import { PLAYER_COLORS, PLAYER_NAMES } from '../types'

interface Props {
  position: GamePosition
  onChange: (p: GamePosition) => void
  onOpenCards: (player: number) => void
  /** Who currently holds Largest Army (−1 if no one qualifies). */
  largestArmyPlayer: number
  /** Win-probability per player (0..1) for the last sim, if any. */
  winProbs?: number[] | null
}

const RESOURCE_EMOJI = ['🧱', '🌲', '🌾', '⛰️', '🐑']
const DEV_EMOJI = ['⚔️', '⭐', '🛣️', '🌽', '💰']

export default function PlayerBar({
  position, onChange, onOpenCards, largestArmyPlayer, winProbs,
}: Props) {
  const visibleVP = (p: number) =>
    (position.settlements[p]?.length ?? 0) + (position.cities[p]?.length ?? 0) * 2
  const totalVP = (p: number) =>
    visibleVP(p) + (position.vp_hidden[p] ?? 0) + (p === largestArmyPlayer ? 2 : 0)
  const resCount = (p: number) => (position.resources[p] ?? []).reduce((a, b) => a + b, 0)
  const devCount = (p: number) => (position.unplayed_dev[p] ?? []).reduce((a, b) => a + b, 0)

  return (
    <div className="player-bar">
      {[0, 1, 2, 3].map(p => {
        const isOnMove = position.current_player === p
        const color = PLAYER_COLORS[p]
        const vp = totalVP(p)
        const hidden = position.vp_hidden[p] ?? 0
        const knights = position.knights_played[p] ?? 0
        const roads = position.roads[p]?.length ?? 0
        const prob = winProbs?.[p]
        return (
          <div
            key={p}
            className={`player-card ${isOnMove ? 'on-move' : ''}`}
            style={{ borderColor: color }}
            onClick={() => onChange({ ...position, current_player: p })}
            title={isOnMove ? `${PLAYER_NAMES[p]} is on the move` : `Set ${PLAYER_NAMES[p]} as on the move`}
          >
            <div className="player-card-stripe" style={{ background: color }} />

            <div className="player-card-head">
              <span className="player-card-name" style={{ color }}>{PLAYER_NAMES[p]}</span>
              {isOnMove && <span className="player-on-move-chip">• ON MOVE</span>}
              <button
                className="player-edit-btn"
                onClick={e => { e.stopPropagation(); onOpenCards(p) }}
                title={`Edit ${PLAYER_NAMES[p]}'s hand & dev cards`}
              >
                ✎
              </button>
            </div>

            <div className="player-card-vp-row">
              <div className="player-vp-big">{vp}</div>
              <div className="player-vp-label">
                <span>VP</span>
                {hidden > 0 && <span className="player-vp-hidden">({hidden} hidden)</span>}
              </div>
              {prob !== undefined && (
                <div className="player-winprob" title="Win probability (most recent sim)">
                  {(prob * 100).toFixed(1)}%
                </div>
              )}
            </div>

            <div className="player-card-pieces">
              <span title="Settlements">🏠 {position.settlements[p]?.length ?? 0}</span>
              <span title="Cities">🏰 {position.cities[p]?.length ?? 0}</span>
              <span title="Roads">🛣 {roads}</span>
              {knights > 0 && (
                <span title="Knights played" className={largestArmyPlayer === p ? 'player-la-active' : ''}>
                  🛡 {knights}{largestArmyPlayer === p ? ' LA' : ''}
                </span>
              )}
            </div>

            <div className="player-card-cards">
              <span className="player-hand-badge" title="Resource cards in hand">
                🎴 <strong>{resCount(p)}</strong>
              </span>
              {devCount(p) > 0 && (
                <span className="player-hand-badge" title="Unplayed dev cards">
                  🃏 <strong>{devCount(p)}</strong>
                </span>
              )}
              {resCount(p) > 0 && (
                <div className="player-resource-mini">
                  {RESOURCE_EMOJI.map((icon, i) => {
                    const n = position.resources[p]?.[i] ?? 0
                    return n > 0 ? (
                      <span key={i} className="player-res-pip" title={`${n}`}>
                        {icon}{n > 1 ? <sub>{n}</sub> : null}
                      </span>
                    ) : null
                  })}
                </div>
              )}
              {devCount(p) > 0 && (
                <div className="player-resource-mini player-dev-mini">
                  {DEV_EMOJI.map((icon, i) => {
                    const n = position.unplayed_dev[p]?.[i] ?? 0
                    return n > 0 ? (
                      <span key={i} className="player-res-pip" title={`${n}`}>
                        {icon}{n > 1 ? <sub>{n}</sub> : null}
                      </span>
                    ) : null
                  })}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
