import { useCallback, useEffect, useState } from 'react'
import HexBoard from './components/HexBoard'
import WinPanel from './components/WinPanel'
import RecordsPanel from './components/RecordsPanel'
import StrategyPanel from './components/StrategyPanel'
import Controls from './components/Controls'
import CardsPanel from './components/CardsPanel'
import PlayerBar from './components/PlayerBar'
import { BEGINNER_BOARD } from './lib/boardData'
import type { BoardData, GamePosition, MethodInfo, RecordsResult, SimResult, SimulateRequest } from './types'
import type { Calibration } from './lib/timing'
import type { SimMode } from './components/Controls'
import { positionProgress } from './lib/timing'
import './App.css'

const DEFAULT_CONFIG: SimulateRequest = {
  n_simulations: 5000,
  policy: 'rule_based',
  antithetic: true,
  seed: 42,
  method: 'monte_carlo',
  coalition_pressure: 1.0,
}

const EMPTY_PER_PLAYER_5 = () => [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

const DEFAULT_POSITION: GamePosition = {
  settlements: [[], [], [], []],
  cities: [[], [], [], []],
  roads: [[], [], [], []],
  resources: EMPTY_PER_PLAYER_5(),
  unplayed_dev: EMPTY_PER_PLAYER_5(),
  knights_played: [0, 0, 0, 0],
  vp_hidden: [0, 0, 0, 0],
  current_player: 0,
}

type BoardMode = 'view' | 'edit' | 'pieces' | 'roads' | 'robber'

const MODE_BUTTONS: { id: BoardMode; label: string; icon: string; title: string }[] = [
  { id: 'pieces', label: 'Pieces', icon: '🏠', title: 'Click a vertex to place / cycle settlement → city' },
  { id: 'roads',  label: 'Roads',  icon: '🛣️', title: 'Click an edge to cycle road ownership' },
  { id: 'robber', label: 'Robber', icon: '🦹', title: 'Click a hex to move the robber' },
  { id: 'edit',   label: 'Board',  icon: '✏️', title: 'Edit tiles and numbers' },
]

export default function App() {
  const [serverBoard, setServerBoard] = useState<BoardData>(BEGINNER_BOARD)
  const [board, setBoard] = useState<BoardData>(BEGINNER_BOARD)
  const [methods, setMethods] = useState<MethodInfo[]>([])
  const [config, setConfig] = useState<SimulateRequest>(DEFAULT_CONFIG)
  const [position, setPosition] = useState<GamePosition>(DEFAULT_POSITION)
  const [result, setResult] = useState<SimResult | null>(null)
  const [recordsResult, setRecordsResult] = useState<RecordsResult | null>(null)
  const [calibration, setCalibration] = useState<Calibration | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [simMode, setSimMode] = useState<SimMode>('winprob')
  const [mode, setMode] = useState<BoardMode>('view')
  const [cardsFocusPlayer, setCardsFocusPlayer] = useState<number | null>(null)
  const [randomizing, setRandomizing] = useState(false)

  const randomizeBoard = useCallback(async () => {
    setRandomizing(true)
    try {
      const seed = Math.floor(Math.random() * 2 ** 32)
      const res = await fetch(`/api/random-board?seed=${seed}`)
      if (!res.ok) throw new Error(`${res.status}`)
      const data: BoardData = await res.json()
      setBoard(data)
      setServerBoard(data)
      setPosition(DEFAULT_POSITION)
    } catch {
      // silently ignore — server may not be running
    } finally {
      setRandomizing(false)
    }
  }, [])

  const isCustomBoard = JSON.stringify(board) !== JSON.stringify(serverBoard)
  const hasPosition =
    position.settlements.some(p => p.length > 0) ||
    position.cities.some(p => p.length > 0) ||
    position.roads.some(p => p.length > 0) ||
    position.resources.some(p => p.some(n => n > 0)) ||
    position.unplayed_dev.some(p => p.some(n => n > 0)) ||
    position.knights_played.some(n => n > 0) ||
    position.vp_hidden.some(n => n > 0) ||
    position.current_player !== 0 ||
    position.robber_hex !== undefined

  const knightsPlayed = [0, 1, 2, 3].map(p => position.knights_played[p] ?? 0)
  const laMax = Math.max(...knightsPlayed)
  const laPlayer = laMax >= 3 && knightsPlayed.filter(k => k === laMax).length === 1
    ? knightsPlayed.indexOf(laMax)
    : -1

  useEffect(() => {
    fetch('/api/board')
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then((data: BoardData) => { setServerBoard(data); setBoard(data) })
      .catch(() => {})

    fetch('/api/methods')
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then((data: MethodInfo[]) => {
        setMethods(data)
        if (data.length > 0) {
          setConfig(c => data.some(m => m.id === c.method) ? c : { ...c, method: data[0].id })
        }
      })
      .catch(() => {})
  }, [])

  const runSimulation = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      if (simMode === 'records') {
        const body: Record<string, unknown> = {
          n_games: config.n_simulations,
          policy: config.policy,
          antithetic: config.antithetic,
          seed: config.seed,
          coalition_pressure: config.coalition_pressure,
        }
        if (isCustomBoard) body.board = board
        if (hasPosition) body.position = position
        const res = await fetch('/api/simulate-records', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
        const data: RecordsResult = await res.json()
        setRecordsResult(data)
      } else {
        const body: Record<string, unknown> = { ...config }
        if (isCustomBoard) body.board = board
        if (hasPosition) body.position = position
        const res = await fetch('/api/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
        const data: SimResult = await res.json()
        setResult(data)

        if (data.games_per_sec > 0 && data.simulations_run > 0) {
          const progress = positionProgress(hasPosition ? position : null)
          setCalibration({
            policy: data.policy,
            msPerGame: data.elapsed_ms / data.simulations_run,
            avgTurns: data.avg_turns,
            turnsPerSec: data.avg_turns * data.games_per_sec,
            progress,
            simulationsRun: data.simulations_run,
          })
        }

        if (config.target_margin !== undefined) {
          // eslint-disable-next-line no-console
          console.log('[auto-mode converged]', {
            policy: data.policy,
            target_margin: config.target_margin,
            simulations_run: data.simulations_run,
            elapsed_ms: data.elapsed_ms,
            max_margin: data.max_margin,
            avg_turns: data.avg_turns,
            games_per_sec: data.games_per_sec,
          })
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [simMode, config, board, isCustomBoard, position, hasPosition])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (cardsFocusPlayer !== null && e.key === 'Escape') {
        setCardsFocusPlayer(null)
        return
      }
      if (e.key === 'Enter' && !loading && cardsFocusPlayer === null) runSimulation()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [loading, runSimulation, cardsFocusPlayer])

  const cycleMode = (next: BoardMode) => setMode(cur => cur === next ? 'view' : next)

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-logo">
          <span className="logo-hex">⬡</span>
          <span className="logo-title">Catan Engine</span>
          <span className="logo-sub">Win-probability sandbox</span>
        </div>
        <div className="header-actions">
          <div className="mode-group" role="group" aria-label="Board mode">
            {MODE_BUTTONS.map(b => (
              <button
                key={b.id}
                className={`mode-btn ${mode === b.id ? 'active' : ''}`}
                onClick={() => cycleMode(b.id)}
                title={b.title}
              >
                <span className="mode-icon">{b.icon}</span>
                <span className="mode-label">{b.label}</span>
              </button>
            ))}
          </div>
          <button
            className="icon-btn"
            onClick={randomizeBoard}
            disabled={randomizing}
            title="Generate a random legal Catan board"
          >
            {randomizing ? '…' : '🎲'}
          </button>
          {isCustomBoard && (
            <button className="icon-btn reset" onClick={() => setBoard(serverBoard)} title="Reset to standard board">
              ↺ Board
            </button>
          )}
          {hasPosition && (
            <button className="icon-btn reset" onClick={() => setPosition(DEFAULT_POSITION)} title="Clear all pieces, hands, and robber">
              ↺ Pieces
            </button>
          )}
          <a
            className="kofi-btn"
            href="https://ko-fi.com/idoshcatan"
            target="_blank"
            rel="noopener noreferrer"
            title="Support this project on Ko-fi"
          >
            ☕ <span className="kofi-label">Buy me a coffee</span>
          </a>
        </div>
      </header>

      <PlayerBar
        position={position}
        onChange={setPosition}
        onOpenCards={(p) => setCardsFocusPlayer(p)}
        largestArmyPlayer={laPlayer}
        winProbs={result?.probabilities ?? null}
      />

      <main className="app-main">
        <section className="board-section">
          <HexBoard
            board={board}
            onBoardChange={mode === 'edit' ? setBoard : undefined}
            position={position}
            onPositionChange={(mode === 'pieces' || mode === 'roads' || mode === 'robber') ? setPosition : undefined}
            piecesMode={mode === 'pieces'}
            roadsMode={mode === 'roads'}
            robberMode={mode === 'robber'}
          />
          {isCustomBoard && mode !== 'edit' && (
            <div className="board-overlay-badge">⬡ Custom board</div>
          )}
          {position.robber_hex !== undefined && (
            <div className="board-overlay-badge robber">🦹 Robber on hex #{position.robber_hex}</div>
          )}
        </section>

        <aside className="sim-rail">
          <Controls
            config={config}
            methods={methods}
            onChange={patch => setConfig(c => ({ ...c, ...patch }))}
            onRun={runSimulation}
            loading={loading}
            position={hasPosition ? position : null}
            calibration={calibration}
            simMode={simMode}
            onSimModeChange={setSimMode}
          />
          {error && (
            <div className="error-banner">
              <strong>Error:</strong> {error}
              <br />
              <small>Is the API server running? <code>cd api && uvicorn main:app --reload</code></small>
            </div>
          )}
          {simMode === 'records'
            ? <>
                <RecordsPanel result={recordsResult} loading={loading} />
                <StrategyPanel
                  config={config}
                  isCustomBoard={isCustomBoard}
                  board={board}
                  hasPosition={hasPosition}
                  position={position}
                  onResult={setRecordsResult}
                />
              </>
            : <WinPanel result={result} loading={loading} />
          }
        </aside>
      </main>

      {cardsFocusPlayer !== null && (
        <div className="cards-modal-backdrop" onClick={() => setCardsFocusPlayer(null)}>
          <div className="cards-modal" onClick={e => e.stopPropagation()}>
            <header className="cards-modal-head">
              <div className="cards-modal-title">Edit hand & dev cards</div>
              <button
                className="cards-modal-close"
                onClick={() => setCardsFocusPlayer(null)}
                title="Close (Esc)"
              >
                ✕
              </button>
            </header>
            <CardsPanel
              position={position}
              onChange={setPosition}
              focusPlayer={cardsFocusPlayer}
            />
          </div>
        </div>
      )}
    </div>
  )
}
