import { useCallback, useEffect, useState } from 'react'
import HexBoard from './components/HexBoard'
import WinPanel from './components/WinPanel'
import Controls from './components/Controls'
import { BEGINNER_BOARD } from './lib/boardData'
import type { BoardData, GamePosition, MethodInfo, SimResult, SimulateRequest } from './types'
import './App.css'

const DEFAULT_CONFIG: SimulateRequest = {
  n_simulations: 5000,
  policy: 'rule_based',
  antithetic: true,
  seed: 42,
  method: 'monte_carlo',
}

const DEFAULT_POSITION: GamePosition = {
  settlements: [[], [], [], []],
  cities: [[], [], [], []],
  current_player: 0,
}

export default function App() {
  const [serverBoard, setServerBoard] = useState<BoardData>(BEGINNER_BOARD)
  const [board, setBoard] = useState<BoardData>(BEGINNER_BOARD)
  const [methods, setMethods] = useState<MethodInfo[]>([])
  const [config, setConfig] = useState<SimulateRequest>(DEFAULT_CONFIG)
  const [position, setPosition] = useState<GamePosition>(DEFAULT_POSITION)
  const [result, setResult] = useState<SimResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [editMode, setEditMode] = useState(false)
  const [piecesMode, setPiecesMode] = useState(false)
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
  const hasPosition = position.settlements.some(p => p.length > 0) || position.cities.some(p => p.length > 0)

  const playerVP = [0, 1, 2, 3].map(p =>
    (position.settlements[p]?.length ?? 0) + (position.cities[p]?.length ?? 0) * 2
  )

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
      const body: Record<string, unknown> = { ...config }
      if (isCustomBoard) body.board = board
      if (hasPosition) body.position = position
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
      setResult(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [config, board, isCustomBoard, position, hasPosition])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && !loading) runSimulation()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [loading, runSimulation])

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-logo">
          <span className="logo-hex">⬡</span>
          <span className="logo-title">Catan Win-Probability Engine</span>
        </div>
        <div className="header-actions">
          <button
            className="randomize-btn"
            onClick={randomizeBoard}
            disabled={randomizing}
            title="Generate a random legal Catan board"
          >
            {randomizing ? '...' : '🎲 Randomize'}
          </button>
          <button
            className={`edit-btn ${editMode ? 'active' : ''}`}
            onClick={() => { setEditMode(v => !v); setPiecesMode(false) }}
            title={editMode ? 'Exit board editor' : 'Edit board tiles'}
          >
            {editMode ? '✓ Editing' : '✏️ Edit Board'}
          </button>
          <button
            className={`pieces-btn ${piecesMode ? 'active' : ''}`}
            onClick={() => { setPiecesMode(v => !v); setEditMode(false) }}
            title={piecesMode ? 'Exit pieces mode' : 'Place settlements & cities'}
          >
            {piecesMode ? '✓ Placing' : '🏠 Pieces'}
          </button>
          {isCustomBoard && (
            <button className="reset-btn" onClick={() => setBoard(serverBoard)} title="Reset to standard board">
              ↺ Board
            </button>
          )}
          {hasPosition && (
            <button className="reset-btn" onClick={() => setPosition(DEFAULT_POSITION)} title="Clear all pieces">
              ↺ Pieces
            </button>
          )}
          <div className="header-meta">Phase 4 · Live Position</div>
        </div>
      </header>

      <main className="app-main">
        <section className="board-section">
          <HexBoard
            board={board}
            onBoardChange={editMode ? setBoard : undefined}
            position={position}
            onPositionChange={piecesMode ? setPosition : undefined}
          />
        </section>

        <aside className="side-panel">
          <Controls
            config={config}
            methods={methods}
            onChange={patch => setConfig(c => ({ ...c, ...patch }))}
            onRun={runSimulation}
            loading={loading}
          />

          {isCustomBoard && !editMode && (
            <div className="custom-board-badge">
              ⬡ Custom board active
            </div>
          )}

          {hasPosition && (
            <div className="position-panel">
              <div className="panel-title">Current Position</div>
              <div className="vp-row">
                {(['Red', 'Blue', 'Green', 'Orange'] as const).map((name, p) => (
                  <div key={p} className="vp-cell" style={{ borderColor: ['#e74c3c','#3498db','#2ecc71','#f39c12'][p] }}>
                    <span className="vp-name">{name}</span>
                    <span className="vp-count">{playerVP[p]} VP</span>
                    <span className="vp-detail">
                      {position.settlements[p].length}S {position.cities[p].length}C
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="error-banner">
              <strong>Error:</strong> {error}
              <br />
              <small>Is the API server running? <code>cd api && uvicorn main:app --reload</code></small>
            </div>
          )}

          <WinPanel result={result} loading={loading} />
        </aside>
      </main>
    </div>
  )
}
