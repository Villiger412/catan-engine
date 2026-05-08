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
  const [simMode] = useState<SimMode>('winprob')
  const [mode, setMode] = useState<BoardMode>('view')
  const [cardsFocusPlayer, setCardsFocusPlayer] = useState<number | null>(null)
  const [randomizing, setRandomizing] = useState(false)
  const [showInfo, setShowInfo] = useState(false)

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
          <button
            className="info-btn"
            onClick={() => setShowInfo(true)}
            title="How to use Catan Engine"
            aria-label="How to use Catan Engine"
          >
            i
          </button>
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

      {showInfo && (
        <div className="cards-modal-backdrop" onClick={() => setShowInfo(false)}>
          <div className="cards-modal info-modal" onClick={e => e.stopPropagation()}>
            <header className="cards-modal-head">
              <div className="cards-modal-title">How to use Catan Engine</div>
              <button
                className="cards-modal-close"
                onClick={() => setShowInfo(false)}
                title="Close"
                aria-label="Close"
              >
                ✕
              </button>
            </header>
            <div className="info-body">
              <section>
                <h3>What this is</h3>
                <p>
                  A sandbox for estimating <strong>win probabilities</strong> in 2-4 player Catan.
                  Set up a board and a mid-game position, run a simulation, and read the
                  per-player win probabilities with 95% confidence intervals.
                </p>
              </section>

              <section>
                <h3>Quick start</h3>
                <ol>
                  <li>The board loads with the standard beginner layout. Hit <strong>🎲</strong> in the top-right to randomize a legal board, or use <strong>Board</strong> mode to edit tiles and numbers by hand.</li>
                  <li>Switch to <strong>Pieces</strong> mode and click vertices to place settlements (click again to upgrade to a city).</li>
                  <li>Switch to <strong>Roads</strong> mode and click edges to add roads, color-cycled by player.</li>
                  <li>Use <strong>Robber</strong> mode to drop the robber on a hex.</li>
                  <li>Click a player's slot in the top bar to edit their hand of resources, dev cards, played knights, and hidden VPs.</li>
                  <li>In the right rail, pick a policy and number of simulations, then press <strong>Run</strong> (or just hit <kbd>Enter</kbd>).</li>
                  <li>The Win panel shows per-player win probability with a 95% CI. Use <strong>↺ Pieces</strong> / <strong>↺ Board</strong> to reset.</li>
                </ol>
              </section>

              <section>
                <h3>How it works</h3>
                <p>
                  A Rust core plays <strong>thousands of full games</strong> from your position in parallel,
                  using a chosen rollout policy for all four seats. The empirical win-rate per seat is
                  reported with a Wilson 95% CI; auto-mode keeps simulating until the widest per-player
                  half-width drops under your target margin.
                </p>
                <ul>
                  <li><strong>rule_based</strong> — a tuned heuristic policy (city &gt; settlement &gt; knight &gt; trade &gt; road…). Fastest realistic baseline.</li>
                  <li><strong>random</strong> — uniform over legal actions. Diagnostic only.</li>
                  <li><strong>mcts_N</strong> — 1-ply UCB1 search with N rollouts per move. Slower but more tactical.</li>
                  <li><strong>Coalition pressure</strong> — at 0 the policy is selfish; at 1 it weighs VP roughly as much as production when picking robber/steal targets; at 2 it focus-fires the leader.</li>
                </ul>
              </section>

              <section>
                <h3>Reading the numbers honestly</h3>
                <p>
                  Catan is stochastic, has hidden info, and is general-sum with 2-4 players — there is no
                  single "GTO" win probability. Every result here is the win-rate under <em>the specific
                  joint policy profile you chose</em>. Two well-converged runs with different policies can
                  legitimately disagree by 10-20pp on the same position. Treat the answer as a band, not a
                  point estimate, and try a coalition-pressure sweep if you care how robust it is.
                </p>
              </section>

              <section>
                <h3>About this project</h3>
                <p>
                  Catan Engine is a small passion project I built and run on my own. I pay for the
                  hosting out of pocket, and there are <strong>no ads, no tracking, and no plans to
                  monetize</strong> — I just enjoy the game and wanted a tool that takes its
                  game-theoretic structure seriously.
                </p>
                <p>
                  If it's useful to you and you'd like to help cover the server costs, donations are
                  very welcome via the <strong>☕ Buy me a coffee</strong> button in the top-right.
                  No pressure at all — using the site freely is just as appreciated.
                </p>
              </section>
            </div>
          </div>
        </div>
      )}

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
