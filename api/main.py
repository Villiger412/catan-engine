"""
Catan Win-Probability Engine — FastAPI backend
Wraps the PyO3 Rust bindings and serves the React frontend in production.
"""

import dataclasses
import json
import sys
import statistics
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Rust extension import ──────────────────────────────────────────────────────
# Add the catan-engine root so Python can find the compiled .pyd
ENGINE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # Add api/ for estimators module

# ── Estimator registry ────────────────────────────────────────────────────────
from estimators import ESTIMATORS, EstimateInput  # noqa: E402

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Catan Win-Probability Engine", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache static board layout once at startup ─────────────────────────────────
_BOARD_CACHE: dict | None = None
try:
    import catan_engine  # type: ignore
    _BOARD_CACHE = json.loads(catan_engine.get_board_layout())
except Exception as _e:
    print(f"[WARN] catan_engine not available: {_e} — run `maturin build --release`")


# ── Request / response models ─────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    n_simulations: int = Field(5000, ge=100, le=100_000)
    policy: str = Field("rule_based", pattern="^(rule_based|random|mcts|mcts_\\d+|mcts_rule|mcts_rule_\\d+)$")
    antithetic: bool = True
    seed: int = 42
    target_margin: Optional[float] = Field(None, gt=0.001, lt=0.5)
    method: str = Field("monte_carlo", description="Estimator key from /api/methods")
    board: Optional[dict] = Field(None, description="Custom board from the editor; None = beginner board")
    position: Optional[dict] = Field(None, description="Mid-game piece placement; None = fresh game")
    coalition_pressure: float = Field(
        1.0, ge=0.0, le=3.0,
        description="How aggressively opponents target the VP leader (0 = selfish, 1 = default, >1 = strong focus-fire). "
                    "Real GTO win-prob for a position is the band you get by sweeping this.",
    )


class SimulateResponse(BaseModel):
    probabilities: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    simulations_run: int
    elapsed_ms: float
    games_per_sec: float
    avg_turns: float
    draws: int
    policy: str
    max_margin: float


class RecordsRequest(BaseModel):
    n_games: int = Field(2000, ge=100, le=50_000)
    policy: str = Field("rule_based", pattern="^(rule_based|random|mcts|mcts_\\d+|mcts_rule|mcts_rule_\\d+)$")
    antithetic: bool = True
    seed: int = 42
    board: Optional[dict] = Field(None, description="Custom board; None = beginner board")
    position: Optional[dict] = Field(None, description="Mid-game position; None = fresh game")
    coalition_pressure: float = Field(1.0, ge=0.0, le=3.0)


class SeatStats(BaseModel):
    win_rate: float
    avg_vp: float
    avg_cities: float
    avg_settlements: float
    avg_roads: float
    avg_knights: float
    lr_rate: float
    la_rate: float


class RecordsResponse(BaseModel):
    games_run: int
    elapsed_ms: float
    games_per_sec: float
    policy: str
    turns_min: int
    turns_p25: float
    turns_median: float
    turns_mean: float
    turns_p75: float
    turns_max: int
    seat_stats: list[SeatStats]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "engine": _BOARD_CACHE is not None}


@app.get("/api/board")
def get_board():
    if _BOARD_CACHE is None:
        raise HTTPException(503, "Engine not compiled — run `maturin build --release`")
    return _BOARD_CACHE


@app.get("/api/random-board")
def get_random_board(seed: Optional[int] = None):
    try:
        import catan_engine  # type: ignore
    except ImportError:
        raise HTTPException(503, "Engine not compiled — run `maturin build --release`")
    import random
    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    return json.loads(catan_engine.get_random_board_layout(actual_seed))


@app.get("/api/methods")
def list_methods():
    """List all available win-probability estimators."""
    return [
        {"id": k, "name": v.name, "description": v.description}
        for k, v in ESTIMATORS.items()
    ]


@app.post("/api/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    est = ESTIMATORS.get(req.method)
    if est is None:
        raise HTTPException(400, f"Unknown method '{req.method}'. Available: {list(ESTIMATORS)}")

    inp = EstimateInput(
        n_simulations=req.n_simulations,
        policy=req.policy,
        antithetic=req.antithetic,
        seed=req.seed,
        target_margin=req.target_margin,
        board_json=json.dumps(req.board) if req.board is not None else None,
        position_json=json.dumps(req.position) if req.position is not None else None,
        coalition_pressure=req.coalition_pressure,
    )
    try:
        out = est.estimate(inp)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return SimulateResponse(**dataclasses.asdict(out))


@app.post("/api/simulate-records", response_model=RecordsResponse)
def simulate_records(req: RecordsRequest):
    try:
        import catan_engine  # type: ignore
    except ImportError:
        raise HTTPException(503, "Engine not compiled — run `maturin build --release`")

    try:
        r = catan_engine.simulate_games_records(
            n_games=req.n_games,
            strategy=req.policy,
            antithetic=req.antithetic,
            seed=req.seed,
            coalition_pressure=req.coalition_pressure,
            board_json=json.dumps(req.board) if req.board is not None else None,
            position_json=json.dumps(req.position) if req.position is not None else None,
        )
    except Exception as e:
        raise HTTPException(503, str(e))

    n = len(r.winner)
    if n == 0:
        raise HTTPException(503, "No games completed")

    turns_sorted = sorted(r.turns)

    def pct(lst: list, p: float) -> float:
        idx = p / 100.0 * (len(lst) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(lst) - 1)
        return lst[lo] + (lst[hi] - lst[lo]) * (idx - lo)

    seat_stats = []
    for s in range(4):
        wins = sum(1 for w in r.winner if w == s)
        vp_vals    = [r.vp_total[i][s]      for i in range(n)]
        city_vals  = [r.cities[i][s]        for i in range(n)]
        set_vals   = [r.settlements[i][s]   for i in range(n)]
        road_vals  = [r.roads[i][s]         for i in range(n)]
        knt_vals   = [r.knights_played[i][s] for i in range(n)]
        lr_count   = sum(1 for i in range(n) if r.longest_road_player[i] == s)
        la_count   = sum(1 for i in range(n) if r.largest_army_player[i] == s)
        seat_stats.append(SeatStats(
            win_rate=wins / n,
            avg_vp=statistics.mean(vp_vals),
            avg_cities=statistics.mean(city_vals),
            avg_settlements=statistics.mean(set_vals),
            avg_roads=statistics.mean(road_vals),
            avg_knights=statistics.mean(knt_vals),
            lr_rate=lr_count / n,
            la_rate=la_count / n,
        ))

    return RecordsResponse(
        games_run=n,
        elapsed_ms=r.elapsed_ms,
        games_per_sec=r.games_per_sec,
        policy=r.policy_label,
        turns_min=int(turns_sorted[0]),
        turns_p25=pct(turns_sorted, 25),
        turns_median=pct(turns_sorted, 50),
        turns_mean=statistics.mean(r.turns),
        turns_p75=pct(turns_sorted, 75),
        turns_max=int(turns_sorted[-1]),
        seat_stats=seat_stats,
    )


def _records_result_to_response(r, n_games_hint: int) -> RecordsResponse:
    """Shared projection from a GameRecordsResult into RecordsResponse."""
    n = len(r.winner)
    if n == 0:
        raise HTTPException(503, "No games completed")
    turns_sorted = sorted(r.turns)

    def pct(lst, p):
        idx = p / 100.0 * (len(lst) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(lst) - 1)
        return lst[lo] + (lst[hi] - lst[lo]) * (idx - lo)

    seat_stats = []
    for s in range(4):
        wins = sum(1 for w in r.winner if w == s)
        seat_stats.append(SeatStats(
            win_rate=wins / n,
            avg_vp=statistics.mean(r.vp_total[i][s] for i in range(n)),
            avg_cities=statistics.mean(r.cities[i][s] for i in range(n)),
            avg_settlements=statistics.mean(r.settlements[i][s] for i in range(n)),
            avg_roads=statistics.mean(r.roads[i][s] for i in range(n)),
            avg_knights=statistics.mean(r.knights_played[i][s] for i in range(n)),
            lr_rate=sum(1 for i in range(n) if r.longest_road_player[i] == s) / n,
            la_rate=sum(1 for i in range(n) if r.largest_army_player[i] == s) / n,
        ))

    return RecordsResponse(
        games_run=n,
        elapsed_ms=r.elapsed_ms,
        games_per_sec=r.games_per_sec,
        policy=r.policy_label,
        turns_min=int(turns_sorted[0]),
        turns_p25=pct(turns_sorted, 25),
        turns_median=pct(turns_sorted, 50),
        turns_mean=statistics.mean(r.turns),
        turns_p75=pct(turns_sorted, 75),
        turns_max=int(turns_sorted[-1]),
        seat_stats=seat_stats,
    )


class StrategyRequest(BaseModel):
    code: str = Field(..., description="Python source defining a Strategy subclass")
    class_name: str = Field(..., min_length=1, max_length=100)
    n_games: int = Field(1000, ge=100, le=20_000)
    seed: int = 42
    antithetic: bool = True
    coalition_pressure: float = Field(1.0, ge=0.0, le=3.0)
    board: Optional[dict] = None
    position: Optional[dict] = None


@app.post("/api/run-strategy", response_model=RecordsResponse)
def run_strategy(req: StrategyRequest):
    try:
        import catan_engine  # type: ignore
        import catan_research  # type: ignore  # noqa: F401 — ensures Strategy is importable
    except ImportError as e:
        raise HTTPException(503, f"Engine not compiled or catan_research missing: {e}")

    # Compile and execute submitted code in a sandboxed namespace.
    try:
        code = compile(req.code, "<strategy>", "exec")
    except SyntaxError as e:
        raise HTTPException(400, f"Syntax error: {e}")

    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102
    except Exception as e:
        raise HTTPException(400, f"Error executing strategy code: {e}")

    if req.class_name not in ns:
        raise HTTPException(400, f"Class '{req.class_name}' not found in submitted code")

    cls = ns[req.class_name]
    try:
        strategy_instance = cls()
    except Exception as e:
        raise HTTPException(400, f"Could not instantiate '{req.class_name}': {e}")

    try:
        r = catan_engine.simulate_games_records(
            n_games=req.n_games,
            strategy=strategy_instance,
            antithetic=req.antithetic,
            seed=req.seed,
            coalition_pressure=req.coalition_pressure,
            board_json=json.dumps(req.board) if req.board is not None else None,
            position_json=json.dumps(req.position) if req.position is not None else None,
        )
    except Exception as e:
        raise HTTPException(503, f"Simulation error: {e}")

    return _records_result_to_response(r, req.n_games)


# ── Serve React build in production ───────────────────────────────────────────
UI_DIST = Path(__file__).parent.parent / "ui" / "dist"
if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(UI_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str = ""):
        index = UI_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(404, "UI not built — run `npm run build` in ui/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
