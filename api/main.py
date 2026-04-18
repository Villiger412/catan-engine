"""
Catan Win-Probability Engine — FastAPI backend
Wraps the PyO3 Rust bindings and serves the React frontend in production.
"""

import dataclasses
import json
import sys
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
    policy: str = Field("rule_based", pattern="^(rule_based|random|mcts|mcts_\\d+)$")
    antithetic: bool = True
    seed: int = 42
    target_margin: Optional[float] = Field(None, gt=0.001, lt=0.5)
    method: str = Field("monte_carlo", description="Estimator key from /api/methods")
    board: Optional[dict] = Field(None, description="Custom board from the editor; None = beginner board")
    position: Optional[dict] = Field(None, description="Mid-game piece placement; None = fresh game")


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
    )
    try:
        out = est.estimate(inp)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return SimulateResponse(**dataclasses.asdict(out))


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
