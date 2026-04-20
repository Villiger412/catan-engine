"""
Estimator abstraction — add a new win-probability method by implementing WinEstimator.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class EstimateInput:
    n_simulations: int
    policy: str
    antithetic: bool
    seed: int
    target_margin: float | None
    board_json: str | None = None      # None → beginner board
    position_json: str | None = None  # None → fresh game (setup phase); otherwise mid-game
    coalition_pressure: float = 1.0    # 0 = selfish, 1 = default VP-targeting, >1 = strong focus-fire


@dataclass
class EstimateOutput:
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


@runtime_checkable
class WinEstimator(Protocol):
    name: str
    description: str

    def estimate(self, inp: EstimateInput) -> EstimateOutput: ...
