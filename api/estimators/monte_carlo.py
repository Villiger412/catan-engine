"""Monte Carlo estimator — wraps the Rust PyO3 simulation engine."""
from __future__ import annotations
from .base import EstimateInput, EstimateOutput

try:
    import catan_engine as _engine
    _ENGINE_OK = True
except ImportError:
    _ENGINE_OK = False


class MonteCarloEstimator:
    name = "Monte Carlo"
    description = "Rust-powered Monte Carlo simulation via PyO3"

    def estimate(self, inp: EstimateInput) -> EstimateOutput:
        if not _ENGINE_OK:
            raise RuntimeError("catan_engine not compiled — run `maturin build --release`")

        board_json = inp.board_json
        has_board = board_json is not None
        has_position = inp.position_json is not None

        if has_position:
            # Mid-game: simulate from current piece placement
            # Board is required alongside position; fall back to beginner if not supplied
            bj = board_json if has_board else _engine.get_board_layout()
            if inp.target_margin is not None:
                r = _engine.simulate_from_position_converged(
                    bj, inp.position_json,
                    target_margin=inp.target_margin,
                    policy=inp.policy,
                    seed=inp.seed,
                )
            else:
                r = _engine.simulate_from_position(
                    bj, inp.position_json,
                    n_simulations=inp.n_simulations,
                    policy=inp.policy,
                    antithetic=inp.antithetic,
                    seed=inp.seed,
                )
        elif has_board:
            # Custom board, fresh game
            if inp.target_margin is not None:
                r = _engine.simulate_until_converged_with_board(
                    board_json,
                    target_margin=inp.target_margin,
                    policy=inp.policy,
                    seed=inp.seed,
                )
            else:
                r = _engine.simulate_batch_with_board(
                    board_json,
                    n_simulations=inp.n_simulations,
                    policy=inp.policy,
                    antithetic=inp.antithetic,
                    seed=inp.seed,
                )
        else:
            # Default beginner board, fresh game
            if inp.target_margin is not None:
                r = _engine.simulate_until_converged(
                    target_margin=inp.target_margin,
                    policy=inp.policy,
                    seed=inp.seed,
                )
            else:
                r = _engine.simulate_batch(
                    n_simulations=inp.n_simulations,
                    policy=inp.policy,
                    antithetic=inp.antithetic,
                    seed=inp.seed,
                )

        return EstimateOutput(
            probabilities=r.probabilities,
            ci_lower=r.ci_lower,
            ci_upper=r.ci_upper,
            simulations_run=r.simulations_run,
            elapsed_ms=r.elapsed_ms,
            games_per_sec=r.games_per_sec,
            avg_turns=r.avg_turns,
            draws=r.draws,
            policy=r.policy,
            max_margin=r.max_margin,
        )
