#!/usr/bin/env python3
"""
Symmetric Empirical Game-Theoretic Analysis (EGTA / PSRO) for Catan.

Goal
----
Given a pool of k candidate policies (π_1,...,π_k), find a symmetric
*meta-strategy* σ ∈ Δ^k — a mixture over the pool — that approximates a
symmetric Nash equilibrium of the restricted 4-player game where each
seat draws its pure policy from σ.

We build a k × k "empirical payoff matrix":

    M[i][j] = P(seat 0 wins | seat 0 plays π_i, seats 1..3 all play π_j)

and then run symmetric *fictitious play* (Brown 1951; Robinson 1951) on
M: at each iteration, the ego seat best-responds to the opponents'
empirical average mixed strategy, and we average the trajectory. For
symmetric payoff matrices this converges in average play to a symmetric
coarse-correlated equilibrium, and — in practice on well-conditioned
matrices — to a symmetric Nash.

This is the restriction-to-pool version of the PSRO loop of
Lanctot et al. (2017) "A Unified Game-Theoretic Approach to Multiagent
Reinforcement Learning", where the outer oracle step is omitted: the
strategy space is fixed to our hand-curated pool. The meta-value
v* = σ*ᵀ M σ* is the best win-probability for seat 0 *achievable by any
symmetric mixture over the pool*, and σ* tells us which policies the
equilibrium actually uses (most pool policies typically receive zero
mass — see Lanctot Fig. 1).

Why symmetric
-------------
Catan is symmetric in policies but *not* in seat (seat order matters
for the initial placement snake). Our engine's position JSON fixes
seat 0 as the query seat, so we estimate only the seat-0 payoff and
assume opponents are interchangeable. This is exactly the EGTA
convention of Wellman (2006).

Why this matters beyond policy_band
------------------------------------
`policy_band.py` reports min / max across the pool (an uncertainty
bound). PSRO picks the *equilibrium mixture*, which is the sharpest
single point-estimate the pool can give you under the Nash solution
concept. If σ* puts all weight on one policy, that's strong evidence
the pool has a dominant strategy. If σ* spreads mass, the true
equilibrium is genuinely mixed and no single policy is "right".

Usage
-----
    python psro.py                               # default pool, default pos
    python psro.py --pool "random,rule_based:0.0,rule_based:1.0,rule_based:2.0,mcts_200"
    python psro.py --sims 2000 --fp-iters 5000
    python psro.py --board-seed 7                # random board

No API server required — calls `catan_engine.simulate_per_seat` directly.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import catan_engine
except ImportError:
    raise SystemExit(
        "catan_engine Python module not installed. Build the Rust extension "
        "first (see README)."
    )

REPORT_PATH = Path(__file__).parent / "report" / "psro.json"

DEFAULT_POOL = "random,rule_based:0.0,rule_based:1.0,rule_based:2.0,mcts_200"


def parse_pool(spec: str):
    """Return list of (label, policy_name, coalition_pressure)."""
    out = []
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if ":" in tok:
            name, cp_s = tok.split(":", 1)
            cp = float(cp_s)
        else:
            name, cp = tok, 1.0
        label = tok if ":" in tok else (
            f"{name}:{cp}"
            if name in ("rule_based", "mcts_rule_200", "mcts_rule_500")
            else name
        )
        out.append((label, name.strip(), cp))
    return out


def build_payoff_matrix(pool, n_sims, board_json, position_json, seed, verbose):
    """Build k×k matrix M[i][j] = P(seat 0 wins | seat 0 plays i, others play j).

    Each cell is an independent Monte Carlo estimate with `n_sims` games.
    We also return the Wilson half-width (CI radius) for diagnostics.
    """
    k = len(pool)
    M = np.zeros((k, k), dtype=float)
    CI = np.zeros((k, k), dtype=float)  # half-width (ci_upper - prob)
    total_cells = k * k
    t0 = time.time()
    cell_idx = 0

    for i, (label_i, name_i, cp_i) in enumerate(pool):
        for j, (label_j, name_j, cp_j) in enumerate(pool):
            cell_idx += 1

            # Seat policies: seat 0 = row (ego), seats 1..3 = column (opp).
            # Per-seat coalition pressure is passed via `coalition_pressures`
            # so the row's cp_i and the column's cp_j are honored
            # independently (needed whenever the pool mixes rule_based
            # policies at different coalition pressures).
            seat_policies = [name_i, name_j, name_j, name_j]
            cps = [cp_i, cp_j, cp_j, cp_j]

            # Derive a unique seed per cell so cells are independent but
            # reproducible.
            cell_seed = seed + 1_000_003 * i + 101 * j

            res = catan_engine.simulate_per_seat(
                seat_policies,
                board_json=board_json,
                position_json=position_json,
                n_simulations=n_sims,
                n_threads=0,
                antithetic=True,
                seed=cell_seed,
                coalition_pressures=cps,
            )
            p = res.probabilities[0]
            ci_up = res.ci_upper[0]
            M[i, j] = p
            CI[i, j] = max(0.0, ci_up - p)

            if verbose:
                elapsed = time.time() - t0
                eta = elapsed / cell_idx * (total_cells - cell_idx)
                print(
                    f"  [{cell_idx:2d}/{total_cells}] "
                    f"M[{label_i:20s}][{label_j:20s}] = "
                    f"{p*100:5.1f} +/- {CI[i,j]*100:4.1f} pp   "
                    f"gps={res.games_per_sec:.0f}  eta={eta:.0f}s"
                )
    return M, CI


def fictitious_play(M, iters, seed=0):
    """Symmetric fictitious play on k×k matrix M (rows = ego, cols = opp).

    At iteration t:
        i_t ∈ argmax_i (M σ_{t-1})_i
        σ_t = ((t-1) σ_{t-1} + e_{i_t}) / t

    Returns σ (average play) and the exploitability trajectory.
    """
    rng = np.random.default_rng(seed)
    k = M.shape[0]
    # Start from uniform + tiny random jitter to break ties
    sigma = np.ones(k) / k
    exploit_traj = []

    for t in range(1, iters + 1):
        expected = M @ sigma  # expected row-payoff against current mixture
        best = np.argmax(expected + rng.uniform(0, 1e-12, k))
        # Averaged play
        sigma = sigma * (t - 1) / t
        sigma[best] += 1.0 / t
        # Exploitability: how much a best-response beats the current mix
        v_current = float(sigma @ M @ sigma)
        v_br = float(np.max(M @ sigma))
        exploit_traj.append(v_br - v_current)

    return sigma, exploit_traj


def symmetric_nash_report(pool, M, CI, sigma, exploit):
    """Print a human-readable summary of the equilibrium."""
    k = len(pool)
    labels = [l for l, _, _ in pool]

    print("\n" + "=" * 72)
    print("EMPIRICAL PAYOFF MATRIX M[i][j] = P(seat 0 wins | row=i, col=j x3)")
    print("=" * 72)
    header = "              " + "  ".join(f"{l:>10s}" for l in labels)
    print(header)
    for i, li in enumerate(labels):
        row = "  ".join(f"{M[i,j]*100:10.2f}" for j in range(k))
        print(f"{li:>12s}  {row}")

    print("\n" + "=" * 72)
    print("MEAN MC HALF-WIDTH (95% Wilson)")
    print("=" * 72)
    print(f"  Average CI radius across cells: +/-{CI.mean()*100:.2f} pp")
    print(f"  Max CI radius:                  +/-{CI.max()*100:.2f} pp")

    print("\n" + "=" * 72)
    print("SYMMETRIC NASH EQUILIBRIUM (fictitious play, averaged)")
    print("=" * 72)
    v_meta = float(sigma @ M @ sigma)
    v_br = float(np.max(M @ sigma))
    exploit_final = v_br - v_meta

    for i, (label, _, _) in enumerate(pool):
        bar = "#" * int(round(sigma[i] * 40))
        print(f"  sigma*[{label:>20s}] = {sigma[i]*100:5.1f}%  {bar}")

    print()
    print(f"  v* (symmetric-Nash seat-0 win rate) : {v_meta*100:.2f} pp")
    print(f"  Best-response value                 : {v_br*100:.2f} pp")
    print(f"  Exploitability (v_br - v*)          : {exploit_final*100:.2f} pp")
    # Also store for telemetry
    _ = (v_meta, v_br, exploit_final)

    if exploit_final < 0.005:
        status = "EQ FOUND (exploitability < 0.5 pp)"
    elif exploit_final < 0.02:
        status = "NEAR-EQ (exploitability < 2 pp — acceptable for 4p stochastic)"
    else:
        status = (
            "NOT CONVERGED — raise --fp-iters, or MC noise is too large "
            "(raise --sims)"
        )
    print(f"  Convergence                         : {status}")

    print()
    print("Interpretation")
    print("-" * 72)
    print("  sigma* is the mixture of pool policies that no seat can profitably")
    print("  deviate from. If it collapses to a single policy, that policy")
    print("  dominates the pool (caveat: only in this pool — adding a")
    print("  stronger policy could shift the equilibrium, per PSRO's outer")
    print("  oracle step we omit here).")
    print()
    print("  v* is our sharpest point estimate of the seat-0 Nash payoff")
    print("  under the restriction to this pool. Compare to:")
    print("    - Monte Carlo RuleBased point estimate (overestimates skilled")
    print("      opponents' coordination)")
    print("    - policy_band min/max (uncertainty bar)")
    print("  PSRO's v* should lie inside the band; if it doesn't, the pool")
    print("  is missing an important strategy.")


def main():
    ap = argparse.ArgumentParser(
        description="Symmetric EGTA / PSRO meta-solver for Catan."
    )
    ap.add_argument(
        "--pool", type=str, default=DEFAULT_POOL,
        help=f"Comma-sep policies. Default: {DEFAULT_POOL}",
    )
    ap.add_argument(
        "--sims", type=int, default=2000,
        help="Monte Carlo sims per payoff cell.",
    )
    ap.add_argument(
        "--fp-iters", type=int, default=5000,
        help="Fictitious play iterations.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--board-seed", type=int, default=None,
        help="If set, use `get_random_board_layout(seed)`; else beginner board.",
    )
    ap.add_argument(
        "--position-json", type=str, default=None,
        help="Optional position JSON file (for mid-game analysis).",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pool = parse_pool(args.pool)
    print(f"PSRO / EGTA symmetric solver")
    print(f"  Pool: {[l for l, _, _ in pool]}")
    print(f"  {len(pool)}x{len(pool)} payoff matrix, {args.sims} sims/cell")
    print(f"  -> {len(pool)*len(pool)*args.sims:,} total games\n")

    # Board
    if args.board_seed is not None:
        board_json = catan_engine.get_random_board_layout(args.board_seed)
    else:
        board_json = None

    # Position
    position_json = None
    if args.position_json:
        position_json = Path(args.position_json).read_text()

    M, CI = build_payoff_matrix(
        pool, args.sims, board_json, position_json,
        seed=args.seed, verbose=not args.quiet,
    )

    sigma, exploit_traj = fictitious_play(
        M, iters=args.fp_iters, seed=args.seed
    )

    symmetric_nash_report(pool, M, CI, sigma, exploit_traj)

    # Dump report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(
            {
                "pool": [list(p) for p in pool],
                "M": M.tolist(),
                "CI": CI.tolist(),
                "sigma": sigma.tolist(),
                "v_meta": float(sigma @ M @ sigma),
                "v_br": float(np.max(M @ sigma)),
                "exploitability_final": float(exploit_traj[-1]),
                "exploitability_trajectory": exploit_traj,
                "n_sims_per_cell": args.sims,
                "fp_iters": args.fp_iters,
                "board_seed": args.board_seed,
            },
            f,
        )
    print(f"\nReport written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
