#!/usr/bin/env python3
"""
Policy-sensitivity sweep.

For each of N sample games, run the engine with `policy=rule_based` and
`policy=random` from the same starting position. The per-game max |Δprob|
between the two runs is a proxy for how policy-sensitive that position is:

    - Low sensitivity → both policies agree → the outcome is driven by board /
      position structure more than agent skill. Probabilities here are robust
      estimates of "what happens under any reasonable play".

    - High sensitivity → the answer depends heavily on *who is playing it*.
      These positions have a larger Nash-uncertainty and a better policy
      (or best-response oracle) would change the estimate most.

This is a cheap proxy for a true exploitability bound: the gap between
`rule_based` self-play and `random` self-play is not the Nash gap, but an
upper-bound-ish signal for how much policy-induced drift a given position
has. If the mean drift is small, the engine's GTO estimate is robust.

Usage:
    python policy_sensitivity.py [--games 30] [--sims 1500]

Requires the API server running on http://localhost:8000.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from analyze import (
    DEFAULT_API, parse_game, stream_games, call_api,
)

REPORT_PATH = Path(__file__).parent / "report" / "policy_sensitivity.json"


def simulate_with_policy(board, settlements, n_sims, api_url, policy):
    """Call the engine with a specific policy, returning probabilities."""
    has_pos = all(len(s) == 2 for s in settlements)
    payload = {
        "board": board,
        "n_simulations": n_sims,
        "method": "monte_carlo",
        "policy": policy,
        "antithetic": True,
    }
    if has_pos:
        payload["position"] = {
            "settlements": settlements,
            "cities": [[], [], [], []],
            "current_player": 0,
        }
    try:
        r = requests.post(f"{api_url}/api/simulate", json=payload, timeout=120)
        r.raise_for_status()
        probs = r.json().get("probabilities")
        if probs and len(probs) == 4:
            return probs
    except Exception:
        pass
    return None


def run_sweep(n_games, n_sims, api_url):
    # Health check
    try:
        requests.get(f"{api_url}/api/health", timeout=5).raise_for_status()
    except Exception:
        print(f"ERROR: API not reachable at {api_url}")
        return

    rows = []
    processed = 0
    t0 = time.time()

    for game in stream_games(n_games * 3, verbose=False):
        if processed >= n_games:
            break
        parsed = parse_game(game)
        if parsed is None or not parsed["has_positions"]:
            continue

        board = parsed["board"]
        stl = parsed["settlements"]
        rb = simulate_with_policy(board, stl, n_sims, api_url, "rule_based")
        rn = simulate_with_policy(board, stl, n_sims, api_url, "random")
        if rb is None or rn is None:
            continue

        rb = np.array(rb)
        rn = np.array(rn)
        drift = float(np.max(np.abs(rb - rn)))
        rows.append({
            "rule_based_probs": rb.tolist(),
            "random_probs":     rn.tolist(),
            "winner_idx":       parsed["winner_idx"],
            "max_drift":        drift,
            "elo_tier":         parsed.get("elo_tier", -1),
        })
        processed += 1

        if processed % 5 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 1
            mean_drift = float(np.mean([r["max_drift"] for r in rows]))
            print(f"  {processed}/{n_games}  "
                  f"rate {rate:.1f} g/s  mean |d|: {mean_drift*100:.1f}pp")

    return rows


def summarize(rows):
    if not rows:
        print("No results.")
        return
    drifts = np.array([r["max_drift"] for r in rows])
    print(f"\nGames: {len(rows)}")
    print(f"  Mean  max |dprob|  : {drifts.mean()*100:.1f} pp")
    print(f"  Median max |dprob|: {np.median(drifts)*100:.1f} pp")
    print(f"  P90   max |dprob| : {np.percentile(drifts, 90)*100:.1f} pp")
    print(f"  Max   max |dprob| : {drifts.max()*100:.1f} pp")
    print()
    print("Interpretation:")
    print(" - Low drift (<5pp): engine prediction is robust to policy choice.")
    print(" - High drift (>15pp): position is policy-sensitive; GTO estimate")
    print("   is uncertain and a better policy (MCTS / best-response) would")
    print("   change the answer most.")

    # Breakdown by rule_based rank of human winner
    rank1_wins = 0
    rank_corr_high_drift = []
    rank_corr_low_drift = []
    for r in rows:
        rb = np.array(r["rule_based_probs"])
        ranked = np.argsort(-rb)
        if ranked[0] == r["winner_idx"]:
            rank1_wins += 1
        (rank_corr_high_drift if r["max_drift"] > 0.15 else rank_corr_low_drift).append(
            int(ranked[0] == r["winner_idx"])
        )
    print(f"\nRule-based rank-1 accuracy overall:       {rank1_wins/len(rows)*100:.1f}%")
    if rank_corr_low_drift:
        print(f"Rank-1 accuracy in low-drift games (<15pp): {np.mean(rank_corr_low_drift)*100:.1f}% (n={len(rank_corr_low_drift)})")
    if rank_corr_high_drift:
        print(f"Rank-1 accuracy in high-drift games (>=15pp): {np.mean(rank_corr_high_drift)*100:.1f}% (n={len(rank_corr_high_drift)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims",  type=int, default=1500)
    ap.add_argument("--api",   type=str, default=DEFAULT_API)
    args = ap.parse_args()

    rows = run_sweep(args.games, args.sims, args.api)
    summarize(rows)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(rows, f)
    print(f"\nRaw results: {REPORT_PATH}")


if __name__ == "__main__":
    main()
