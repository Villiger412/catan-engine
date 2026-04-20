#!/usr/bin/env python3
"""
Coalition-pressure sensitivity sweep.

For each starting position we vary `coalition_pressure` ∈ {0.0, 0.5, 1.0, 1.5, 2.0}
and record how the engine's win-probability vector moves. In an N-player game
Nash is not unique, so the "GTO win probability" for a position is really a
band rather than a single number — this script estimates the width of that
band.

Reported quantities (per game):
  - rank1_leader      : identity of the player the baseline (cp=1.0) run ranks #1
  - leader_p_[cp]     : P(leader wins) at each coalition_pressure
  - leader_delta      : leader_p[cp=2.0] − leader_p[cp=0.0]
                        (how much the leader loses to full focus-fire)
  - max_drift         : max over all 4 players of (max cp prob − min cp prob)

Aggregate interpretation:
  - mean(leader_delta) near 0 → positions are coalition-insensitive; GTO is well-defined.
  - mean(leader_delta) significantly negative (−5pp+) → coalition behaviour is a real
    load-bearing assumption in the estimate; the engine's number should be reported
    as a band, not a point.

Usage:
    python coalition_sweep.py [--games 30] [--sims 1500] [--api URL] [--policy rule_based]

Requires the API server running on http://localhost:8000.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from analyze import DEFAULT_API, parse_game, stream_games

REPORT_PATH = Path(__file__).parent / "report" / "coalition_sweep.json"
SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]


def simulate_once(board, settlements, api_url, policy, n_sims, coalition_pressure):
    has_pos = all(len(s) == 2 for s in settlements)
    payload = {
        "board": board,
        "n_simulations": n_sims,
        "method": "monte_carlo",
        "policy": policy,
        "antithetic": True,
        "coalition_pressure": coalition_pressure,
    }
    if has_pos:
        payload["position"] = {
            "settlements": settlements,
            "cities": [[], [], [], []],
            "current_player": 0,
        }
    try:
        r = requests.post(f"{api_url}/api/simulate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("probabilities")
    except Exception:
        return None


def run_sweep(n_games, n_sims, api_url, policy):
    try:
        requests.get(f"{api_url}/api/health", timeout=5).raise_for_status()
    except Exception:
        print(f"ERROR: API not reachable at {api_url}")
        return []

    rows = []
    processed = 0
    t0 = time.time()

    for game in stream_games(n_games * 3, verbose=False):
        if processed >= n_games:
            break
        parsed = parse_game(game)
        if parsed is None or not parsed["has_positions"]:
            continue

        probs_by_cp = {}
        ok = True
        for cp in SWEEP:
            p = simulate_once(parsed["board"], parsed["settlements"], api_url, policy, n_sims, cp)
            if p is None or len(p) != 4:
                ok = False
                break
            probs_by_cp[cp] = p
        if not ok:
            continue

        baseline = np.array(probs_by_cp[1.0])
        leader = int(np.argmax(baseline))

        leader_probs = [probs_by_cp[cp][leader] for cp in SWEEP]
        leader_delta = leader_probs[-1] - leader_probs[0]

        all_matrix = np.array([probs_by_cp[cp] for cp in SWEEP])  # [cp, player]
        per_player_range = all_matrix.max(axis=0) - all_matrix.min(axis=0)
        max_drift = float(per_player_range.max())

        rows.append({
            "leader": leader,
            "leader_probs": leader_probs,
            "leader_delta": float(leader_delta),
            "max_drift": max_drift,
            "all_probs": all_matrix.tolist(),
            "winner_idx": parsed["winner_idx"],
            "elo_tier": parsed.get("elo_tier", -1),
        })
        processed += 1

        if processed % 5 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 1
            ld = np.mean([r["leader_delta"] for r in rows])
            mx = np.mean([r["max_drift"] for r in rows])
            print(f"  {processed}/{n_games}  "
                  f"rate {rate:.2f} g/s  "
                  f"mean leader_delta {ld*100:+.1f}pp  "
                  f"mean max_drift {mx*100:.1f}pp")

    return rows


def summarize(rows, policy):
    if not rows:
        print("No results.")
        return
    leader_deltas = np.array([r["leader_delta"] for r in rows])
    max_drifts = np.array([r["max_drift"] for r in rows])

    print(f"\nPolicy: {policy}  |  Games: {len(rows)}  |  Sweep: {SWEEP}")
    print(f"  leader_delta   (prob drop from cp=0 -> cp=2)")
    print(f"    mean   {leader_deltas.mean()*100:+.2f} pp")
    print(f"    median {np.median(leader_deltas)*100:+.2f} pp")
    print(f"    p10    {np.percentile(leader_deltas,10)*100:+.2f} pp")
    print(f"    p90    {np.percentile(leader_deltas,90)*100:+.2f} pp")
    print(f"  max_drift      (per-position band width)")
    print(f"    mean   {max_drifts.mean()*100:.2f} pp")
    print(f"    median {np.median(max_drifts)*100:.2f} pp")
    print(f"    p90    {np.percentile(max_drifts,90)*100:.2f} pp")
    print(f"    max    {max_drifts.max()*100:.2f} pp")
    print()
    print("Interpretation:")
    print("  leader_delta close to 0  → GTO win-prob is coalition-insensitive here.")
    print("  leader_delta strongly negative → coalition-against-leader is real and")
    print("  the 'single-number' win probability understates leader risk. The honest")
    print("  output for the user is the band [cp=0.0 prob, cp=2.0 prob].")

    # Does the baseline leader still top the chart at cp=2.0?
    held = sum(1 for r in rows if np.argmax(r["all_probs"][-1]) == r["leader"])
    print(f"\nLeader preserved at cp=2.0: {held}/{len(rows)} ({held/len(rows)*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",  type=int, default=30)
    ap.add_argument("--sims",   type=int, default=1500)
    ap.add_argument("--api",    type=str, default=DEFAULT_API)
    ap.add_argument("--policy", type=str, default="rule_based",
                    help="rule_based | mcts_rule_<N> — must honour coalition_pressure.")
    args = ap.parse_args()

    rows = run_sweep(args.games, args.sims, args.api, args.policy)

    # Write JSON *before* printing summary so a console-encoding crash
    # (Windows cp1252 can't encode some glyphs) doesn't lose the data.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "policy": args.policy,
            "sims": args.sims,
            "sweep": SWEEP,
            "rows": rows,
        }, f)

    try:
        summarize(rows, args.policy)
    except UnicodeEncodeError as e:
        print(f"(skipped pretty summary: {e})")
    print(f"\nRaw results: {REPORT_PATH}")


if __name__ == "__main__":
    main()
