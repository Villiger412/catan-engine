#!/usr/bin/env python3
"""
N-policy win-probability band.

Generalizes `policy_sensitivity.py` from a pairwise (rule_based, random)
drift metric to a full *portfolio band*: given a pool of policies, run the
engine under each one from the same position and report, per player, the
[min, max] win-probability across the pool. The width of that interval is
the honest uncertainty bar for the position under policy misspecification.

Why this is the right summary for 4-player Catan:
    A Monte Carlo estimate under a fixed joint policy profile π = (π₁...π₄)
    estimates E[win_i | π], not a Nash value (no unique Nash for n>2
    general-sum stochastic games — Shoham & Leyton-Brown 2009). Different
    policies realize different joint profiles. The true Nash win-prob for
    any sensible notion of Nash must lie somewhere the pool *covers* — so
    the min/max across a diverse pool is a principled uncertainty band,
    not a point estimate.

Default pool:
    random                      -- no-tactics baseline
    rule_based (cp=0.0)         -- expert, selfish (pure best-response)
    rule_based (cp=1.0)         -- expert, default coalition weighting
    rule_based (cp=2.0)         -- expert, focus-fire on the leader
    mcts_200                    -- 1-ply UCB search (Szita & Chaslot 2010)

Usage:
    python policy_band.py [--games 30] [--sims 1500] \\
                          [--pool "random,rule_based:1.0,mcts_200"]

Requires the API server on http://localhost:8000.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from analyze import DEFAULT_API, parse_game, stream_games

REPORT_PATH = Path(__file__).parent / "report" / "policy_band.json"

# Pool entry syntax: "name[:coalition_pressure]". Coalition pressure only
# affects rule_based and mcts_rule; it's silently ignored for random / mcts_N.
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
            f"{name}:{cp}" if name in ("rule_based", "mcts_rule_200", "mcts_rule_500") else name
        )
        out.append((label, name.strip(), cp))
    return out


def simulate(board, settlements, n_sims, api_url, policy, coalition_pressure):
    has_pos = all(len(s) == 2 for s in settlements)
    payload = {
        "board": board,
        "n_simulations": n_sims,
        "method": "monte_carlo",
        "policy": policy,
        "coalition_pressure": coalition_pressure,
        "antithetic": True,
        "seed": 42,
    }
    if has_pos:
        payload["position"] = {
            "settlements": settlements,
            "cities": [[], [], [], []],
            "current_player": 0,
        }
    try:
        r = requests.post(f"{api_url}/api/simulate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        probs = data.get("probabilities")
        if probs and len(probs) == 4:
            return {
                "probs": probs,
                "sims":  data.get("simulations_run"),
                "elapsed_ms": data.get("elapsed_ms"),
            }
    except Exception as e:
        print(f"    [warn] {policy} cp={coalition_pressure}: {e}")
    return None


def run_sweep(n_games, n_sims, api_url, pool):
    try:
        requests.get(f"{api_url}/api/health", timeout=5).raise_for_status()
    except Exception:
        print(f"ERROR: API not reachable at {api_url}")
        return None

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

        per_policy = {}
        all_ok = True
        for label, policy, cp in pool:
            res = simulate(board, stl, n_sims, api_url, policy, cp)
            if res is None:
                all_ok = False
                break
            per_policy[label] = res
        if not all_ok:
            continue

        # Build [min, max] band per player
        mat = np.array([per_policy[l]["probs"] for l, _, _ in pool])  # (P, 4)
        band_lo = mat.min(axis=0).tolist()
        band_hi = mat.max(axis=0).tolist()
        band_w  = [hi - lo for lo, hi in zip(band_lo, band_hi)]

        rows.append({
            "winner_idx":  parsed["winner_idx"],
            "per_policy":  per_policy,
            "band_lo":     band_lo,
            "band_hi":     band_hi,
            "band_width":  band_w,
            "max_band_width": max(band_w),
            "elo_tier":    parsed.get("elo_tier", -1),
        })
        processed += 1

        elapsed = time.time() - t0
        mean_w = float(np.mean([r["max_band_width"] for r in rows]))
        print(f"  [{processed}/{n_games}] "
              f"max band this game: {rows[-1]['max_band_width']*100:5.1f}pp  "
              f"running mean: {mean_w*100:5.1f}pp  t={elapsed:.0f}s")

    return rows


def summarize(rows, pool):
    if not rows:
        print("No results.")
        return
    widths = np.array([r["max_band_width"] for r in rows])
    print(f"\nPool ({len(pool)} policies): {[l for l, _, _ in pool]}")
    print(f"Games:         {len(rows)}")
    print(f"  Mean  max band : {widths.mean()*100:5.1f} pp")
    print(f"  Median:          {np.median(widths)*100:5.1f} pp")
    print(f"  P90:             {np.percentile(widths, 90)*100:5.1f} pp")
    print(f"  Max:             {widths.max()*100:5.1f} pp")
    print()
    print("Interpretation:")
    print("  The band is a principled uncertainty bar — the true Nash")
    print("  value for any sensible notion of Nash lies (with high prob)")
    print("  inside this range. A narrow band (<5 pp) means the position")
    print("  is policy-robust; a wide band means the answer is driven by")
    print("  opponent behaviour, not board structure.")

    # Per-policy mean win-prob for each seat, averaged across sampled games.
    print("\nMean per-player win-prob by policy (sanity check):")
    for label, _, _ in pool:
        mean_probs = np.mean(
            [r["per_policy"][label]["probs"] for r in rows], axis=0
        )
        print(f"  {label:25s} {[f'{p*100:5.1f}' for p in mean_probs]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims",  type=int, default=1500)
    ap.add_argument("--api",   type=str, default=DEFAULT_API)
    ap.add_argument("--pool",  type=str, default=DEFAULT_POOL,
                    help=f"Comma-sep policies. Default: {DEFAULT_POOL}")
    args = ap.parse_args()

    pool = parse_pool(args.pool)
    print(f"Pool: {[l for l, _, _ in pool]}")
    print(f"Games target: {args.games}, sims/policy/game: {args.sims}")
    print(f"  → ~{args.games * len(pool) * args.sims:,} total sims\n")

    rows = run_sweep(args.games, args.sims, args.api, pool)
    if rows is None:
        return
    summarize(rows, pool)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({"pool": [list(p) for p in pool], "rows": rows}, f)
    print(f"\nRaw results: {REPORT_PATH}")


if __name__ == "__main__":
    main()
