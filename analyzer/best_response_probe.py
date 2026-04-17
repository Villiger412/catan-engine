#!/usr/bin/env python3
"""
Best-response oracle (MVP) — exploitability lower bound from last placement.

For each sampled game with recorded starting positions:
  1. Compute the baseline 4-player win probabilities under `rule_based` self-play
     given the actual human placements.
  2. For each player P, enumerate every legal alternative for P's *second*
     settlement (first is fixed; varying the whole setup tree requires engine
     changes). For each alternative, re-simulate and record P's win rate.
  3. max(alternative win rate) − baseline = a lower bound on P's exploitability
     against rule_based opponents: "I could have done at least this much
     better by choosing differently on my last placement."

This does NOT measure true Nash-distance — only one decision is varied, and
only for one player at a time. But it gives a concrete, cheap signal that
the current rule_based placement is leaving win-probability on the table, and
whose magnitude bounds how much a better policy could improve the GTO estimate
for this specific position.

Usage:
    python best_response_probe.py [--games 10] [--sims 800]

Requires the API server on http://localhost:8000.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from analyze import (
    DEFAULT_API, parse_game, stream_games,
    HEX_VERTICES, VERTEX_POSITIONS,
)

REPORT_PATH = Path(__file__).parent / "report" / "best_response_probe.json"

# Vertex adjacency (two vertices that share an edge). Derived on the fly.
_VERTEX_NEIGHBORS = None

def vertex_neighbors():
    """Build the 54-vertex adjacency map from HEX_VERTICES, matching the Rust
    topology (two vertices adjacent iff consecutive around some hex)."""
    global _VERTEX_NEIGHBORS
    if _VERTEX_NEIGHBORS is not None:
        return _VERTEX_NEIGHBORS
    neigh = [set() for _ in range(54)]
    for hex_verts in HEX_VERTICES:
        for i in range(6):
            a, b = hex_verts[i], hex_verts[(i + 1) % 6]
            neigh[a].add(b)
            neigh[b].add(a)
    _VERTEX_NEIGHBORS = [sorted(s) for s in neigh]
    return _VERTEX_NEIGHBORS


def legal_alternatives(settlements, excluded_player, excluded_slot):
    """Return vertices that satisfy the distance rule w.r.t. all placed
    settlements except the one being swapped out (player P's 2nd)."""
    neigh = vertex_neighbors()
    occupied = set()
    for p in range(4):
        for s_idx, v in enumerate(settlements[p]):
            if p == excluded_player and s_idx == excluded_slot:
                continue
            occupied.add(v)
    # A candidate v is legal iff v not occupied and no neighbor occupied
    legal = []
    for v in range(54):
        if v in occupied:
            continue
        if any(n in occupied for n in neigh[v]):
            continue
        legal.append(v)
    return legal


def simulate_position(board, settlements, n_sims, api_url, seed=42):
    payload = {
        "board": board,
        "n_simulations": n_sims,
        "method": "monte_carlo",
        "policy": "rule_based",
        "antithetic": True,
        "seed": seed,
        "position": {
            "settlements": settlements,
            "cities": [[], [], [], []],
            "current_player": 0,
        },
    }
    try:
        r = requests.post(f"{api_url}/api/simulate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("probabilities")
    except Exception:
        return None


def probe_game(game, n_sims, api_url):
    parsed = parse_game(game)
    if parsed is None or not parsed["has_positions"]:
        return None
    board = parsed["board"]
    settlements = [list(s) for s in parsed["settlements"]]
    winner = parsed["winner_idx"]

    baseline = simulate_position(board, settlements, n_sims, api_url, seed=42)
    if baseline is None:
        return None

    per_player = []
    for P in range(4):
        orig_second = settlements[P][1]
        alts = legal_alternatives(settlements, P, 1)
        best_prob = baseline[P]
        best_v = orig_second
        tested = 0
        for v in alts:
            if v == orig_second:
                continue
            new_positions = [list(s) for s in settlements]
            new_positions[P][1] = v
            p = simulate_position(board, new_positions, n_sims, api_url, seed=42)
            if p is None:
                continue
            tested += 1
            if p[P] > best_prob:
                best_prob = p[P]
                best_v = v
        per_player.append({
            "player":        P,
            "baseline_prob": baseline[P],
            "best_prob":     best_prob,
            "best_vertex":   int(best_v),
            "orig_vertex":   int(orig_second),
            "improvement":   best_prob - baseline[P],
            "alternatives_tested": tested,
        })

    return {
        "baseline": baseline,
        "winner":   winner,
        "per_player": per_player,
        "max_exploitability": max(pp["improvement"] for pp in per_player),
    }


def run(n_games, n_sims, api_url):
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
        result = probe_game(game, n_sims, api_url)
        if result is None:
            continue
        rows.append(result)
        processed += 1
        elapsed = time.time() - t0
        avg_expl = np.mean([r["max_exploitability"] for r in rows])
        print(f"  [{processed}/{n_games}] "
              f"max exploit this game: {result['max_exploitability']*100:5.1f}pp  "
              f"running mean: {avg_expl*100:5.1f}pp  "
              f"t={elapsed:.0f}s")
    return rows


def summarize(rows):
    if not rows:
        print("No results.")
        return
    maxes = np.array([r["max_exploitability"] for r in rows])
    print(f"\nGames probed: {len(rows)}")
    print(f"  Mean max-player exploit bound: {maxes.mean()*100:.1f} pp")
    print(f"  Median:                        {np.median(maxes)*100:.1f} pp")
    print(f"  P90:                           {np.percentile(maxes, 90)*100:.1f} pp")
    print(f"  Max:                           {maxes.max()*100:.1f} pp")
    print()
    print("Interpretation:")
    print(" - A small mean exploit bound (<2pp): rule_based is close to a")
    print("   local optimum for 2nd-placement decisions; improving the policy")
    print("   probably won't shift per-game GTO estimates much.")
    print(" - A large mean (>5pp): rule_based leaves meaningful value on the")
    print("   table; its GTO estimate is systematically biased toward its own")
    print("   heuristic. A better policy (MCTS / CFR) could shift the GTO")
    print("   prediction by up to this much per game, per player.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims",  type=int, default=800)
    ap.add_argument("--api",   type=str, default=DEFAULT_API)
    args = ap.parse_args()

    rows = run(args.games, args.sims, args.api)
    if rows is None:
        return
    summarize(rows)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(rows, f)
    print(f"\nRaw results: {REPORT_PATH}")


if __name__ == "__main__":
    main()
