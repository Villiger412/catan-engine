# Catan Engine

A research-grade Monte Carlo simulator for 4-player Settlers of Catan: a Rust
core wrapped in a FastAPI server, driven by a React UI. The goal is not to
build an agent that beats humans — it is to estimate **win probabilities**
for arbitrary board layouts and mid-game positions, with honest uncertainty
bars, and to use those estimates to probe the game-theoretic structure of
Catan (coalition dynamics, policy sensitivity, exploitability).

This README covers the theory behind what is already implemented, the error
rates we actually measure, and the three game-theoretic tools we are adding
in the current iteration.

---

## 1 · Why this exists

"What is my win probability from this position?" is a well-posed question in
2-player zero-sum games (AlphaZero, Libratus) but *not* in 4-player Catan.
Catan is:

- **Stochastic** — dice rolls, shuffled dev cards.
- **Imperfect information** — hidden resources and hidden VP cards.
- **General-sum, n > 2** — no minimax value. Nash equilibria are generically
  non-unique, coalition-dependent, and the folk theorem applies (Shoham &
  Leyton-Brown 2009, *Multiagent Systems*, ch. 3).

So "the GTO win probability" for a Catan position is not a single number. It
is a **band** whose width depends on how much the answer varies with rollout
policy and with coalition behavior against the leader. Every analysis tool in
this repo is designed to characterize that band rather than collapse it to a
point estimate.

---

## 2 · Architecture

```
┌───────────────┐   POST /api/simulate   ┌──────────────────┐  PyO3  ┌──────────────────┐
│   React UI    │ ─────────────────────▶ │   FastAPI + py   │ ─────▶ │  Rust core       │
│   (Vite)      │ ◀───────────────────── │   estimators/    │ ◀───── │  (rayon)         │
└───────────────┘    JSON: probs, CI,    │   monte_carlo.py │        │  state, engine,  │
                    games/s, avg_turns,  └──────────────────┘        │  policies, stats │
                    max_margin                                       └──────────────────┘
```

- **`src/`** (Rust, `rayon` multi-threaded): `state`, `engine`, `actions`,
  `board`, `simulation`, `stats`, `policy/{rule_based,mcts}`.
- **`api/`** (FastAPI): `estimators/monte_carlo.py` wraps the PyO3 module.
- **`ui/`** (React + TypeScript + Vite): board editor, piece editor, policy
  controls, self-correcting ETAs (`lib/timing.ts`).
- **`analyzer/`** (Python): offline studies on a corpus of recorded human
  games — `coalition_sweep.py`, `policy_sensitivity.py`,
  `best_response_probe.py`, and the new `policy_band.py` + `psro.py`.

### Error model

Each run reports a **95% CI** per player, computed from a Wilson interval
over the empirical winrate (see `src/stats.rs`). Auto-mode runs sequentially
until the widest per-player half-width falls under a target (default 2 pp).
Antithetic variates (paired sims that share dice streams but flip a policy
bit) are on by default for variance reduction.

Observed throughput on the current dev box (Windows 10, 12-core, release
build, beginner board, rayon default threads):

| Policy          | games/s     | ms/game | 5 000-sim wall time |
|-----------------|-------------|---------|---------------------|
| `rule_based`    | 6 000–8 000 | 0.13    | ~0.7 s              |
| `random`        | 2 500–3 500 | 0.33    | ~1.7 s              |
| `mcts_200`      | 20–25       | 45      | ~3.8 min            |
| `mcts_rule_200` | 10–12       | 90      | ~7.5 min            |

The UI calibrates these numbers on-the-fly: after each run, the observed
`turns_per_sec = avg_turns × games_per_sec` is stashed as a `Calibration`
and folded into the next ETA (`ui/src/lib/timing.ts`).

---

## 3 · Policies

Three families live in `src/policy/`. Each implements the single-method
`Policy` trait.

### 3.1 `RandomPolicy`

Uniformly samples from the legal action set. This is the baseline for
simulation-balance diagnostics, not a realistic opponent.

### 3.2 `RuleBasedPolicy`

An expert heuristic policy with a **single free parameter**, `coalition_pressure`
(default 1.0). Key mechanics (`src/policy/rule_based.rs`):

- **Setup placement.** Vertices are scored by production value
  `Σ resource_weight × production_weight × 36` plus a diversity bonus
  (`+1.5` per distinct adjacent resource) and a port bonus (2:1 port =
  `+2.0` *only* if the player produces that resource, else `+0.5`). Resource
  weights are re-scaled per-board by scarcity via `BoardContext`: a resource
  producing half the mean gets up to 2× weight, abundant resources get
  down-weighted, clamped to `[0.5, 2.0]`. Round-0 settlements also include a
  denial bonus weighted `0.30` (blocking adjacent vertices an opponent could
  reach via the distance rule); round-1 denial weight is `0.05` because
  there are no further own-placements to benefit.
- **Main action priority.** City > settlement > knight (when close to largest
  army) > dev card > goal-directed bank trade > road > road-building card >
  YoP/Monopoly > end turn.
- **Robber + steal.** Targets the opponent on the highest-production hex,
  biased toward the VP leader by `1 + vp × 0.3 × coalition_pressure`. At
  `cp = 0` the policy is pure best-response-to-production (selfish); at
  `cp = 1` it weights VP roughly as much as production; at `cp = 2` it
  approaches focus-fire on the leader. **Coalition-pressure sweeps are the
  principled way to expose the coalition-dependence of a win-probability
  estimate** (§5.2).
- **YoP / Monopoly.** YoP picks the resource with the largest marginal need
  (`weight / (held + 1)`); Monopoly picks the resource opponents hold the
  most of, scaled by base weight.

This is not a trained agent. It is a tuned policy in the spirit of the
JSettlers heuristic bot (Thomas 2003, *Real-time Decision Making for
Adversarial Environments Using a Plan-based Heuristic*).

### 3.3 `MctsPolicy` — Flat-UCB

A 1-ply UCB1 search (`src/policy/mcts.rs`) following Szita, Chaslot &
Spronck (ACG 2010, *Monte-Carlo Tree Search in Settlers of Catan*), who
reported ~49% win-rate vs JSettlers with 10 000 simulations per move. We
implement both variants from that paper:

- `mcts_{N}` — random rollouts. The original Szita & Chaslot setting.
- `mcts_rule_{N}` — rule-based rollouts at the given coalition pressure.
  ~5–10× slower per rollout; opponents play strategically during playouts
  so coalition-against-the-leader emerges endogenously in the tree's
  action values.

Action selection uses UCB1 with `c = √2`; the most-visited (not
highest-mean) action is returned, which is Kocsis & Szepesvári's (ECML
2006, *Bandit Based Monte-Carlo Planning*) standard robust-choice rule.

> **Rollout-policy counterintuition.** The MCTS literature has a robust
> finding — Gelly & Silver (ICML 2007, *Combining Online and Offline
> Knowledge in UCT*) and especially Silver & Tesauro (ICML 2009,
> *Monte-Carlo Simulation Balancing*) — that **stronger rollout policies
> often yield worse MCTS value estimates**, because they over- or
> under-weight specific tactical patterns. The fix is "simulation
> balancing": tune the rollout so that its *value estimates* are unbiased,
> not that individual playouts are strong. We do not yet apply simulation
> balancing — using random rollouts is the safer default until we do.

---

## 4 · What the numbers mean

A Monte Carlo estimate under a fixed joint policy profile `π = (π₁, …, π₄)`
estimates `E[win_i | π]`: the value of that specific strategy profile. It
is **not**:

- a Nash value (no unique Nash exists in 4-player Catan);
- a best-response value (no player is optimizing against the others);
- the *game's* value (there is no such scalar).

It is the empirical expectation under the joint profile you chose. Every
number the engine produces must be read with that caveat. The practical
consequence is that **two honest, well-converged runs with different
rollout policies can and do disagree by double-digit percentage points**
on the same position — and neither is wrong.

---

## 5 · What we've measured so far

### 5.1 Policy sensitivity (pairwise, rule-based vs random)

`analyzer/policy_sensitivity.py` re-simulates recorded human games under
`rule_based` and `random` from the same post-setup position and reports
the max per-player |Δprob| per game. Over our current 20-game sample
(1 500 sims/policy/game):

```
Games:         20
Mean max |Δp|: 15.8 pp
Median:        15.0 pp
P90:           27.9 pp
Max:           28.6 pp
```

That is a **lot** — the median position shifts 15 pp across the two
policies. This is the Gelly-Silver / Silver-Tesauro phenomenon in action:
different fixed policies are different Markov chains over the game tree,
and their stationary win distributions differ. See §6.1 below for how the
new `policy_band.py` tool generalizes this into a proper N-policy band.

### 5.2 Coalition-pressure sweep

`analyzer/coalition_sweep.py` holds a position fixed and sweeps
`coalition_pressure ∈ {0.0, 0.5, 1.0, 1.5, 2.0}`. For games where the
human leader ended up winning, we measure how the leader's estimated
win-prob varies with coalition pressure. The spread across that sweep is
a lower bound on the Nash-band width: if opponents actively target the
leader, the leader's prob drops; our estimate of "the" win-prob for the
position is therefore a range, not a point.

### 5.3 Best-response probe (exploitability lower bound)

`analyzer/best_response_probe.py` fixes everyone on `rule_based`, then
for each player P enumerates every legal alternative for P's **second
settlement** and re-simulates. `max(alt) − baseline` is a lower bound on
P's exploitability of the profile — "I could have gained at least this
much by choosing differently on one placement." Current 3-game sample
(this tool is slow; see §6.2 for the speed-ups):

```
Mean max-player exploit bound: 18.0 pp
Median:                        16.8 pp
P90:                           24.8 pp
```

Even varying a single placement leaves ~18 pp of value on the table for
*some* player. That puts a floor on how much a best-response oracle
could shift the GTO estimate — at least ~10 pp in most positions.

---

## 6 · What we're adding this iteration

The lit review in §1 ends with a concrete three-part recommendation:
report a sensitivity band, probe exploitability, and solve a meta-game
over the policy pool. §5.1–5.3 cover partial versions of the first two.
This iteration brings all three up to reporting quality.

### 6.1 `policy_band.py` — N-policy win-prob band

Generalizes `policy_sensitivity.py` from pairwise `(rule_based, random)`
drift to a proper **portfolio band**: for each sampled position, run the
engine under every policy in a configurable pool and report per-player
`[min, max]` across the pool. The width of that interval is the honest
uncertainty bar for the position under policy misspecification.

Pool default: `{random, rule_based_cp=0.0, rule_based_cp=1.0,
rule_based_cp=2.0, mcts_200}`. This spans "nobody plays tactically" to
"everyone plays expert + focus-fires the leader" to "1-ply search at every
decision" — a defensible cross-section of the plausible joint-policy
space. Not Nash — but a band the true Nash value almost certainly lies
inside.

### 6.2 `best_response_probe.py` — MCTS-deepened single-seat probe

The existing MVP uses `rule_based` as the "best response": we swap one
placement and check if the new placement is *better than rule_based
picked*. That is really a "rule-based has a local blind spot" detector.
This iteration adds a `--deep-policy mcts_500` option: the swapped-in
placement is evaluated by letting *that* seat play `mcts_500` (deeper
search) while the other three stay on `rule_based`. The resulting gap
is a tighter lower bound on exploitability — closer to a true
best-response-in-pool calculation.

### 6.3 `psro.py` — Empirical game-theoretic meta-solver

Implements a lightweight PSRO / EGTA loop in the spirit of Lanctot et
al. (NeurIPS 2017, *A Unified Game-Theoretic Approach to Multiagent
RL*):

1. Fix a finite pool of policies.
2. Build a **symmetric meta-game payoff matrix**
   `M[i][j]` = win-rate of a player who plays policy `i` while the
   other three all play policy `j`. (Asymmetric cell `(i, j, j, j)`;
   fully asymmetric 4^4 = 256 cells is feasible but adds little for a
   symmetric game and is left as a flag.)
3. Solve `M` for a symmetric Nash over the pool via **fictitious play**
   (Brown 1951, *Iterative Solution of Games by Fictitious Play*). For
   two-player-zero-sum FP provably converges; for our general-sum 4p
   meta-game it does not, but the cycle-averaged distribution is still
   a useful "what mix of policies is hardest to exploit" fingerprint.
4. Output: the FP mixed strategy `σ*` over the pool and the
   meta-game value `v* = σ*ᵀ M σ*` — the predicted win-rate under the
   symmetric meta-distribution. This is the closest tractable thing
   to a Nash win-probability estimate in this repo.

> CFR (Zinkevich et al. NeurIPS 2007, *Regret Minimization in Games with
> Incomplete Information*) is explicitly **not** a good fit here: its
> convergence guarantees apply to two-player zero-sum games with
> perfect recall and imperfect information. Catan violates all three.
> PSRO / fictitious play is the right tool for the shape of this game.

---

## 7 · Running things

```bash
# Rust core
cargo build --release
maturin develop --release          # build the PyO3 extension into the venv

# API (http://localhost:8000)
cd api && uvicorn main:app --reload

# UI  (http://localhost:5173, proxies /api → :8000)
cd ui && npm install && npm run dev

# Analyses
cd analyzer
python policy_sensitivity.py --games 30 --sims 1500   # pairwise drift
python policy_band.py        --games 30 --sims 1500   # N-policy band  (new)
python best_response_probe.py --games 10 --sims 800 \
       --deep-policy mcts_500                          # exploitability (deeper)
python psro.py --sims 1500 --fp-iters 2000            # meta-solver     (new)
```

---

## 9 · Research mode (Python-pluggable strategies + per-game CSV)

The aggregate `simulate_batch` API gives one number (winrate per seat). Often
you want the opposite: write your own strategy in Python without touching
Rust, run N games, and get *one row per game* you can analyze in pandas.

That's `catan_research`:

```python
from catan_research import Strategy, simulate_games

class CornerHoarder(Strategy):
    def score_setup_vertex(self, ctx, vertex_id):
        prod      = ctx.vertex_production_value(vertex_id)
        diversity = ctx.vertex_resource_diversity(vertex_id)
        return prod * (diversity ** 2)   # my hypothesis

df = simulate_games(strategy=CornerHoarder(), n_games=2000, seed=42)
print(df.groupby("winner").size() / len(df))      # winrate by seat
print(df.query("winner == 0")["turns"].describe())
```

Anything you don't override falls through to the built-in rule-based
default — Tier-1 (built-in) speed for those decisions, GIL-callback cost
only for the hooks you replace.

### Hooks

| Hook | Signature | ~Calls/game | Notes |
|---|---|---|---|
| `score_setup_vertex` | `(ctx, vertex_id) -> float` | ~16 | Argmax across legal vertices. |
| `score_setup_road` | `(ctx, edge_id) -> float` | ~16 | Argmax across legal edges. |
| `choose_robber_hex` | `(ctx, candidates: list[int]) -> int` | ~8-20 | Returns hex ID. |
| `choose_steal_target` | `(ctx, candidates: list[int]) -> int` | ~8-20 | Returns seat ID. |
| `choose_yop` / `choose_monopoly` | `(ctx) -> int` | ~2-4 | Returns resource (0-4). |
| `select_action` *(Tier-3)* | `(ctx, actions) -> int \| tuple` | 100-300 | Replaces the priority chain wholesale. Slow. |

`ctx` is a `StrategyCtx` exposed by the Rust side. Useful methods:
`ctx.vertex_production_value(v)`, `ctx.vertex_resource_diversity(v)`,
`ctx.adjacent_hexes(v)`, `ctx.adjacent_vertices(v)`, `ctx.is_port(v)`,
`ctx.port_kind(v)`, `ctx.player_vp(p)`, `ctx.player_resources(p)`,
`ctx.player_total_resources(p)`, `ctx.player_settlements(p)`,
`ctx.player_cities(p)`, `ctx.player_roads(p)`, `ctx.player_knights_played(p)`,
`ctx.hex_vertices(h)`, `ctx.hex_resource(h)`, `ctx.hex_number(h)`,
`ctx.hex_production(h)`, `ctx.player_has_building(p, v)`,
`ctx.player_has_road(p, e)`, `ctx.edge_endpoints(e)`, `ctx.vp_totals()`,
plus properties `current_player`, `turn_number`, `phase`, `robber_hex`,
`longest_road_player`, `largest_army_player`.

### Per-seat strategies

```python
# A/B vs baseline: my strategy in seat 0, rule_based for the rest
df = simulate_games(
    seat_strategies=[CornerHoarder(), "rule_based", "rule_based", "rule_based"],
    n_games=2000,
)
```

### `sweep()` over strategies and params

```python
from catan_research import sweep

df = sweep(
    grid={
        "strategy":           [CornerHoarder(), "rule_based"],
        "coalition_pressure": [0.0, 1.0, 2.0],
    },
    n_games=2000, seed=42, out="sweeps/mine_vs_baseline.csv",
)
```

### Output schema (one row per game)

`game_id`, `seed`, `antithetic_pair_id`, `winner` (-1 = draw), `turns`,
`longest_road_player`, `largest_army_player`, plus `p{i}_*` for `i ∈ 0..3`
and `* ∈ {vp_total, vp_hidden, settlements, cities, roads, knights_played,
longest_road_len, total_resources, unplayed_dev}`.

### Performance

| Tier | Description | Throughput hit |
|---|---|---|
| 1 | Built-in name (`"rule_based"`) | 1× (full speed) |
| 2 | A few setup or rare-phase hooks overridden | 2-5× slower |
| 3 | `select_action` overridden (every decision is Python) | 20-50× slower |

When any seat has live Python hooks, the records pipeline runs
single-threaded — GIL contention destroys parallel scaling, and Tier-2
strategies are fast enough not to need it.

### CLI

```bash
python -m catan_research.run --n-games 1000 --strategy rule_based --out r.csv
python -m catan_research.run --strategy-module my_strats.py:CornerHoarder \
    --n-games 500 --out ch.csv
python -m catan_research.run --sweep coalition_pressure=0,1,2 --n-games 200 \
    --out sweep.csv
```

---

## 8 · Key references

- Szita, I.; Chaslot, G. & Spronck, P. (2010). *Monte-Carlo Tree Search in Settlers of Catan.* ACG.
- Thomas, R. (2003). *Real-time Decision Making for Adversarial Environments Using a Plan-based Heuristic.* JSettlers thesis.
- Pfeiffer, M. (2004). *Reinforcement Learning of Strategies for Settlers of Catan.*
- Kocsis, L. & Szepesvári, C. (2006). *Bandit Based Monte-Carlo Planning.* ECML.
- Gelly, S. & Silver, D. (2007). *Combining Online and Offline Knowledge in UCT.* ICML.
- Silver, D. & Tesauro, G. (2009). *Monte-Carlo Simulation Balancing.* ICML.
- Shoham, Y. & Leyton-Brown, K. (2009). *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations.*
- Lanctot, M. et al. (2017). *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning.* NeurIPS (PSRO).
- Brown, G. W. (1951). *Iterative Solution of Games by Fictitious Play.*
- Zinkevich, M.; Johanson, M.; Bowling, M. & Piccione, C. (2007). *Regret Minimization in Games with Incomplete Information.* NeurIPS (CFR).
