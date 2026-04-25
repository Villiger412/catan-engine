"""Top-level entry points for research mode.

`simulate_games` runs N games and returns a per-game DataFrame.
`sweep` runs a grid of `simulate_games` calls and concatenates with the
sweep-key columns prepended.

Heavy lifting happens in Rust (`catan_engine.simulate_games_records`);
this module's job is argument coercion, DataFrame construction, and the
sweep plumbing.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import catan_engine

from .strategy import Strategy


def _coerce_strategy_arg(s: Any) -> Any:
    """Pass-through for built-in policy strings and `Strategy` instances.

    The Rust side accepts both — strings map to `PolicyType` variants;
    Strategy instances are introspected for `_overridden_hooks` and
    bound-method handles.
    """
    if isinstance(s, str):
        return s
    if isinstance(s, Strategy):
        return s
    raise TypeError(
        f"strategy must be a string ('rule_based', 'mcts_200', ...) or a "
        f"catan_research.Strategy instance, got {type(s).__name__}"
    )


def _records_to_df(records, n_games_requested: int, seed: int) -> pd.DataFrame:
    """Project the columnar GameRecordsResult into a one-row-per-game DataFrame.

    Per-seat fields arrive as List[List[u8]] of shape [n_games][4]; we
    transpose with numpy and emit `p{i}_*` columns for i in 0..3.
    """
    n = len(records.winner)
    base = {
        "game_id": np.arange(n, dtype=np.int64),
        "seed": np.full(n, np.int64(seed), dtype=np.int64),
        "antithetic_pair_id": np.asarray(records.antithetic_pair_id, dtype=np.int32),
        "winner": np.asarray(records.winner, dtype=np.int8),
        "turns": np.asarray(records.turns, dtype=np.int32),
        "longest_road_player": np.asarray(records.longest_road_player, dtype=np.int8),
        "largest_army_player": np.asarray(records.largest_army_player, dtype=np.int8),
    }

    per_seat_fields = (
        ("vp_total", records.vp_total),
        ("vp_hidden", records.vp_hidden),
        ("settlements", records.settlements),
        ("cities", records.cities),
        ("roads", records.roads),
        ("knights_played", records.knights_played),
        ("longest_road_len", records.longest_road_len),
        ("total_resources", records.total_resources),
        ("unplayed_dev", records.unplayed_dev),
    )
    cols: dict = dict(base)
    for name, data in per_seat_fields:
        if not data:
            for i in range(4):
                cols[f"p{i}_{name}"] = np.zeros(n, dtype=np.int16)
            continue
        arr = np.asarray(data, dtype=np.int16)  # shape [n, 4]
        for i in range(4):
            cols[f"p{i}_{name}"] = arr[:, i]

    return pd.DataFrame(cols)


def simulate_games(
    strategy: Any = None,
    *,
    seat_strategies: Sequence[Any] | None = None,
    n_games: int = 5000,
    seed: int = 42,
    coalition_pressure: float = 1.0,
    board: str | None = None,
    position: str | None = None,
    antithetic: bool = True,
    n_threads: int = 0,
) -> pd.DataFrame:
    """Run `n_games` games and return a per-game DataFrame.

    Args:
        strategy: Single strategy used for all 4 seats. Either a built-in
            policy name string (`"rule_based"`, `"mcts_200"`, ...) or a
            `catan_research.Strategy` instance. Defaults to "rule_based".
        seat_strategies: 4-element sequence overriding `strategy`. Use this
            to A/B a custom strategy in seat 0 against rule_based controls.
        n_games: Total games (split across threads automatically).
        seed: Base RNG seed; each thread derives its own.
        coalition_pressure: How aggressively the rule-based fallback (and
            built-in policies) target the VP leader. 0 = selfish, 1 =
            default, 2 = strong focus-fire.
        board: Optional JSON board layout (from `get_board_layout()` or
            `get_random_board_layout()`); defaults to the beginner board.
        position: Optional JSON mid-game position (settlements, cities,
            etc.); defaults to a fresh start with full setup phase.
        antithetic: Use antithetic dice rolls for variance reduction.
            Adds an `antithetic_pair_id` column (-1 if disabled).
        n_threads: 0 = auto. Forced to 1 when any seat is a Python
            strategy with hooks set (GIL contention).

    Returns:
        DataFrame with one row per game and the following columns:
            game_id, seed, antithetic_pair_id, winner, turns,
            longest_road_player, largest_army_player,
            p{i}_{vp_total, vp_hidden, settlements, cities, roads,
                  knights_played, longest_road_len, total_resources,
                  unplayed_dev} for i in 0..3.
        winner = -1 means a draw (or 300-turn timeout).
    """
    strategy_arg = _coerce_strategy_arg(strategy) if strategy is not None else None
    seat_arg = (
        [_coerce_strategy_arg(s) for s in seat_strategies]
        if seat_strategies is not None
        else None
    )

    records = catan_engine.simulate_games_records(
        n_games=n_games,
        strategy=strategy_arg,
        seat_strategies=seat_arg,
        n_threads=n_threads,
        antithetic=antithetic,
        seed=seed,
        coalition_pressure=coalition_pressure,
        board_json=board,
        position_json=position,
    )

    df = _records_to_df(records, n_games_requested=n_games, seed=seed)
    df.attrs["elapsed_ms"] = records.elapsed_ms
    df.attrs["games_per_sec"] = records.games_per_sec
    df.attrs["policy_label"] = records.policy_label
    return df


def _strategy_label(s: Any) -> str:
    if isinstance(s, str):
        return s
    if isinstance(s, Strategy):
        return s.label or type(s).__name__
    return repr(s)


def sweep(
    grid: Mapping[str, Iterable[Any]],
    *,
    n_games: int = 2000,
    seed: int = 42,
    out: str | None = None,
    **simulate_kwargs: Any,
) -> pd.DataFrame:
    """Cartesian product over `grid`, run `simulate_games` per cell, concat.

    Args:
        grid: Dict mapping a `simulate_games` kwarg name (e.g. `"strategy"`,
            `"coalition_pressure"`) to the list of values to sweep. Strategy
            instances are stringified by their `label` attr (or class name).
            Pass `(label, value)` tuples to override.
        n_games: Games per cell.
        seed: Base seed; passed unchanged to each cell. Use the
            antithetic_pair_id column for within-cell variance reduction.
        out: Optional CSV path; if given, the merged DataFrame is written
            with `index=False`.

    Returns:
        Concatenated DataFrame. Sweep keys are prepended to the columns.
    """
    # Normalize each grid value to (label, raw_value).
    norm: list[tuple[str, list[tuple[str, Any]]]] = []
    for key, values in grid.items():
        labelled: list[tuple[str, Any]] = []
        for v in values:
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
                labelled.append(v)  # explicit (label, value)
            elif key == "strategy":
                labelled.append((_strategy_label(v), v))
            else:
                labelled.append((str(v), v))
        norm.append((key, labelled))

    keys = [k for k, _ in norm]
    value_lists = [v for _, v in norm]

    frames: list[pd.DataFrame] = []
    for combo in itertools.product(*value_lists):
        cell_kwargs: dict = {"n_games": n_games, "seed": seed, **simulate_kwargs}
        labels: dict[str, str] = {}
        for k, (label, raw) in zip(keys, combo):
            labels[k] = label
            cell_kwargs[k] = raw
        df = simulate_games(**cell_kwargs)
        # Prepend sweep columns so they show up first in CSV / display.
        for k in reversed(keys):
            df.insert(0, k, labels[k])
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    if out:
        full.to_csv(out, index=False)
    return full
