"""CLI wrapper around `simulate_games` and `sweep`.

Examples::

    python -m catan_research.run --n-games 1000 --strategy rule_based --out r.csv
    python -m catan_research.run --strategy-module my_strats.py:CornerHoarder \\
        --n-games 500 --out ch.csv
    python -m catan_research.run --strategy rule_based \\
        --sweep coalition_pressure=0.0,1.0,2.0 --n-games 200 --out s.csv
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .core import simulate_games, sweep


def _load_strategy_from_spec(spec: str) -> Any:
    """Resolve `module.path:ClassName` or `path/to/file.py:ClassName`."""
    if ":" not in spec:
        raise SystemExit(
            f"--strategy-module needs 'module:Class' or 'path.py:Class', got {spec!r}"
        )
    mod_part, cls_part = spec.rsplit(":", 1)
    if mod_part.endswith(".py") or "/" in mod_part or "\\" in mod_part:
        path = Path(mod_part).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise SystemExit(f"could not load {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[path.stem] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)
    if not hasattr(module, cls_part):
        raise SystemExit(f"{spec}: class {cls_part!r} not found")
    return getattr(module, cls_part)()


def _parse_sweep_arg(spec: str) -> tuple[str, list]:
    """`coalition_pressure=0,1,2` → (`coalition_pressure`, [0.0, 1.0, 2.0])."""
    if "=" not in spec:
        raise SystemExit(f"--sweep entry must be KEY=v1,v2,...; got {spec!r}")
    key, vals = spec.split("=", 1)
    parts = [p.strip() for p in vals.split(",")]
    out: list = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
    return key.strip(), out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Catan research-mode games and emit a per-game CSV."
    )
    ap.add_argument(
        "--n-games", type=int, default=2000, help="Games per cell (default 2000)."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--strategy",
        type=str,
        default="rule_based",
        help="Built-in policy name. Ignored if --strategy-module is set.",
    )
    ap.add_argument(
        "--strategy-module",
        type=str,
        default=None,
        help="'module:Class' or 'path.py:Class' to import a custom Strategy.",
    )
    ap.add_argument(
        "--coalition-pressure",
        type=float,
        default=1.0,
        help="Rule-based / MCTS coalition pressure (0=selfish, 1=default, 2=focus-fire).",
    )
    ap.add_argument(
        "--no-antithetic",
        action="store_true",
        help="Disable antithetic-variates dice pairing.",
    )
    ap.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="0 = auto. Forced to 1 if Python hooks are set.",
    )
    ap.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Sweep KEY=v1,v2,...; repeatable (cartesian product).",
    )
    ap.add_argument("--out", type=str, default=None, help="Output CSV path.")
    args = ap.parse_args(argv)

    strategy: Any
    if args.strategy_module:
        strategy = _load_strategy_from_spec(args.strategy_module)
    else:
        strategy = args.strategy

    common_kwargs = dict(
        n_games=args.n_games,
        seed=args.seed,
        coalition_pressure=args.coalition_pressure,
        antithetic=not args.no_antithetic,
        n_threads=args.n_threads,
    )

    if args.sweep:
        grid = dict(_parse_sweep_arg(s) for s in args.sweep)
        if "strategy" not in grid:
            common_kwargs["strategy"] = strategy
        df = sweep(grid, out=args.out, **common_kwargs)
    else:
        df = simulate_games(strategy=strategy, **common_kwargs)
        if args.out:
            df.to_csv(args.out, index=False)

    n = len(df)
    elapsed = df.attrs.get("elapsed_ms")
    gps = df.attrs.get("games_per_sec")
    print(
        f"{n} games"
        + (f" in {elapsed:.0f} ms ({gps:.0f} g/s)" if elapsed is not None else "")
    )
    if "winner" in df:
        winrates = (df["winner"] == 0).mean(), (df["winner"] == 1).mean(), (
            df["winner"] == 2
        ).mean(), (df["winner"] == 3).mean()
        print("seat winrates: " + ", ".join(f"{w:.3f}" for w in winrates))
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
