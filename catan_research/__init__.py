"""Research mode for the Catan engine.

Write strategies in Python, run N games, get a per-game DataFrame.

Quick start::

    from catan_research import Strategy, simulate_games

    class CornerHoarder(Strategy):
        def score_setup_vertex(self, ctx, vertex_id):
            prod = ctx.vertex_production_value(vertex_id)
            diversity = ctx.vertex_resource_diversity(vertex_id)
            return prod * (diversity ** 2)

    df = simulate_games(strategy=CornerHoarder(), n_games=2000, seed=42)
    print(df.groupby("winner").size() / len(df))

The headline feature is `Strategy` — subclass it, override one or more
hook methods, and any decision you don't override falls through to the
built-in rule-based policy. So a 10-line setup-placement experiment is a
10-line file, not a 300-line policy reimplementation.
"""

from .strategy import (
    Strategy,
    DefaultStrategy,
    UniformRandom,
    ProductionMax,
)
from .core import simulate_games, sweep

__all__ = [
    "Strategy",
    "DefaultStrategy",
    "UniformRandom",
    "ProductionMax",
    "simulate_games",
    "sweep",
]
