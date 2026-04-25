"""Example custom strategy. Use with::

    python -m catan_research.run --strategy-module examples/corner_hoarder.py:CornerHoarder \\
        --n-games 500 --out ch.csv

Compares the production×diversity² hypothesis vs rule_based's
production+diversity+port scoring. CornerHoarder *should* underperform
rule_based slightly because it ignores ports — that's the negative result
you can read straight from the per-game DataFrame.
"""

from catan_research import Strategy


class CornerHoarder(Strategy):
    """Scores setup vertices by production × diversity²."""

    label = "corner_hoarder"

    def score_setup_vertex(self, ctx, vertex_id):
        prod = ctx.vertex_production_value(vertex_id)
        diversity = ctx.vertex_resource_diversity(vertex_id)
        return prod * (diversity ** 2)
