"""Strategy base class and a few worked examples.

A `Strategy` is a Python object whose method overrides plug into the Rust
engine's decision pipeline. Each hook handles one phase of play; anything
you don't override falls through to the built-in rule-based default.

Hooks (in order of usefulness; rough call frequency in parens):

    score_setup_vertex(ctx, vertex_id) -> float    (~16/game)
    score_setup_road(ctx, edge_id) -> float        (~16/game)
    choose_robber_hex(ctx, candidates) -> int      (~8-20/game)
    choose_steal_target(ctx, candidates) -> int    (~8-20/game)
    choose_yop(ctx) -> int                         (~2/game)
    choose_monopoly(ctx) -> int                    (~2/game)
    select_action(ctx, actions) -> int|tuple       (~100-300/game; Tier-3)

`ctx` is a `StrategyCtx` exposed by the Rust side — read-only, with helper
methods like `ctx.vertex_production_value(v)`, `ctx.player_vp(p)`,
`ctx.adjacent_hexes(v)`, etc. See the README for the full list.

Resources are encoded as ints with these meanings::

    0 = Brick, 1 = Wood, 2 = Wheat, 3 = Ore, 4 = Sheep
"""

from __future__ import annotations


_HOOK_NAMES = (
    "score_setup_vertex",
    "score_setup_road",
    "choose_robber_hex",
    "choose_steal_target",
    "choose_yop",
    "choose_monopoly",
    "select_action",
)


class Strategy:
    """Base class for Python-pluggable Catan strategies.

    Subclass this, override one or more hooks, and pass an instance to
    `simulate_games(strategy=...)`. Anything you don't override runs the
    built-in rule-based default — full speed for those decisions.
    """

    #: Optional human-readable label. Used as the column value in `sweep()`.
    #: Defaults to the subclass's class name.
    label: str | None = None

    def __init__(self):
        cls = type(self)
        # Find which hook methods this subclass actually overrode. The Rust
        # adapter checks this set at construction time, so unoverridden hooks
        # never pay the GIL-acquire cost during simulation.
        overridden = {
            name
            for name in _HOOK_NAMES
            if getattr(cls, name) is not getattr(Strategy, name)
        }
        # Stored as a frozenset under the hard-coded attr name the Rust
        # adapter introspects.
        self._overridden_hooks = frozenset(overridden)
        if self.label is None:
            self.label = cls.__name__

    # ── Hooks (default no-ops; the Rust side never calls these) ───────────
    # Subclassing checks identity against THESE methods. Don't add any
    # behaviour here — that's what the rule-based fallback in Rust is for.

    def score_setup_vertex(self, ctx, vertex_id):
        """Score a vertex for initial settlement placement. Higher = better."""
        return 0.0

    def score_setup_road(self, ctx, edge_id):
        """Score an edge for initial road placement. Higher = better."""
        return 0.0

    def choose_robber_hex(self, ctx, candidates):
        """Pick a hex to move the robber to. Return one of `candidates`."""
        return candidates[0]

    def choose_steal_target(self, ctx, candidates):
        """Pick a player to steal from. Return one of `candidates`."""
        return candidates[0]

    def choose_yop(self, ctx):
        """Pick a resource for Year of Plenty (0..=4)."""
        return 0

    def choose_monopoly(self, ctx):
        """Pick a resource for Monopoly (0..=4)."""
        return 0

    def select_action(self, ctx, actions):
        """Tier-3 escape hatch — replaces ALL action selection.

        `actions` is a list of `(kind: str, arg: int)` tuples. Return
        either an int index into `actions` or one of the tuples. Slow
        (~20-50× the Tier-1 throughput); use Tier-2 hooks if possible.
        """
        return 0


# ── Example subclasses (also used as smoke tests) ─────────────────────────


class DefaultStrategy(Strategy):
    """Identical to `strategy="rule_based"`. Override no hooks → all
    decisions fall through to the built-in default. Useful as a no-op
    sanity check that the Strategy plumbing is wired up correctly."""

    label = "default"


class UniformRandom(Strategy):
    """Random main-action choice via the Tier-3 select_action hook.

    This is intentionally a bad player — it's a sanity check that
    overrides actually take effect (winrate should crater).
    """

    label = "uniform_random"

    def select_action(self, ctx, actions):
        import random
        return random.randrange(len(actions))


class ProductionMax(Strategy):
    """Setup-only experiment: place settlements purely on production
    value, ignoring diversity, ports, denial. Compares vs rule_based to
    quantify how much the latter's diversity/port bonuses are worth."""

    label = "production_max"

    def score_setup_vertex(self, ctx, vertex_id):
        return ctx.vertex_production_value(vertex_id)
