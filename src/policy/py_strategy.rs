//! Python-supplied strategy adapter. The headline research-mode feature.
//!
//! A `PyStrategy` carries an embedded `RuleBasedPolicy` for fall-through plus
//! optional `Py<PyAny>` hooks for each pluggable decision point. When a hook
//! is `Some`, we acquire the GIL and call into Python; when `None`, we run
//! the rule-based default. That preserves Tier-1 (built-in) speed and only
//! pays Python costs for the decisions the researcher actually overrode.
//!
//! The `StrategyCtx` `#[pyclass]` exposed to hooks is a thin read-only view
//! over `&GameState` + `&BoardLayout`. It's `unsendable` and uses raw
//! pointers internally — the pointers are valid only during the synchronous
//! Python call, and the user can't smuggle the ctx out of the hook because
//! `unsendable` blocks cross-thread moves.

use crate::actions::Action;
use crate::board::*;
use crate::policy::rule_based::RuleBasedPolicy;
use crate::policy::Policy;
use crate::state::*;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng;

/// Names of all hooks. Matched against the Python `Strategy` subclass to
/// detect which hooks are overridden.
pub const HOOK_NAMES: &[&str] = &[
    "score_setup_vertex",
    "score_setup_road",
    "choose_robber_hex",
    "choose_steal_target",
    "choose_yop",
    "choose_monopoly",
    "select_action",
];

/// Adapter that lets a Python `Strategy` subclass plug into the engine.
pub struct PyStrategy {
    pub label: String,
    pub defaults: RuleBasedPolicy,
    pub hook_setup_vertex: Option<Py<PyAny>>,
    pub hook_setup_road: Option<Py<PyAny>>,
    pub hook_robber_hex: Option<Py<PyAny>>,
    pub hook_steal_target: Option<Py<PyAny>>,
    pub hook_yop: Option<Py<PyAny>>,
    pub hook_monopoly: Option<Py<PyAny>>,
    pub hook_select_action: Option<Py<PyAny>>,
}

impl PyStrategy {
    /// Build a `PyStrategy` from a Python `Strategy` instance.
    ///
    /// Only methods listed in `obj._overridden_hooks` are stored. Everything
    /// else falls through to the embedded rule-based policy.
    pub fn from_pyobject(
        _py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        coalition_pressure: f64,
    ) -> PyResult<Self> {
        let label: String = obj
            .getattr("label")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| "PyStrategy".to_string());

        let overridden: std::collections::HashSet<String> =
            obj.getattr("_overridden_hooks")?.extract()?;

        let take = |name: &str| -> PyResult<Option<Py<PyAny>>> {
            if overridden.contains(name) {
                Ok(Some(obj.getattr(name)?.unbind()))
            } else {
                Ok(None)
            }
        };

        Ok(PyStrategy {
            label,
            defaults: RuleBasedPolicy::new(coalition_pressure),
            hook_setup_vertex: take("score_setup_vertex")?,
            hook_setup_road: take("score_setup_road")?,
            hook_robber_hex: take("choose_robber_hex")?,
            hook_steal_target: take("choose_steal_target")?,
            hook_yop: take("choose_yop")?,
            hook_monopoly: take("choose_monopoly")?,
            hook_select_action: take("select_action")?,
        })
    }

    /// True iff at least one Python hook is set. If not, this strategy is
    /// equivalent to plain rule-based and the records pipeline can take a
    /// no-GIL fast path.
    pub fn is_pure_default(&self) -> bool {
        self.hook_setup_vertex.is_none()
            && self.hook_setup_road.is_none()
            && self.hook_robber_hex.is_none()
            && self.hook_steal_target.is_none()
            && self.hook_yop.is_none()
            && self.hook_monopoly.is_none()
            && self.hook_select_action.is_none()
    }
}

impl Policy for PyStrategy {
    fn select_action<R: Rng>(
        &self,
        state: &GameState,
        board: &BoardLayout,
        actions: &[Action],
        rng: &mut R,
    ) -> Action {
        if actions.len() == 1 {
            return actions[0];
        }

        // Tier-3 escape hatch: if the user overrode `select_action`, route
        // every decision through it.
        if let Some(hook) = &self.hook_select_action {
            if let Some(a) = call_select_action(hook, state, board, actions) {
                return a;
            }
        }

        match state.phase {
            GamePhase::SetupSettlement { .. } => {
                if let Some(hook) = &self.hook_setup_vertex {
                    if let Some(a) = score_argmax_vertex(hook, state, board, actions) {
                        return a;
                    }
                }
            }
            GamePhase::SetupRoad { .. } => {
                if let Some(hook) = &self.hook_setup_road {
                    if let Some(a) = score_argmax_road(hook, state, board, actions) {
                        return a;
                    }
                }
            }
            GamePhase::MoveRobber => {
                if let Some(hook) = &self.hook_robber_hex {
                    if let Some(a) = call_chooser_hex(hook, state, board, actions) {
                        return a;
                    }
                }
            }
            GamePhase::StealResource => {
                if let Some(hook) = &self.hook_steal_target {
                    if let Some(a) = call_chooser_steal(hook, state, board, actions) {
                        return a;
                    }
                }
            }
            GamePhase::YearOfPlentyPick { .. } => {
                if let Some(hook) = &self.hook_yop {
                    if let Some(a) = call_chooser_resource(
                        hook,
                        state,
                        board,
                        actions,
                        Action::YearOfPlentyPick,
                    ) {
                        return a;
                    }
                }
            }
            GamePhase::MonopolyPick => {
                if let Some(hook) = &self.hook_monopoly {
                    if let Some(a) = call_chooser_resource(
                        hook,
                        state,
                        board,
                        actions,
                        Action::MonopolyPick,
                    ) {
                        return a;
                    }
                }
            }
            _ => {}
        }

        // Fallback: rule-based default for any phase the user didn't override.
        self.defaults.select_action(state, board, actions, rng)
    }
}

// ── Hook callers ───────────────────────────────────────────────────────────

/// Build a `StrategyCtx` Python object backed by raw pointers to `state`
/// and `board`. The pointers are valid for the duration of the GIL section
/// the caller wraps this in.
fn make_ctx<'py>(
    py: Python<'py>,
    state: &GameState,
    board: &BoardLayout,
) -> PyResult<Bound<'py, StrategyCtx>> {
    Bound::new(
        py,
        StrategyCtx {
            state_ptr: state as *const GameState,
            board_ptr: board as *const BoardLayout,
        },
    )
}

fn score_argmax_vertex(
    hook: &Py<PyAny>,
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
) -> Option<Action> {
    Python::attach(|py| {
        let ctx = make_ctx(py, state, board).ok()?;
        let mut best: Option<(f64, Action)> = None;
        for &a in actions {
            if let Action::PlaceSettlement(v) = a {
                let res = hook.call1(py, (&ctx, v as u32)).ok()?;
                let score: f64 = res.extract(py).ok()?;
                if best.map_or(true, |(s, _)| score > s) {
                    best = Some((score, a));
                }
            }
        }
        best.map(|(_, a)| a)
    })
}

fn score_argmax_road(
    hook: &Py<PyAny>,
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
) -> Option<Action> {
    Python::attach(|py| {
        let ctx = make_ctx(py, state, board).ok()?;
        let mut best: Option<(f64, Action)> = None;
        for &a in actions {
            if let Action::PlaceRoad(e) = a {
                let res = hook.call1(py, (&ctx, e as u32)).ok()?;
                let score: f64 = res.extract(py).ok()?;
                if best.map_or(true, |(s, _)| score > s) {
                    best = Some((score, a));
                }
            }
        }
        best.map(|(_, a)| a)
    })
}

fn call_chooser_hex(
    hook: &Py<PyAny>,
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
) -> Option<Action> {
    Python::attach(|py| {
        let ctx = make_ctx(py, state, board).ok()?;
        let candidates: Vec<u32> = actions
            .iter()
            .filter_map(|a| match a {
                Action::MoveRobber(h) => Some(*h as u32),
                _ => None,
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let py_list = PyList::new(py, &candidates).ok()?;
        let res = hook.call1(py, (&ctx, py_list)).ok()?;
        let chosen: u32 = res.extract(py).ok()?;
        let chosen_u8 = chosen as u8;
        actions
            .iter()
            .copied()
            .find(|a| matches!(a, Action::MoveRobber(h) if *h == chosen_u8))
    })
}

fn call_chooser_steal(
    hook: &Py<PyAny>,
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
) -> Option<Action> {
    Python::attach(|py| {
        let ctx = make_ctx(py, state, board).ok()?;
        let candidates: Vec<u32> = actions
            .iter()
            .filter_map(|a| match a {
                Action::StealFrom(t) => Some(*t as u32),
                _ => None,
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let py_list = PyList::new(py, &candidates).ok()?;
        let res = hook.call1(py, (&ctx, py_list)).ok()?;
        let chosen: u32 = res.extract(py).ok()?;
        let chosen_u8 = chosen as u8;
        actions
            .iter()
            .copied()
            .find(|a| matches!(a, Action::StealFrom(t) if *t == chosen_u8))
    })
}

fn call_chooser_resource<F>(
    hook: &Py<PyAny>,
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
    builder: F,
) -> Option<Action>
where
    F: Fn(u8) -> Action,
{
    Python::attach(|py| {
        let ctx = make_ctx(py, state, board).ok()?;
        let res = hook.call1(py, (&ctx,)).ok()?;
        let chosen: u32 = res.extract(py).ok()?;
        let target = builder(chosen as u8);
        if actions.iter().any(|&a| a == target) {
            Some(target)
        } else {
            None
        }
    })
}

/// Tier-3 escape hatch. Hands the full action list to Python as a list of
/// (kind: str, arg: int) tuples. The hook returns either the index of the
/// chosen action or one of the tuples.
fn call_select_action(
    hook: &Py<PyAny>,
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
) -> Option<Action> {
    Python::attach(|py| {
        let ctx = make_ctx(py, state, board).ok()?;
        let actions_py = PyList::empty(py);
        for &a in actions {
            let (kind, arg) = action_to_kind_arg(a);
            let tup = (kind, arg).into_pyobject(py).ok()?;
            actions_py.append(tup).ok()?;
        }
        let res = hook.call1(py, (&ctx, actions_py)).ok()?;
        if let Ok(idx) = res.extract::<usize>(py) {
            return actions.get(idx).copied();
        }
        // Tuple form: (kind, arg)
        let tup: (String, i64) = res.extract(py).ok()?;
        let candidate = action_from_kind_arg(&tup.0, tup.1)?;
        if actions.contains(&candidate) {
            Some(candidate)
        } else {
            None
        }
    })
}

/// Encode a Rust `Action` as a `(kind: &str, arg: i64)` pair.
/// `BankTrade` packs (give, receive) into one int as `give * 256 + receive`.
fn action_to_kind_arg(action: Action) -> (&'static str, i64) {
    match action {
        Action::PlaceSettlement(v) => ("place_settlement", v as i64),
        Action::PlaceRoad(e) => ("place_road", e as i64),
        Action::BuildCity(v) => ("build_city", v as i64),
        Action::BuyDevCard => ("buy_dev_card", 0),
        Action::PlayKnight(h) => ("play_knight", h as i64),
        Action::MoveRobber(h) => ("move_robber", h as i64),
        Action::StealFrom(t) => ("steal_from", t as i64),
        Action::Discard(_) => ("discard", 0),
        Action::BankTrade { give, receive } => {
            ("bank_trade", (give as i64) * 256 + receive as i64)
        }
        Action::YearOfPlentyPick(r) => ("yop_pick", r as i64),
        Action::MonopolyPick(r) => ("monopoly_pick", r as i64),
        Action::PlayRoadBuilding => ("play_road_building", 0),
        Action::EndTurn => ("end_turn", 0),
        Action::Pass => ("pass", 0),
    }
}

fn action_from_kind_arg(kind: &str, arg: i64) -> Option<Action> {
    let arg_u8 = arg as u8;
    Some(match kind {
        "place_settlement" => Action::PlaceSettlement(arg_u8),
        "place_road" => Action::PlaceRoad(arg_u8),
        "build_city" => Action::BuildCity(arg_u8),
        "buy_dev_card" => Action::BuyDevCard,
        "play_knight" => Action::PlayKnight(arg_u8),
        "move_robber" => Action::MoveRobber(arg_u8),
        "steal_from" => Action::StealFrom(arg_u8),
        "bank_trade" => Action::BankTrade {
            give: ((arg / 256) & 0xff) as u8,
            receive: (arg & 0xff) as u8,
        },
        "yop_pick" => Action::YearOfPlentyPick(arg_u8),
        "monopoly_pick" => Action::MonopolyPick(arg_u8),
        "play_road_building" => Action::PlayRoadBuilding,
        "end_turn" => Action::EndTurn,
        "pass" => Action::Pass,
        _ => return None,
    })
}

// ── StrategyCtx ────────────────────────────────────────────────────────────

/// Read-only view over `GameState` + `BoardLayout`, exposed to Python hooks.
///
/// Holds raw pointers — valid only inside the synchronous Python call where
/// it was constructed. `unsendable` blocks cross-thread escape; the engine
/// drops the ctx as soon as the hook returns.
#[pyclass(unsendable)]
pub struct StrategyCtx {
    state_ptr: *const GameState,
    board_ptr: *const BoardLayout,
}

impl StrategyCtx {
    fn state(&self) -> &GameState {
        unsafe { &*self.state_ptr }
    }
    fn board(&self) -> &BoardLayout {
        unsafe { &*self.board_ptr }
    }
}

#[pymethods]
impl StrategyCtx {
    #[getter]
    fn current_player(&self) -> u8 {
        self.state().acting_player()
    }

    #[getter]
    fn turn_number(&self) -> u16 {
        self.state().turn_number
    }

    #[getter]
    fn robber_hex(&self) -> u8 {
        self.state().robber_hex
    }

    #[getter]
    fn longest_road_player(&self) -> i8 {
        self.state().longest_road_player
    }

    #[getter]
    fn largest_army_player(&self) -> i8 {
        self.state().largest_army_player
    }

    fn player_vp(&self, player: u8) -> u8 {
        self.state().victory_points(player as usize)
    }

    fn player_resources(&self, player: u8) -> [u8; RESOURCE_COUNT] {
        self.state().players[player as usize].resources
    }

    fn player_total_resources(&self, player: u8) -> u8 {
        self.state().players[player as usize].total_resources()
    }

    fn player_knights_played(&self, player: u8) -> u8 {
        self.state().players[player as usize].knights_played
    }

    fn player_settlements(&self, player: u8) -> u32 {
        self.state().settlements[player as usize].count_ones()
    }

    fn player_cities(&self, player: u8) -> u32 {
        self.state().cities[player as usize].count_ones()
    }

    fn player_roads(&self, player: u8) -> u32 {
        self.state().roads[player as usize].count_ones()
    }

    /// `score = sum_h production(h) * 36`. Mirrors the rule-based policy's
    /// helper so research hooks can build on top of the standard scorer.
    fn vertex_production_value(&self, vertex: u32) -> f64 {
        let v = vertex as usize;
        let board = self.board();
        let mut value = 0.0;
        for &h in vertex_hex_ids(v) {
            if h < 0 {
                continue;
            }
            let hex = h as usize;
            if board.hex_resource(hex).is_some() {
                value += production_weight(board.tile_numbers[hex]) * 36.0;
            }
        }
        value
    }

    fn vertex_resource_diversity(&self, vertex: u32) -> u32 {
        let v = vertex as usize;
        let board = self.board();
        let mut mask = 0u8;
        for &h in vertex_hex_ids(v) {
            if h < 0 {
                continue;
            }
            if let Some(r) = board.hex_resource(h as usize) {
                mask |= 1 << (r as u8);
            }
        }
        mask.count_ones()
    }

    fn adjacent_hexes(&self, vertex: u32) -> Vec<u32> {
        vertex_hex_ids(vertex as usize)
            .iter()
            .filter(|&&h| h >= 0)
            .map(|&h| h as u32)
            .collect()
    }

    fn adjacent_vertices(&self, vertex: u32) -> Vec<u32> {
        vertex_neighbors(vertex as usize)
            .iter()
            .filter(|&&n| n >= 0)
            .map(|&n| n as u32)
            .collect()
    }

    fn edge_endpoints(&self, edge: u32) -> (u32, u32) {
        let (a, b) = edge_endpoint_ids(edge as usize);
        (a as u32, b as u32)
    }

    fn is_port(&self, vertex: u32) -> bool {
        self.board().port_at_vertex(vertex as u8).is_some()
    }

    fn port_kind(&self, vertex: u32) -> Option<String> {
        self.board().port_at_vertex(vertex as u8).map(|p| match p {
            PortType::ThreeToOne => "3:1".to_string(),
            PortType::TwoToOne(r) => format!("2:1:{}", resource_name(r)),
        })
    }

    fn hex_production(&self, hex: u32) -> f64 {
        production_weight(self.board().tile_numbers[hex as usize])
    }

    fn hex_resource(&self, hex: u32) -> Option<String> {
        self.board()
            .hex_resource(hex as usize)
            .map(|r| resource_name(r).to_string())
    }

    fn hex_number(&self, hex: u32) -> u8 {
        self.board().tile_numbers[hex as usize]
    }

    fn player_has_building(&self, player: u8, vertex: u32) -> bool {
        self.state()
            .player_has_building(player as usize, vertex as usize)
    }

    fn player_has_settlement(&self, player: u8, vertex: u32) -> bool {
        self.state()
            .player_has_settlement(player as usize, vertex as usize)
    }

    fn player_has_road(&self, player: u8, edge: u32) -> bool {
        self.state()
            .player_has_road(player as usize, edge as usize)
    }

    fn hex_vertices(&self, hex: u32) -> Vec<u32> {
        hex_vertex_ids(hex as usize)
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    /// Phase tag: "setup_settlement" | "setup_road" | "main_action" |
    /// "move_robber" | "steal_resource" | "discard" | "year_of_plenty" |
    /// "monopoly" | "road_building" | "roll_dice" | "finished".
    #[getter]
    fn phase(&self) -> &'static str {
        match self.state().phase {
            GamePhase::SetupSettlement { .. } => "setup_settlement",
            GamePhase::SetupRoad { .. } => "setup_road",
            GamePhase::RollDice => "roll_dice",
            GamePhase::MoveRobber => "move_robber",
            GamePhase::StealResource => "steal_resource",
            GamePhase::Discard { .. } => "discard",
            GamePhase::MainAction => "main_action",
            GamePhase::RoadBuildingPlace { .. } => "road_building",
            GamePhase::YearOfPlentyPick { .. } => "year_of_plenty",
            GamePhase::MonopolyPick => "monopoly",
            GamePhase::Finished => "finished",
        }
    }

    fn vp_totals(&self) -> Vec<u8> {
        (0..PLAYER_COUNT)
            .map(|p| self.state().victory_points(p))
            .collect()
    }
}

fn resource_name(r: Resource) -> &'static str {
    match r {
        Resource::Brick => "brick",
        Resource::Wood => "wood",
        Resource::Wheat => "wheat",
        Resource::Ore => "ore",
        Resource::Sheep => "sheep",
    }
}
