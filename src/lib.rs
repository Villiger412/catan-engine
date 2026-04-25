pub mod actions;
pub mod board;
pub mod engine;
pub mod policy;
pub mod simulation;
pub mod state;
pub mod stats;

use pyo3::prelude::*;
use std::sync::Arc;

/// Python-facing win probability result.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct WinProbResult {
    #[pyo3(get)]
    pub probabilities: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub simulations_run: u32,
    #[pyo3(get)]
    pub elapsed_ms: f64,
    #[pyo3(get)]
    pub games_per_sec: f64,
    #[pyo3(get)]
    pub avg_turns: f64,
    #[pyo3(get)]
    pub draws: u32,
    #[pyo3(get)]
    pub policy: String,
    #[pyo3(get)]
    pub max_margin: f64,
}

/// Run a simulation batch and return win probabilities.
///
/// Args:
///     n_simulations: Number of games to simulate (default: 5000)
///     n_threads: Number of parallel threads (0 = auto, default: 0)
///     policy: "random" or "rule_based" (default: "rule_based")
///     antithetic: Use antithetic variates (default: true)
///     seed: Random seed (default: 42)
///
/// Returns:
///     WinProbResult with probabilities, confidence intervals, and metadata.
#[pyfunction]
#[pyo3(signature = (n_simulations=5000, n_threads=0, policy="rule_based", antithetic=true, seed=42, coalition_pressure=1.0))]
fn simulate_batch(
    n_simulations: u32,
    n_threads: usize,
    policy: &str,
    antithetic: bool,
    seed: u64,
    coalition_pressure: f64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::beginner();
    let initial_state = state::GameState::new(&board);
    let config = engine::SimulationConfig {
        n_simulations, n_threads, antithetic,
        policy: parse_policy(policy)?,
        coalition_pressure,
        seed,
    };
    Ok(win_prob_result(engine::run_simulation(&board, &initial_state, &config)))
}

/// Run simulation until a target margin of error is achieved.
///
/// Args:
///     target_margin: Target half-width of 95% CI (default: 0.02 = +/-2%)
///     max_simulations: Maximum simulations before stopping (default: 50000)
///     policy: "random" or "rule_based" (default: "rule_based")
///     seed: Random seed (default: 42)
#[pyfunction]
#[pyo3(signature = (target_margin=0.02, max_simulations=50000, policy="rule_based", seed=42, coalition_pressure=1.0))]
fn simulate_until_converged(
    target_margin: f64,
    max_simulations: u32,
    policy: &str,
    seed: u64,
    coalition_pressure: f64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::beginner();
    let initial_state = state::GameState::new(&board);
    Ok(win_prob_result(engine::run_until_converged(
        &board, &initial_state, target_margin, max_simulations, parse_policy(policy)?, coalition_pressure, seed,
    )))
}

/// Run a simulation batch with a custom board supplied as the JSON string
/// produced by `get_board_layout()` (or the board editor).
#[pyfunction]
#[pyo3(signature = (board_json, n_simulations=5000, n_threads=0, policy="rule_based", antithetic=true, seed=42, coalition_pressure=1.0))]
fn simulate_batch_with_board(
    board_json: &str,
    n_simulations: u32,
    n_threads: usize,
    policy: &str,
    antithetic: bool,
    seed: u64,
    coalition_pressure: f64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let initial_state = state::GameState::new(&board);
    let policy_type = parse_policy(policy)?;
    let config = engine::SimulationConfig {
        n_simulations, n_threads, antithetic,
        policy: policy_type,
        coalition_pressure,
        seed,
    };
    let result = engine::run_simulation(&board, &initial_state, &config);
    Ok(win_prob_result(result))
}

/// Run simulation until converged with a custom board JSON string.
#[pyfunction]
#[pyo3(signature = (board_json, target_margin=0.02, max_simulations=50000, policy="rule_based", seed=42, coalition_pressure=1.0))]
fn simulate_until_converged_with_board(
    board_json: &str,
    target_margin: f64,
    max_simulations: u32,
    policy: &str,
    seed: u64,
    coalition_pressure: f64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let initial_state = state::GameState::new(&board);
    let policy_type = parse_policy(policy)?;
    let result = engine::run_until_converged(
        &board, &initial_state, target_margin, max_simulations, policy_type, coalition_pressure, seed,
    );
    Ok(win_prob_result(result))
}

/// Get the minimum number of simulations needed for a given margin of error.
#[pyfunction]
#[pyo3(signature = (target_margin=0.02))]
fn required_simulations(target_margin: f64) -> u32 {
    stats::SimulationStats::required_simulations(target_margin)
}

/// Simulate a batch starting from a mid-game position.
///
/// `board_json`: JSON string from `get_board_layout()` or the board editor.
/// `position_json`: `{ "settlements": [[v,...],[v,...],[...],[...]], "cities": [[...],[...],[...],[...]], "current_player": 0 }`
#[pyfunction]
#[pyo3(signature = (board_json, position_json, n_simulations=5000, n_threads=0, policy="rule_based", antithetic=true, seed=42, coalition_pressure=1.0))]
fn simulate_from_position(
    board_json: &str,
    position_json: &str,
    n_simulations: u32,
    n_threads: usize,
    policy: &str,
    antithetic: bool,
    seed: u64,
    coalition_pressure: f64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let state = parse_position_json(position_json, &board)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let config = engine::SimulationConfig {
        n_simulations, n_threads, antithetic,
        policy: parse_policy(policy)?,
        coalition_pressure,
        seed,
    };
    Ok(win_prob_result(engine::run_simulation(&board, &state, &config)))
}

/// Auto-converging version of `simulate_from_position`.
#[pyfunction]
#[pyo3(signature = (board_json, position_json, target_margin=0.02, max_simulations=50000, policy="rule_based", seed=42, coalition_pressure=1.0))]
fn simulate_from_position_converged(
    board_json: &str,
    position_json: &str,
    target_margin: f64,
    max_simulations: u32,
    policy: &str,
    seed: u64,
    coalition_pressure: f64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let state = parse_position_json(position_json, &board)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    Ok(win_prob_result(engine::run_until_converged(
        &board, &state, target_margin, max_simulations, parse_policy(policy)?, coalition_pressure, seed,
    )))
}

/// Run a simulation batch with a distinct policy per seat.
///
/// `seat_policies` is a 4-element list of policy strings (same spec as
/// `policy` on the other functions — `"random"`, `"rule_based"`, `"mcts"`,
/// `"mcts_200"`, `"mcts_rule_500"`, ...). Optional `board_json` /
/// `position_json` override the default beginner board / fresh-game state.
///
/// Returns per-seat win probabilities, with the standard `WinProbResult`
/// fields. Used by the PSRO / empirical-game-theoretic analysis pipeline
/// (`analyzer/psro.py`) to estimate meta-game payoff cells
/// `M[i][j] = win rate of seat 0 under (i, j, j, j)`.
#[pyfunction]
#[pyo3(signature = (
    seat_policies,
    board_json=None,
    position_json=None,
    n_simulations=2000,
    n_threads=0,
    antithetic=true,
    seed=42,
    coalition_pressure=1.0,
    coalition_pressures=None,
))]
fn simulate_per_seat(
    seat_policies: Vec<String>,
    board_json: Option<&str>,
    position_json: Option<&str>,
    n_simulations: u32,
    n_threads: usize,
    antithetic: bool,
    seed: u64,
    coalition_pressure: f64,
    coalition_pressures: Option<Vec<f64>>,
) -> PyResult<WinProbResult> {
    if seat_policies.len() != board::PLAYER_COUNT {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "seat_policies must have exactly {} entries, got {}",
            board::PLAYER_COUNT, seat_policies.len()
        )));
    }
    let mut parsed = [engine::PolicyType::Random; board::PLAYER_COUNT];
    for (i, p) in seat_policies.iter().enumerate() {
        parsed[i] = parse_policy(p)?;
    }
    // Per-seat coalition pressure, falling back to the scalar default.
    // Needed by PSRO when the row and column of the meta-game matrix have
    // different coalition pressures baked in.
    let mut cps = [coalition_pressure; board::PLAYER_COUNT];
    if let Some(v) = coalition_pressures {
        if v.len() != board::PLAYER_COUNT {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "coalition_pressures must have exactly {} entries, got {}",
                board::PLAYER_COUNT, v.len()
            )));
        }
        for (i, cp) in v.iter().enumerate() { cps[i] = *cp; }
    }
    let board = match board_json {
        Some(bj) => board::BoardLayout::from_frontend_json(bj)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?,
        None => board::BoardLayout::beginner(),
    };
    let state = match position_json {
        Some(pj) => parse_position_json(pj, &board)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?,
        None => state::GameState::new(&board),
    };
    Ok(win_prob_result(engine::run_per_seat_simulation(
        &board, &state, parsed, cps,
        n_simulations, n_threads, antithetic, seed,
    )))
}

// ── Shared helpers ─────────────────────────────────────────────────────────────

/// Parse a position JSON blob into a mid-game `GameState`.
///
/// Required: `settlements`, `cities`, `current_player` (legacy schema).
/// Optional extensions: `roads`, `resources`, `unplayed_dev`,
/// `knights_played`, `vp_hidden`, `robber_hex`. All default to empty/zero,
/// and indices out of range are silently clamped.
fn parse_position_json(json: &str, board: &board::BoardLayout) -> Result<state::GameState, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

    let mut pos = state::MidGamePosition {
        robber_hex: board.desert_hex() as u8,
        current_player: v["current_player"].as_u64().unwrap_or(0) as u8,
        ..Default::default()
    };

    let mask_u64 = |arr: &serde_json::Value, limit: u64| -> u64 {
        let mut m = 0u64;
        if let Some(items) = arr.as_array() {
            for item in items {
                if let Some(id) = item.as_u64() {
                    if id < limit {
                        m |= 1u64 << id;
                    }
                }
            }
        }
        m
    };
    let mask_u128 = |arr: &serde_json::Value, limit: u64| -> u128 {
        let mut m = 0u128;
        if let Some(items) = arr.as_array() {
            for item in items {
                if let Some(id) = item.as_u64() {
                    if id < limit {
                        m |= 1u128 << id;
                    }
                }
            }
        }
        m
    };
    let parse_u8_vec = |arr: &serde_json::Value, slot: &mut [u8]| {
        if let Some(items) = arr.as_array() {
            for (i, item) in items.iter().enumerate() {
                if i >= slot.len() { break; }
                if let Some(n) = item.as_u64() {
                    slot[i] = n.min(u8::MAX as u64) as u8;
                }
            }
        }
    };

    if let Some(arr) = v["settlements"].as_array() {
        for (p, player_s) in arr.iter().enumerate() {
            if p < board::PLAYER_COUNT {
                pos.settlements[p] = mask_u64(player_s, board::VERTEX_COUNT as u64);
            }
        }
    }
    if let Some(arr) = v["cities"].as_array() {
        for (p, player_c) in arr.iter().enumerate() {
            if p < board::PLAYER_COUNT {
                pos.cities[p] = mask_u64(player_c, board::VERTEX_COUNT as u64);
            }
        }
    }
    if let Some(arr) = v["roads"].as_array() {
        for (p, player_r) in arr.iter().enumerate() {
            if p < board::PLAYER_COUNT {
                pos.roads[p] = mask_u128(player_r, board::EDGE_COUNT as u64);
            }
        }
    }
    if let Some(arr) = v["resources"].as_array() {
        for (p, player_r) in arr.iter().enumerate() {
            if p < board::PLAYER_COUNT {
                parse_u8_vec(player_r, &mut pos.resources[p]);
            }
        }
    }
    if let Some(arr) = v["unplayed_dev"].as_array() {
        for (p, player_d) in arr.iter().enumerate() {
            if p < board::PLAYER_COUNT {
                parse_u8_vec(player_d, &mut pos.unplayed_dev[p]);
            }
        }
    }
    parse_u8_vec(&v["knights_played"], &mut pos.knights_played);
    parse_u8_vec(&v["vp_hidden"], &mut pos.vp_hidden);

    if let Some(hex) = v["robber_hex"].as_u64() {
        if hex < board::HEX_COUNT as u64 {
            pos.robber_hex = hex as u8;
        }
    }

    Ok(state::GameState::from_mid_game_full(&pos))
}

fn parse_policy(policy: &str) -> PyResult<engine::PolicyType> {
    match policy {
        "random" => Ok(engine::PolicyType::Random),
        "rule_based" | "rule-based" => Ok(engine::PolicyType::RuleBased),
        "mcts" => Ok(engine::PolicyType::Mcts(200)),
        "mcts_rule" => Ok(engine::PolicyType::McRule(200)),
        _ if policy.starts_with("mcts_rule_") => {
            let sims: u32 = policy[10..].parse().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid mcts_rule spec '{}'. Use 'mcts_rule' or 'mcts_rule_<N>'.", policy),
                )
            })?;
            Ok(engine::PolicyType::McRule(sims))
        }
        _ if policy.starts_with("mcts_") => {
            let sims: u32 = policy[5..].parse().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid mcts spec '{}'. Use 'mcts' or 'mcts_<N>' (e.g. 'mcts_500').", policy),
                )
            })?;
            Ok(engine::PolicyType::Mcts(sims))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!(
                "Unknown policy: '{}'. Use 'random', 'rule_based', 'mcts[_<N>]', or 'mcts_rule[_<N>]'.",
                policy
            ),
        )),
    }
}

fn win_prob_result(result: engine::SimulationResult) -> WinProbResult {
    let probs = result.stats.win_probabilities();
    let cis = result.stats.confidence_intervals_95();
    WinProbResult {
        probabilities: probs.to_vec(),
        ci_lower: cis.iter().map(|(lo, _)| *lo).collect(),
        ci_upper: cis.iter().map(|(_, hi)| *hi).collect(),
        simulations_run: result.stats.n,
        elapsed_ms: result.elapsed_ms,
        games_per_sec: result.games_per_sec,
        avg_turns: result.stats.avg_turns(),
        draws: result.stats.draws,
        policy: result.policy_name,
        max_margin: result.stats.max_margin(),
    }
}

/// Return a random legal Catan board as a JSON string (same format as `get_board_layout()`).
///
/// The board uses the standard tile distribution and number tokens.
/// The "no adjacent red numbers" rule (no two neighboring 6s or 8s) is enforced.
/// Ports are shuffled across the 9 fixed outer-edge slots.
///
/// Args:
///     seed: Random seed for reproducibility (default: 42)
#[pyfunction]
#[pyo3(signature = (seed=42))]
fn get_random_board_layout(seed: u64) -> String {
    use board::BoardLayout;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let b = BoardLayout::random(&mut rng);
    board_layout_to_json(&b)
}

fn board_layout_to_json(b: &board::BoardLayout) -> String {
    use board::{TileType, PortType, Resource, HEX_COORDS};

    let hexes: Vec<String> = (0..board::HEX_COUNT)
        .map(|i| {
            let resource = match b.tile_types[i] {
                TileType::Wood => "wood",
                TileType::Brick => "brick",
                TileType::Wheat => "wheat",
                TileType::Ore => "ore",
                TileType::Sheep => "sheep",
                TileType::Desert => "desert",
            };
            let coord = HEX_COORDS[i];
            format!(
                r#"{{"id":{},"resource":"{}","number":{},"q":{},"r":{}}}"#,
                i, resource, b.tile_numbers[i], coord.q, coord.r
            )
        })
        .collect();

    let ports: Vec<String> = b
        .ports
        .iter()
        .map(|(pt, v1, v2)| {
            let type_str = match pt {
                PortType::ThreeToOne => "3:1".to_string(),
                PortType::TwoToOne(r) => format!(
                    "2:1:{}",
                    match r {
                        Resource::Brick => "brick",
                        Resource::Wood => "wood",
                        Resource::Wheat => "wheat",
                        Resource::Ore => "ore",
                        Resource::Sheep => "sheep",
                    }
                ),
            };
            format!(r#"{{"type":"{}","v1":{},"v2":{}}}"#, type_str, v1, v2)
        })
        .collect();

    format!(
        r#"{{"hexes":[{}],"ports":[{}]}}"#,
        hexes.join(","),
        ports.join(",")
    )
}

/// Return the beginner board layout as a JSON string for frontend rendering.
/// Includes hex tile types, number tokens, axial coordinates, and port data.
#[pyfunction]
fn get_board_layout() -> String {
    board_layout_to_json(&board::BoardLayout::beginner())
}

// ── Research-mode records pipeline ─────────────────────────────────────────

/// Columnar (struct-of-arrays) result for `simulate_games_records`.
///
/// SoA over AoS because pandas/numpy build DataFrames roughly 10× faster
/// from one column at a time than from a list of dicts. Per-seat fields are
/// `Vec<Vec<u8>>` of shape `[n_games][PLAYER_COUNT]`; the Python side
/// transposes them into `p{i}_*` columns.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct GameRecordsResult {
    #[pyo3(get)]
    pub winner: Vec<i8>,
    #[pyo3(get)]
    pub turns: Vec<u16>,
    #[pyo3(get)]
    pub antithetic_pair_id: Vec<i32>,

    // Per-seat fields use `u16` rather than `u8`: pyo3 converts `Vec<u8>` to
    // Python `bytes` (which numpy then can't `asarray` as a 2-D int matrix
    // without per-row decoding). `u16` lifts that special-case.
    #[pyo3(get)]
    pub vp_total: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub vp_hidden: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub settlements: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub cities: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub roads: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub knights_played: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub longest_road_len: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub total_resources: Vec<Vec<u16>>,
    #[pyo3(get)]
    pub unplayed_dev: Vec<Vec<u16>>,

    #[pyo3(get)]
    pub longest_road_player: Vec<i8>,
    #[pyo3(get)]
    pub largest_army_player: Vec<i8>,

    #[pyo3(get)]
    pub elapsed_ms: f64,
    #[pyo3(get)]
    pub games_per_sec: f64,
    #[pyo3(get)]
    pub policy_label: String,
    #[pyo3(get)]
    pub seed: u64,
}

fn parse_record_policy(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    coalition_pressure: f64,
) -> PyResult<engine::RecordPolicy> {
    // String: built-in policy by name.
    if let Ok(name) = obj.extract::<String>() {
        return Ok(engine::RecordPolicy::Builtin(parse_policy(&name)?));
    }
    // Object with `_overridden_hooks` attr → treat as a Strategy instance.
    if obj.hasattr("_overridden_hooks").unwrap_or(false) {
        let strat = policy::py_strategy::PyStrategy::from_pyobject(py, obj, coalition_pressure)?;
        return Ok(engine::RecordPolicy::Python(Arc::new(strat)));
    }
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "strategy must be a policy name string (e.g. 'rule_based') or a \
         catan_research.Strategy instance",
    ))
}

/// Run a batch of games and return per-game records as a columnar result.
///
/// `strategy`: a single strategy used for all 4 seats. May be a built-in
/// policy name (`"rule_based"`, `"mcts_200"`, ...) or a
/// `catan_research.Strategy` instance.
///
/// `seat_strategies`: optional 4-element list of the same. Overrides
/// `strategy` if provided.
///
/// Returns a `GameRecordsResult` whose per-seat fields are length-`n_games`
/// vectors of length-4 vectors — Python expands them to `p{i}_*` columns.
#[pyfunction]
#[pyo3(signature = (
    n_games=5000,
    strategy=None,
    seat_strategies=None,
    n_threads=0,
    antithetic=true,
    seed=42,
    coalition_pressure=1.0,
    board_json=None,
    position_json=None,
))]
#[allow(clippy::too_many_arguments)]
fn simulate_games_records(
    py: Python<'_>,
    n_games: u32,
    strategy: Option<Py<PyAny>>,
    seat_strategies: Option<Vec<Py<PyAny>>>,
    n_threads: usize,
    antithetic: bool,
    seed: u64,
    coalition_pressure: f64,
    board_json: Option<&str>,
    position_json: Option<&str>,
) -> PyResult<GameRecordsResult> {
    let board = match board_json {
        Some(bj) => board::BoardLayout::from_frontend_json(bj)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?,
        None => board::BoardLayout::beginner(),
    };
    let initial_state = match position_json {
        Some(pj) => parse_position_json(pj, &board)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?,
        None => state::GameState::new(&board),
    };

    let seat_specs: [engine::RecordPolicy; board::PLAYER_COUNT] = match seat_strategies {
        Some(list) => {
            if list.len() != board::PLAYER_COUNT {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "seat_strategies must have exactly {} entries, got {}",
                    board::PLAYER_COUNT,
                    list.len()
                )));
            }
            let mut out: [Option<engine::RecordPolicy>; board::PLAYER_COUNT] = Default::default();
            for (i, item) in list.iter().enumerate() {
                let bound = item.bind(py);
                out[i] =
                    Some(parse_record_policy(py, bound, coalition_pressure)?);
            }
            [
                out[0].take().unwrap(),
                out[1].take().unwrap(),
                out[2].take().unwrap(),
                out[3].take().unwrap(),
            ]
        }
        None => {
            let one = match strategy {
                Some(obj) => parse_record_policy(py, obj.bind(py), coalition_pressure)?,
                None => engine::RecordPolicy::Builtin(engine::PolicyType::RuleBased),
            };
            [one.clone(), one.clone(), one.clone(), one]
        }
    };

    let cps = [coalition_pressure; board::PLAYER_COUNT];

    // Long Python callbacks would deadlock GIL-poll; release while running.
    let result = py.detach(|| {
        engine::run_records_batch(
            &board,
            &initial_state,
            &seat_specs,
            cps,
            n_games,
            n_threads,
            antithetic,
            seed,
        )
    });

    // Project AoS records into SoA columns for fast DataFrame construction.
    let n = result.records.len();
    let mut winner = Vec::with_capacity(n);
    let mut turns = Vec::with_capacity(n);
    let mut vp_total = Vec::with_capacity(n);
    let mut vp_hidden = Vec::with_capacity(n);
    let mut settlements = Vec::with_capacity(n);
    let mut cities = Vec::with_capacity(n);
    let mut roads = Vec::with_capacity(n);
    let mut knights_played = Vec::with_capacity(n);
    let mut longest_road_len = Vec::with_capacity(n);
    let mut total_resources = Vec::with_capacity(n);
    let mut unplayed_dev = Vec::with_capacity(n);
    let mut longest_road_player = Vec::with_capacity(n);
    let mut largest_army_player = Vec::with_capacity(n);

    let widen = |a: &[u8; board::PLAYER_COUNT]| -> Vec<u16> {
        a.iter().map(|&v| v as u16).collect()
    };
    for rec in result.records.iter() {
        winner.push(rec.winner);
        turns.push(rec.turns);
        vp_total.push(widen(&rec.vp_total));
        vp_hidden.push(widen(&rec.vp_hidden));
        settlements.push(widen(&rec.settlements));
        cities.push(widen(&rec.cities));
        roads.push(widen(&rec.roads));
        knights_played.push(widen(&rec.knights_played));
        longest_road_len.push(widen(&rec.longest_road_len));
        total_resources.push(widen(&rec.total_resources));
        unplayed_dev.push(widen(&rec.unplayed_dev));
        longest_road_player.push(rec.longest_road_player);
        largest_army_player.push(rec.largest_army_player);
    }

    Ok(GameRecordsResult {
        winner,
        turns,
        antithetic_pair_id: result.antithetic_pair_id,
        vp_total,
        vp_hidden,
        settlements,
        cities,
        roads,
        knights_played,
        longest_road_len,
        total_resources,
        unplayed_dev,
        longest_road_player,
        largest_army_player,
        elapsed_ms: result.elapsed_ms,
        games_per_sec: result.games_per_sec,
        policy_label: result.policy_label,
        seed,
    })
}

/// Python module definition.
#[pymodule]
fn catan_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_until_converged, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_batch_with_board, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_until_converged_with_board, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_from_position, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_from_position_converged, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_per_seat, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_games_records, m)?)?;
    m.add_function(wrap_pyfunction!(required_simulations, m)?)?;
    m.add_function(wrap_pyfunction!(get_board_layout, m)?)?;
    m.add_function(wrap_pyfunction!(get_random_board_layout, m)?)?;
    m.add_class::<WinProbResult>()?;
    m.add_class::<GameRecordsResult>()?;
    m.add_class::<policy::py_strategy::StrategyCtx>()?;
    Ok(())
}
