pub mod actions;
pub mod board;
pub mod engine;
pub mod policy;
pub mod simulation;
pub mod state;
pub mod stats;

use pyo3::prelude::*;

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
#[pyo3(signature = (n_simulations=5000, n_threads=0, policy="rule_based", antithetic=true, seed=42))]
fn simulate_batch(
    n_simulations: u32,
    n_threads: usize,
    policy: &str,
    antithetic: bool,
    seed: u64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::beginner();
    let initial_state = state::GameState::new(&board);
    let config = engine::SimulationConfig {
        n_simulations, n_threads, antithetic, policy: parse_policy(policy)?, seed,
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
#[pyo3(signature = (target_margin=0.02, max_simulations=50000, policy="rule_based", seed=42))]
fn simulate_until_converged(
    target_margin: f64,
    max_simulations: u32,
    policy: &str,
    seed: u64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::beginner();
    let initial_state = state::GameState::new(&board);
    Ok(win_prob_result(engine::run_until_converged(
        &board, &initial_state, target_margin, max_simulations, parse_policy(policy)?, seed,
    )))
}

/// Run a simulation batch with a custom board supplied as the JSON string
/// produced by `get_board_layout()` (or the board editor).
#[pyfunction]
#[pyo3(signature = (board_json, n_simulations=5000, n_threads=0, policy="rule_based", antithetic=true, seed=42))]
fn simulate_batch_with_board(
    board_json: &str,
    n_simulations: u32,
    n_threads: usize,
    policy: &str,
    antithetic: bool,
    seed: u64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let initial_state = state::GameState::new(&board);
    let policy_type = parse_policy(policy)?;
    let config = engine::SimulationConfig { n_simulations, n_threads, antithetic, policy: policy_type, seed };
    let result = engine::run_simulation(&board, &initial_state, &config);
    Ok(win_prob_result(result))
}

/// Run simulation until converged with a custom board JSON string.
#[pyfunction]
#[pyo3(signature = (board_json, target_margin=0.02, max_simulations=50000, policy="rule_based", seed=42))]
fn simulate_until_converged_with_board(
    board_json: &str,
    target_margin: f64,
    max_simulations: u32,
    policy: &str,
    seed: u64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let initial_state = state::GameState::new(&board);
    let policy_type = parse_policy(policy)?;
    let result = engine::run_until_converged(&board, &initial_state, target_margin, max_simulations, policy_type, seed);
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
#[pyo3(signature = (board_json, position_json, n_simulations=5000, n_threads=0, policy="rule_based", antithetic=true, seed=42))]
fn simulate_from_position(
    board_json: &str,
    position_json: &str,
    n_simulations: u32,
    n_threads: usize,
    policy: &str,
    antithetic: bool,
    seed: u64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let state = parse_position_json(position_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let config = engine::SimulationConfig {
        n_simulations, n_threads, antithetic, policy: parse_policy(policy)?, seed,
    };
    Ok(win_prob_result(engine::run_simulation(&board, &state, &config)))
}

/// Auto-converging version of `simulate_from_position`.
#[pyfunction]
#[pyo3(signature = (board_json, position_json, target_margin=0.02, max_simulations=50000, policy="rule_based", seed=42))]
fn simulate_from_position_converged(
    board_json: &str,
    position_json: &str,
    target_margin: f64,
    max_simulations: u32,
    policy: &str,
    seed: u64,
) -> PyResult<WinProbResult> {
    let board = board::BoardLayout::from_frontend_json(board_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let state = parse_position_json(position_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    Ok(win_prob_result(engine::run_until_converged(
        &board, &state, target_margin, max_simulations, parse_policy(policy)?, seed,
    )))
}

// ── Shared helpers ─────────────────────────────────────────────────────────────

/// Parse a position JSON blob into a mid-game `GameState`.
/// Position JSON: `{ "settlements": [[v,...], ...], "cities": [[v,...], ...], "current_player": 0 }`
fn parse_position_json(json: &str) -> Result<state::GameState, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

    let mut settlements = [0u64; board::PLAYER_COUNT];
    let mut cities = [0u64; board::PLAYER_COUNT];

    let parse_player_verts = |arr: &serde_json::Value, mask: &mut u64| {
        if let Some(verts) = arr.as_array() {
            for vert in verts {
                if let Some(vid) = vert.as_u64() {
                    if vid < board::VERTEX_COUNT as u64 {
                        *mask |= 1u64 << vid;
                    }
                }
            }
        }
    };

    if let Some(s_arr) = v["settlements"].as_array() {
        for (p, player_s) in s_arr.iter().enumerate() {
            if p < board::PLAYER_COUNT {
                parse_player_verts(player_s, &mut settlements[p]);
            }
        }
    }

    if let Some(c_arr) = v["cities"].as_array() {
        for (p, player_c) in c_arr.iter().enumerate() {
            if p < board::PLAYER_COUNT {
                parse_player_verts(player_c, &mut cities[p]);
                // A city replaces the settlement on the same vertex
                settlements[p] &= !cities[p];
            }
        }
    }

    let current_player = v["current_player"].as_u64().unwrap_or(0) as u8;

    // We need a board to call from_mid_game, but it's only used for robber_hex.
    // Use beginner board; the actual board is passed separately to run_simulation.
    let dummy_board = board::BoardLayout::beginner();
    Ok(state::GameState::from_mid_game(dummy_board.desert_hex() as u8, settlements, cities, current_player))
}

fn parse_policy(policy: &str) -> PyResult<engine::PolicyType> {
    match policy {
        "random" => Ok(engine::PolicyType::Random),
        "rule_based" | "rule-based" => Ok(engine::PolicyType::RuleBased),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown policy: '{}'. Use 'random' or 'rule_based'.", policy),
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

/// Python module definition.
#[pymodule]
fn catan_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_until_converged, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_batch_with_board, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_until_converged_with_board, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_from_position, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_from_position_converged, m)?)?;
    m.add_function(wrap_pyfunction!(required_simulations, m)?)?;
    m.add_function(wrap_pyfunction!(get_board_layout, m)?)?;
    m.add_function(wrap_pyfunction!(get_random_board_layout, m)?)?;
    m.add_class::<WinProbResult>()?;
    Ok(())
}
