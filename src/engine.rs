use crate::board::{BoardLayout, PLAYER_COUNT};
use crate::policy::mcts::MctsPolicy;
use crate::policy::py_strategy::PyStrategy;
use crate::policy::rule_based::RuleBasedPolicy;
use crate::policy::{PerSeatPolicy, Policy, RandomPolicy, SeatPolicy};
use crate::simulation::{
    simulate_antithetic_pair, simulate_antithetic_pair_record, simulate_game,
    simulate_game_record, GameRecord,
};
use crate::state::GameState;
use crate::stats::SimulationStats;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Configuration for a simulation batch.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of simulations to run.
    pub n_simulations: u32,
    /// Number of parallel threads (0 = auto-detect).
    pub n_threads: usize,
    /// Use antithetic variates for variance reduction.
    pub antithetic: bool,
    /// Policy to use.
    pub policy: PolicyType,
    /// How hard `RuleBased` and MCTS rollouts target the VP leader.
    /// 0.0 = selfish independent best-response; 1.0 = legacy default;
    /// 2.0 = strong focus-fire coalition approximation.
    /// Ignored by `Random`.
    pub coalition_pressure: f64,
    /// Base random seed (each thread gets a derived seed).
    pub seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum PolicyType {
    Random,
    RuleBased,
    /// Flat-UCB MCTS with `RandomPolicy` rollouts. Parameter = rollouts/move.
    Mcts(u32),
    /// Flat-UCB MCTS with `RuleBasedPolicy` rollouts (uses `coalition_pressure`).
    /// Slower but lets coalition-against-the-leader emerge in the tree's
    /// action values, which is the closest the engine gets to multiplayer Nash.
    McRule(u32),
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            n_simulations: 5000,
            n_threads: 0,
            antithetic: true,
            policy: PolicyType::RuleBased,
            coalition_pressure: RuleBasedPolicy::DEFAULT_COALITION_PRESSURE,
            seed: 42,
        }
    }
}

/// Full result of a simulation batch.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub stats: SimulationStats,
    pub elapsed_ms: f64,
    pub games_per_sec: f64,
    pub policy_name: String,
}

/// Run a batch of simulations with the given configuration.
pub fn run_simulation(
    board: &BoardLayout,
    initial_state: &GameState,
    config: &SimulationConfig,
) -> SimulationResult {
    let start = Instant::now();

    if config.n_threads == 1 {
        // Single-threaded path
        let stats = run_batch_single(board, initial_state, config);
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let games_per_sec = stats.n as f64 / elapsed.as_secs_f64();

        return SimulationResult {
            stats,
            elapsed_ms,
            games_per_sec,
            policy_name: policy_name(config.policy),
        };
    }

    // Multi-threaded path using rayon
    let n_threads = if config.n_threads == 0 {
        rayon::current_num_threads()
    } else {
        config.n_threads
    };

    let sims_per_thread = config.n_simulations / n_threads as u32;
    let remainder = config.n_simulations % n_threads as u32;

    let partial_stats: Vec<SimulationStats> = (0..n_threads)
        .into_par_iter()
        .map(|thread_id| {
            let n = sims_per_thread + if (thread_id as u32) < remainder { 1 } else { 0 };
            let thread_seed = config.seed.wrapping_add(thread_id as u64 * 1_000_003);
            let thread_config = SimulationConfig {
                n_simulations: n,
                n_threads: 1,
                antithetic: config.antithetic,
                policy: config.policy,
                coalition_pressure: config.coalition_pressure,
                seed: thread_seed,
            };
            run_batch_single(board, initial_state, &thread_config)
        })
        .collect();

    let mut stats = SimulationStats::new();
    for partial in &partial_stats {
        stats.merge(partial);
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let games_per_sec = stats.n as f64 / elapsed.as_secs_f64();

    SimulationResult {
        stats,
        elapsed_ms,
        games_per_sec,
        policy_name: policy_name(config.policy),
    }
}

/// Run simulations on a single thread.
fn run_batch_single(
    board: &BoardLayout,
    initial_state: &GameState,
    config: &SimulationConfig,
) -> SimulationStats {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut stats = SimulationStats::new();

    match config.policy {
        PolicyType::Random => {
            run_batch_with_policy(board, initial_state, &RandomPolicy, config, &mut stats, &mut rng)
        }
        PolicyType::RuleBased => {
            let p = RuleBasedPolicy::new(config.coalition_pressure);
            run_batch_with_policy(board, initial_state, &p, config, &mut stats, &mut rng)
        }
        PolicyType::Mcts(sims) => {
            run_batch_with_policy(board, initial_state, &MctsPolicy::new(sims), config, &mut stats, &mut rng)
        }
        PolicyType::McRule(sims) => {
            let p = MctsPolicy::with_rule_based_rollout(sims, config.coalition_pressure);
            run_batch_with_policy(board, initial_state, &p, config, &mut stats, &mut rng)
        }
    }

    stats
}

fn run_batch_with_policy<P: Policy>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    config: &SimulationConfig,
    stats: &mut SimulationStats,
    rng: &mut ChaCha8Rng,
) {
    let n = config.n_simulations;

    if config.antithetic {
        // Run pairs: each pair counts as 2 simulations
        let pairs = n / 2;
        for _ in 0..pairs {
            let (r1, r2) = simulate_antithetic_pair(board, initial_state, policy, rng);
            stats.record_win(r1.winner, r1.turns);
            stats.record_win(r2.winner, r2.turns);
        }
        // If odd, run one more regular simulation
        if n % 2 == 1 {
            let result = simulate_game(board, initial_state, policy, rng);
            stats.record_win(result.winner, result.turns);
        }
    } else {
        for _ in 0..n {
            let result = simulate_game(board, initial_state, policy, rng);
            stats.record_win(result.winner, result.turns);
        }
    }
}

/// Build a fresh owned `SeatPolicy` from a `PolicyType`. Each thread
/// constructs its own set (policies own per-instance config but no state).
fn make_seat_policy(p: PolicyType, coalition_pressure: f64) -> SeatPolicy {
    match p {
        PolicyType::Random => SeatPolicy::Random(RandomPolicy),
        PolicyType::RuleBased => SeatPolicy::RuleBased(RuleBasedPolicy::new(coalition_pressure)),
        PolicyType::Mcts(sims) => SeatPolicy::Mcts(MctsPolicy::new(sims)),
        PolicyType::McRule(sims) => SeatPolicy::Mcts(MctsPolicy::with_rule_based_rollout(sims, coalition_pressure)),
    }
}

/// Run a simulation batch with a distinct policy per seat. Used by the
/// PSRO / empirical-game-theoretic analysis pipeline to estimate payoff
/// cells `M[i][j] = "seat 0 plays policy i, other seats play policy j"`.
pub fn run_per_seat_simulation(
    board: &BoardLayout,
    initial_state: &GameState,
    seat_policies: [PolicyType; PLAYER_COUNT],
    coalition_pressures: [f64; PLAYER_COUNT],
    n_simulations: u32,
    n_threads: usize,
    antithetic: bool,
    seed: u64,
) -> SimulationResult {
    let start = Instant::now();
    let threads = if n_threads == 0 { rayon::current_num_threads() } else { n_threads };
    let sims_per_thread = n_simulations / threads as u32;
    let remainder = n_simulations % threads as u32;

    let partial_stats: Vec<SimulationStats> = (0..threads)
        .into_par_iter()
        .map(|thread_id| {
            let n = sims_per_thread + if (thread_id as u32) < remainder { 1 } else { 0 };
            let thread_seed = seed.wrapping_add(thread_id as u64 * 1_000_003);
            let mut rng = ChaCha8Rng::seed_from_u64(thread_seed);
            let mut stats = SimulationStats::new();

            let dispatcher = PerSeatPolicy::new([
                make_seat_policy(seat_policies[0], coalition_pressures[0]),
                make_seat_policy(seat_policies[1], coalition_pressures[1]),
                make_seat_policy(seat_policies[2], coalition_pressures[2]),
                make_seat_policy(seat_policies[3], coalition_pressures[3]),
            ]);

            let config = SimulationConfig {
                n_simulations: n,
                n_threads: 1,
                antithetic,
                policy: PolicyType::RuleBased, // unused when passing explicit policy
                coalition_pressure: coalition_pressures[0], // unused downstream
                seed: thread_seed,
            };
            run_batch_with_policy(board, initial_state, &dispatcher, &config, &mut stats, &mut rng);
            stats
        })
        .collect();

    let mut stats = SimulationStats::new();
    for partial in &partial_stats { stats.merge(partial); }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let games_per_sec = stats.n as f64 / elapsed.as_secs_f64();
    let name = format!(
        "per_seat[{},{},{},{}]",
        policy_name(seat_policies[0]), policy_name(seat_policies[1]),
        policy_name(seat_policies[2]), policy_name(seat_policies[3]),
    );
    SimulationResult { stats, elapsed_ms, games_per_sec, policy_name: name }
}

fn policy_name(policy: PolicyType) -> String {
    match policy {
        PolicyType::Random => "random".to_string(),
        PolicyType::RuleBased => "rule_based".to_string(),
        PolicyType::Mcts(sims) => format!("mcts_{sims}"),
        PolicyType::McRule(sims) => format!("mcts_rule_{sims}"),
    }
}

// ── Records path ──────────────────────────────────────────────────────────
//
// The research-mode pipeline returns one `GameRecord` per game (full
// terminal state per seat) rather than aggregate stats. Strategy spec
// per seat can be either a built-in policy type or a Python-supplied
// `PyStrategy`.

/// Per-seat strategy spec for the records path. `PolicyType` doesn't carry
/// `Arc<PyStrategy>` (that would break its `Copy` impl), so we wrap both
/// possibilities here.
#[derive(Clone)]
pub enum RecordPolicy {
    Builtin(PolicyType),
    Python(Arc<PyStrategy>),
}

impl RecordPolicy {
    fn make_seat(&self, coalition_pressure: f64) -> SeatPolicy {
        match self {
            RecordPolicy::Builtin(p) => make_seat_policy(*p, coalition_pressure),
            RecordPolicy::Python(p) => SeatPolicy::Py(p.clone()),
        }
    }

    fn label(&self) -> String {
        match self {
            RecordPolicy::Builtin(p) => policy_name(*p),
            RecordPolicy::Python(p) => p.label.clone(),
        }
    }
}

/// Result of a `run_records_batch` call.
#[derive(Debug, Clone)]
pub struct RecordsBatchResult {
    pub records: Vec<GameRecord>,
    /// Antithetic-pair index for each record. Same value for both members
    /// of a pair, fresh value otherwise. -1 if antithetic was disabled.
    pub antithetic_pair_id: Vec<i32>,
    pub elapsed_ms: f64,
    pub games_per_sec: f64,
    pub policy_label: String,
}

/// Run a batch of games and return per-game `GameRecord`s.
///
/// `seat_policies` length must equal `PLAYER_COUNT`. If any seat is
/// `RecordPolicy::Python` and that strategy has any hooks set, simulation
/// runs single-threaded inside one rayon job to keep GIL contention sane;
/// pure-Rust seats parallelize as usual.
pub fn run_records_batch(
    board: &BoardLayout,
    initial_state: &GameState,
    seat_policies: &[RecordPolicy; PLAYER_COUNT],
    coalition_pressures: [f64; PLAYER_COUNT],
    n_games: u32,
    n_threads: usize,
    antithetic: bool,
    seed: u64,
) -> RecordsBatchResult {
    let start = Instant::now();

    // Decide thread count. If any seat has live Python hooks, drop to 1
    // thread — pyo3 GIL contention destroys parallel scaling and rule_based
    // is already cheap. Pure-Python-default seats can still parallelize.
    let any_live_py = seat_policies.iter().any(|p| match p {
        RecordPolicy::Python(s) => !s.is_pure_default(),
        _ => false,
    });
    let threads = if any_live_py {
        1
    } else if n_threads == 0 {
        rayon::current_num_threads()
    } else {
        n_threads
    };

    let label = format!(
        "records[{},{},{},{}]",
        seat_policies[0].label(),
        seat_policies[1].label(),
        seat_policies[2].label(),
        seat_policies[3].label(),
    );

    let games_per_thread = n_games / threads as u32;
    let remainder = n_games % threads as u32;

    let partials: Vec<(Vec<GameRecord>, Vec<i32>)> = (0..threads)
        .into_par_iter()
        .map(|thread_id| {
            let n = games_per_thread + if (thread_id as u32) < remainder { 1 } else { 0 };
            let thread_seed = seed.wrapping_add(thread_id as u64 * 1_000_003);
            let mut rng = ChaCha8Rng::seed_from_u64(thread_seed);

            let dispatcher = PerSeatPolicy::new([
                seat_policies[0].make_seat(coalition_pressures[0]),
                seat_policies[1].make_seat(coalition_pressures[1]),
                seat_policies[2].make_seat(coalition_pressures[2]),
                seat_policies[3].make_seat(coalition_pressures[3]),
            ]);

            let mut records: Vec<GameRecord> = Vec::with_capacity(n as usize);
            let mut pair_ids: Vec<i32> = Vec::with_capacity(n as usize);

            // Pair IDs are unique per (thread, pair). Encode as
            //   thread_id * 1_000_000 + local_pair_id
            // so they're globally unique without coordination.
            let pair_base = (thread_id as i32) * 1_000_000;

            if antithetic {
                let pairs = n / 2;
                for i in 0..pairs {
                    let (r1, r2) = simulate_antithetic_pair_record(
                        board, initial_state, &dispatcher, &mut rng,
                    );
                    let pid = pair_base + i as i32;
                    records.push(r1);
                    pair_ids.push(pid);
                    records.push(r2);
                    pair_ids.push(pid);
                }
                if n % 2 == 1 {
                    let r = simulate_game_record(board, initial_state, &dispatcher, &mut rng);
                    records.push(r);
                    pair_ids.push(-1);
                }
            } else {
                for _ in 0..n {
                    let r = simulate_game_record(board, initial_state, &dispatcher, &mut rng);
                    records.push(r);
                    pair_ids.push(-1);
                }
            }

            (records, pair_ids)
        })
        .collect();

    let mut all_records: Vec<GameRecord> = Vec::with_capacity(n_games as usize);
    let mut all_pair_ids: Vec<i32> = Vec::with_capacity(n_games as usize);
    for (recs, pids) in partials {
        all_records.extend(recs);
        all_pair_ids.extend(pids);
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let games_per_sec = all_records.len() as f64 / elapsed.as_secs_f64();

    RecordsBatchResult {
        records: all_records,
        antithetic_pair_id: all_pair_ids,
        elapsed_ms,
        games_per_sec,
        policy_label: label,
    }
}

/// Adaptive simulation: keep running until target margin is achieved.
pub fn run_until_converged(
    board: &BoardLayout,
    initial_state: &GameState,
    target_margin: f64,
    max_simulations: u32,
    policy: PolicyType,
    coalition_pressure: f64,
    seed: u64,
) -> SimulationResult {
    let start = Instant::now();
    let batch_size = 500u32;
    let mut stats = SimulationStats::new();
    let mut current_seed = seed;

    while stats.n < max_simulations {
        let config = SimulationConfig {
            n_simulations: batch_size,
            n_threads: 0,
            antithetic: true,
            policy,
            coalition_pressure,
            seed: current_seed,
        };

        let batch_result = run_simulation(board, initial_state, &config);
        stats.merge(&batch_result.stats);
        current_seed = current_seed.wrapping_add(batch_size as u64 * 7);

        // Check convergence
        if stats.n >= 100 && stats.max_margin() <= target_margin {
            break;
        }
    }

    let elapsed = start.elapsed();
    SimulationResult {
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        games_per_sec: stats.n as f64 / elapsed.as_secs_f64(),
        policy_name: policy_name(policy),
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_simulation_random() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        let config = SimulationConfig {
            n_simulations: 10,
            n_threads: 1,
            antithetic: false,
            policy: PolicyType::Random,
            coalition_pressure: RuleBasedPolicy::DEFAULT_COALITION_PRESSURE,
            seed: 42,
        };

        let result = run_simulation(&board, &state, &config);
        assert_eq!(result.stats.n, 10);
        assert!(result.elapsed_ms > 0.0);
    }

    #[test]
    fn test_run_simulation_parallel() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        let config = SimulationConfig {
            n_simulations: 100,
            n_threads: 2,
            antithetic: true,
            policy: PolicyType::Random,
            coalition_pressure: RuleBasedPolicy::DEFAULT_COALITION_PRESSURE,
            seed: 42,
        };

        let result = run_simulation(&board, &state, &config);
        assert_eq!(result.stats.n, 100);
    }
}
