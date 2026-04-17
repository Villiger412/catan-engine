use crate::board::BoardLayout;
use crate::policy::rule_based::RuleBasedPolicy;
use crate::policy::{Policy, RandomPolicy};
use crate::simulation::{simulate_antithetic_pair, simulate_game};
use crate::state::GameState;
use crate::stats::SimulationStats;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
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
    /// Policy to use: "random" or "rule_based".
    pub policy: PolicyType,
    /// Base random seed (each thread gets a derived seed).
    pub seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum PolicyType {
    Random,
    RuleBased,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            n_simulations: 5000,
            n_threads: 0,
            antithetic: true,
            policy: PolicyType::RuleBased,
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
            run_batch_with_policy(board, initial_state, &RuleBasedPolicy, config, &mut stats, &mut rng)
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

fn policy_name(policy: PolicyType) -> String {
    match policy {
        PolicyType::Random => "random".to_string(),
        PolicyType::RuleBased => "rule_based".to_string(),
    }
}

/// Adaptive simulation: keep running until target margin is achieved.
pub fn run_until_converged(
    board: &BoardLayout,
    initial_state: &GameState,
    target_margin: f64,
    max_simulations: u32,
    policy: PolicyType,
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
            seed: 42,
        };

        let result = run_simulation(&board, &state, &config);
        assert_eq!(result.stats.n, 100);
    }
}
