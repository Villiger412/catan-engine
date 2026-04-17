use catan_engine::board::BoardLayout;
use catan_engine::engine::{run_simulation, PolicyType, SimulationConfig};
use catan_engine::state::GameState;
use catan_engine::stats::SimulationStats;

fn main() {
    let board = BoardLayout::beginner();
    let state = GameState::new(&board);

    println!("=== Catan Win-Probability Engine Benchmark ===\n");
    println!(
        "Minimum sims for ±2% margin at 95% CI: {}\n",
        SimulationStats::required_simulations(0.02)
    );

    // Benchmark random policy
    let config = SimulationConfig {
        n_simulations: 1000,
        n_threads: 1,
        antithetic: false,
        policy: PolicyType::Random,
        seed: 42,
    };
    let result = run_simulation(&board, &state, &config);
    println!("--- Random Policy (1000 games, 1 thread) ---");
    print_result(&result);

    // Benchmark rule-based policy
    let config = SimulationConfig {
        n_simulations: 1000,
        n_threads: 1,
        antithetic: false,
        policy: PolicyType::RuleBased,
        seed: 42,
    };
    let result = run_simulation(&board, &state, &config);
    println!("--- Rule-Based Policy (1000 games, 1 thread) ---");
    print_result(&result);

    // Benchmark rule-based with antithetic variates
    let config = SimulationConfig {
        n_simulations: 5000,
        n_threads: 0, // auto
        antithetic: true,
        policy: PolicyType::RuleBased,
        seed: 42,
    };
    let result = run_simulation(&board, &state, &config);
    println!("--- Rule-Based + Antithetic (5000 games, multi-thread) ---");
    print_result(&result);

    // Large benchmark
    let config = SimulationConfig {
        n_simulations: 10000,
        n_threads: 0,
        antithetic: true,
        policy: PolicyType::RuleBased,
        seed: 42,
    };
    let result = run_simulation(&board, &state, &config);
    println!("--- Rule-Based + Antithetic (10000 games, multi-thread) ---");
    print_result(&result);
}

fn print_result(result: &catan_engine::engine::SimulationResult) {
    let probs = result.stats.win_probabilities();
    let cis = result.stats.confidence_intervals_95();

    println!("  Simulations: {}", result.stats.n);
    println!("  Elapsed: {:.1}ms", result.elapsed_ms);
    println!("  Throughput: {:.0} games/sec", result.games_per_sec);
    println!("  Avg game length: {:.1} turns", result.stats.avg_turns());
    println!("  Draws/timeouts: {}", result.stats.draws);
    println!("  Max margin: ±{:.2}%", result.stats.max_margin() * 100.0);
    println!();
    for i in 0..4 {
        println!(
            "  Player {}: {:.1}% [{:.1}%, {:.1}%]",
            i,
            probs[i] * 100.0,
            cis[i].0 * 100.0,
            cis[i].1 * 100.0
        );
    }
    println!();
}
