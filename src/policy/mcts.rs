use crate::actions::{apply_action, Action};
use crate::board::BoardLayout;
use crate::policy::{Policy, RandomPolicy};
use crate::simulation::simulate_game;
use crate::state::GameState;
use rand::Rng;

/// Flat-UCB MCTS policy (1-ply search).
///
/// For each call to `select_action`, seeds every candidate action with an
/// initial batch of random rollouts, then uses UCB1 to guide the remaining
/// simulation budget. Based on Szita & Chaslot (2010) "Monte-Carlo Tree
/// Search in Settlers of Catan" (best reported result: 49 % vs JSettlers
/// with 10 000 simulations/move).
///
/// Inner rollouts use `RandomPolicy` for speed. Use this policy for
/// best-move selection rather than bulk win-probability estimation.
pub struct MctsPolicy {
    /// Total random rollouts per `select_action` call.
    pub simulations_per_move: u32,
    /// UCB1 exploration constant (sqrt(2) ≈ 1.414 is the textbook default).
    pub exploration: f64,
}

impl MctsPolicy {
    pub fn new(simulations_per_move: u32) -> Self {
        Self {
            simulations_per_move,
            exploration: 1.414,
        }
    }
}

impl Policy for MctsPolicy {
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

        let n = actions.len();
        let player = state.current_player as usize;
        let mut wins = vec![0u32; n];
        let mut visits = vec![0u32; n];

        // Phase 1: seed each action with at least one rollout
        let seed_each = ((self.simulations_per_move as usize / n) + 1)
            .min(self.simulations_per_move as usize);
        for i in 0..n {
            for _ in 0..seed_each {
                wins[i] += rollout(board, state, actions[i], player, rng);
                visits[i] += 1;
            }
        }

        // Phase 2: UCB1-guided remainder
        let used: u32 = visits.iter().sum();
        let remaining = self.simulations_per_move.saturating_sub(used);
        for _ in 0..remaining {
            let total: u32 = visits.iter().sum();
            let best = (0..n)
                .max_by(|&a, &b| {
                    ucb1(wins[a], visits[a], total, self.exploration)
                        .partial_cmp(&ucb1(wins[b], visits[b], total, self.exploration))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            wins[best] += rollout(board, state, actions[best], player, rng);
            visits[best] += 1;
        }

        // Most-visited action is the most robust choice
        (0..n)
            .max_by_key(|&i| visits[i])
            .map(|i| actions[i])
            .unwrap_or(actions[0])
    }
}

fn ucb1(wins: u32, visits: u32, total: u32, c: f64) -> f64 {
    if visits == 0 {
        return f64::INFINITY;
    }
    let exploit = wins as f64 / visits as f64;
    let explore = c * ((total as f64).ln() / visits as f64).sqrt();
    exploit + explore
}

/// Apply `action` to a clone of `state`, run a random rollout to completion,
/// and return 1 if `player` wins, 0 otherwise.
fn rollout<R: Rng>(
    board: &BoardLayout,
    state: &GameState,
    action: Action,
    player: usize,
    rng: &mut R,
) -> u32 {
    let mut next = state.clone();
    apply_action(&mut next, board, action, rng);
    let result = simulate_game(board, &next, &RandomPolicy, rng);
    u32::from(result.winner == Some(player as u8))
}
