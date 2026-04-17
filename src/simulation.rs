use crate::actions::{apply_action, legal_actions, Action};
use crate::board::BoardLayout;
use crate::policy::Policy;
use crate::state::{GamePhase, GameState};
use rand::Rng;

/// Maximum turns before declaring a draw (prevents infinite games).
const MAX_TURNS: u16 = 300;

/// Result of a single simulated game.
#[derive(Debug, Clone)]
pub struct GameResult {
    /// Which player won (None if draw/timeout).
    pub winner: Option<u8>,
    /// Total turns played.
    pub turns: u16,
}

/// Simulate a single complete game from the given state.
pub fn simulate_game<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
) -> GameResult {
    let mut state = initial_state.clone();
    state.shuffle_deck(rng);

    loop {
        if state.is_terminal() || state.turn_number >= MAX_TURNS {
            break;
        }

        match state.phase {
            GamePhase::RollDice => {
                // Check if policy wants to play a knight before rolling
                let actions = legal_actions(&state, board);
                let has_knight = actions
                    .iter()
                    .any(|a| matches!(a, Action::PlayKnight(_)));

                if has_knight {
                    let action = policy.select_action(&state, board, &actions, rng);
                    if matches!(action, Action::PlayKnight(_)) {
                        apply_action(&mut state, board, action, rng);
                        continue;
                    }
                }

                // Roll dice
                let roll = roll_2d6(rng);
                state.distribute_resources(board, roll);

                if roll == 7 {
                    // Check for discard
                    let mut discard_player = None;
                    for p in 0..crate::board::PLAYER_COUNT {
                        if state.players[p].total_resources() > 7 {
                            discard_player = Some(p as u8);
                            break;
                        }
                    }
                    if let Some(dp) = discard_player {
                        state.phase = GamePhase::Discard { player: dp };
                    } else {
                        state.phase = GamePhase::MoveRobber;
                    }
                } else {
                    state.phase = GamePhase::MainAction;
                }
            }

            GamePhase::Finished => break,

            _ => {
                let actions = legal_actions(&state, board);
                if actions.is_empty() {
                    // Shouldn't happen, but safety valve
                    match state.phase {
                        GamePhase::MainAction => {
                            state.next_player();
                        }
                        _ => {
                            state.phase = GamePhase::Finished;
                            break;
                        }
                    }
                    continue;
                }

                let action = policy.select_action(&state, board, &actions, rng);
                if !apply_action(&mut state, board, action, rng) {
                    break; // game over
                }
            }
        }
    }

    GameResult {
        winner: state.winner(),
        turns: state.turn_number,
    }
}

/// Roll 2d6 and return the sum.
fn roll_2d6<R: Rng>(rng: &mut R) -> u8 {
    let d1 = rng.gen_range(1u8..=6);
    let d2 = rng.gen_range(1u8..=6);
    d1 + d2
}

/// Simulate a game with antithetic dice rolls (variance reduction).
/// Returns two game results: one with original rolls, one with mirrored rolls.
pub fn simulate_antithetic_pair<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
) -> (GameResult, GameResult) {
    // Generate a sequence of dice rolls
    let mut rolls: Vec<u8> = Vec::with_capacity(300);
    for _ in 0..300 {
        rolls.push(roll_2d6(rng));
    }

    // Mirror: 14 - roll maps the 2d6 distribution to its complement
    let mirror_rolls: Vec<u8> = rolls.iter().map(|&r| 14 - r).collect();

    let r1 = simulate_with_rolls(board, initial_state, policy, &rolls, rng);
    let r2 = simulate_with_rolls(board, initial_state, policy, &mirror_rolls, rng);

    (r1, r2)
}

/// Simulate a game with predetermined dice rolls.
fn simulate_with_rolls<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rolls: &[u8],
    rng: &mut R,
) -> GameResult {
    let mut state = initial_state.clone();
    state.shuffle_deck(rng);
    let mut roll_idx = 0;

    loop {
        if state.is_terminal() || state.turn_number >= MAX_TURNS {
            break;
        }

        match state.phase {
            GamePhase::RollDice => {
                let actions = legal_actions(&state, board);
                let has_knight = actions
                    .iter()
                    .any(|a| matches!(a, Action::PlayKnight(_)));

                if has_knight {
                    let action = policy.select_action(&state, board, &actions, rng);
                    if matches!(action, Action::PlayKnight(_)) {
                        apply_action(&mut state, board, action, rng);
                        continue;
                    }
                }

                let roll = if roll_idx < rolls.len() {
                    let r = rolls[roll_idx];
                    roll_idx += 1;
                    r
                } else {
                    roll_2d6(rng)
                };

                state.distribute_resources(board, roll);

                if roll == 7 {
                    let mut discard_player = None;
                    for p in 0..crate::board::PLAYER_COUNT {
                        if state.players[p].total_resources() > 7 {
                            discard_player = Some(p as u8);
                            break;
                        }
                    }
                    if let Some(dp) = discard_player {
                        state.phase = GamePhase::Discard { player: dp };
                    } else {
                        state.phase = GamePhase::MoveRobber;
                    }
                } else {
                    state.phase = GamePhase::MainAction;
                }
            }

            GamePhase::Finished => break,

            _ => {
                let actions = legal_actions(&state, board);
                if actions.is_empty() {
                    match state.phase {
                        GamePhase::MainAction => {
                            state.next_player();
                        }
                        _ => {
                            state.phase = GamePhase::Finished;
                            break;
                        }
                    }
                    continue;
                }

                let action = policy.select_action(&state, board, &actions, rng);
                if !apply_action(&mut state, board, action, rng) {
                    break;
                }
            }
        }
    }

    GameResult {
        winner: state.winner(),
        turns: state.turn_number,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::RandomPolicy;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_simulate_random_game() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        let policy = RandomPolicy;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = simulate_game(&board, &state, &policy, &mut rng);
        // Game should terminate (either winner or timeout)
        assert!(result.turns <= MAX_TURNS);
    }

    #[test]
    fn test_antithetic_pair() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        let policy = RandomPolicy;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let (r1, r2) = simulate_antithetic_pair(&board, &state, &policy, &mut rng);
        // Both should terminate
        assert!(r1.turns <= MAX_TURNS);
        assert!(r2.turns <= MAX_TURNS);
    }
}
