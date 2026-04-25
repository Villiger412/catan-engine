use crate::actions::{apply_action, legal_actions, Action};
use crate::board::{BoardLayout, PLAYER_COUNT};
use crate::policy::Policy;
use crate::state::{DevCardType, GamePhase, GameState, DEV_CARD_TYPES};
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

/// Per-game outcome record (one row per game in the research export).
///
/// Captured at terminal state via `GameRecord::from_state`. Per-seat fields
/// are length-`PLAYER_COUNT` vectors so the Python side can transpose into
/// `p{i}_*` columns without per-seat field plumbing in Rust.
#[derive(Debug, Clone)]
pub struct GameRecord {
    pub winner: i8, // -1 = draw / timeout
    pub turns: u16,

    pub vp_total: [u8; PLAYER_COUNT],
    pub vp_hidden: [u8; PLAYER_COUNT],
    pub settlements: [u8; PLAYER_COUNT],
    pub cities: [u8; PLAYER_COUNT],
    pub roads: [u8; PLAYER_COUNT],
    pub knights_played: [u8; PLAYER_COUNT],
    pub longest_road_len: [u8; PLAYER_COUNT],
    pub total_resources: [u8; PLAYER_COUNT],
    pub unplayed_dev: [u8; PLAYER_COUNT],

    pub longest_road_player: i8, // -1 = unheld
    pub largest_army_player: i8, // -1 = unheld
}

impl GameRecord {
    pub fn from_state(state: &GameState) -> Self {
        let mut rec = GameRecord {
            winner: state.winner().map(|w| w as i8).unwrap_or(-1),
            turns: state.turn_number,
            vp_total: [0; PLAYER_COUNT],
            vp_hidden: [0; PLAYER_COUNT],
            settlements: [0; PLAYER_COUNT],
            cities: [0; PLAYER_COUNT],
            roads: [0; PLAYER_COUNT],
            knights_played: [0; PLAYER_COUNT],
            longest_road_len: [0; PLAYER_COUNT],
            total_resources: [0; PLAYER_COUNT],
            unplayed_dev: [0; PLAYER_COUNT],
            longest_road_player: state.longest_road_player,
            largest_army_player: state.largest_army_player,
        };
        for p in 0..PLAYER_COUNT {
            rec.vp_total[p] = state.victory_points(p);
            rec.vp_hidden[p] = state.players[p].victory_points_hidden;
            rec.settlements[p] = state.settlements[p].count_ones() as u8;
            rec.cities[p] = state.cities[p].count_ones() as u8;
            rec.roads[p] = state.roads[p].count_ones() as u8;
            rec.knights_played[p] = state.players[p].knights_played;
            rec.longest_road_len[p] = state.longest_road_len[p];
            rec.total_resources[p] = state.players[p].total_resources();
            let mut unplayed = 0u16;
            for i in 0..DEV_CARD_TYPES {
                // VP cards count as unplayed since they're never explicitly played
                if i == DevCardType::VictoryPoint as usize {
                    continue;
                }
                unplayed += state.players[p].dev_cards[i] as u16;
                unplayed += state.players[p].new_dev_cards[i] as u16;
            }
            rec.unplayed_dev[p] = unplayed.min(u8::MAX as u16) as u8;
        }
        rec
    }
}

/// Roll 2d6 and return the sum.
fn roll_2d6<R: Rng>(rng: &mut R) -> u8 {
    let d1 = rng.gen_range(1u8..=6);
    let d2 = rng.gen_range(1u8..=6);
    d1 + d2
}

/// Core game loop, parameterized by how a `RollDice` outcome is produced.
/// Returns the terminal `GameState`. Both the live-rng and antithetic paths
/// project from this single loop instead of duplicating it.
fn run_game_inner<P, R, F>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
    mut next_roll: F,
) -> GameState
where
    P: Policy,
    R: Rng,
    F: FnMut(&mut R) -> u8,
{
    let mut state = initial_state.clone();
    state.shuffle_deck(rng);

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

                let roll = next_roll(rng);
                state.distribute_resources(board, roll);

                if roll == 7 {
                    let mut discard_player = None;
                    for p in 0..PLAYER_COUNT {
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

    state
}

/// Simulate a single complete game from the given state.
pub fn simulate_game<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
) -> GameResult {
    let final_state = run_game_inner(board, initial_state, policy, rng, |r| roll_2d6(r));
    GameResult {
        winner: final_state.winner(),
        turns: final_state.turn_number,
    }
}

/// Same as `simulate_game` but returns the full per-game record.
pub fn simulate_game_record<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
) -> GameRecord {
    let final_state = run_game_inner(board, initial_state, policy, rng, |r| roll_2d6(r));
    GameRecord::from_state(&final_state)
}

/// Simulate a game with antithetic dice rolls (variance reduction).
/// Returns two game results: one with original rolls, one with mirrored rolls.
pub fn simulate_antithetic_pair<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
) -> (GameResult, GameResult) {
    let (s1, s2) = simulate_antithetic_pair_states(board, initial_state, policy, rng);
    (
        GameResult {
            winner: s1.winner(),
            turns: s1.turn_number,
        },
        GameResult {
            winner: s2.winner(),
            turns: s2.turn_number,
        },
    )
}

/// Antithetic-pair variant for the record path.
pub fn simulate_antithetic_pair_record<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
) -> (GameRecord, GameRecord) {
    let (s1, s2) = simulate_antithetic_pair_states(board, initial_state, policy, rng);
    (GameRecord::from_state(&s1), GameRecord::from_state(&s2))
}

/// Shared kernel for both antithetic variants — runs once with the rolled
/// dice sequence and once with `14 - roll` mirrors. The mirror is exactly
/// the antithetic complement of the 2d6 distribution.
fn simulate_antithetic_pair_states<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rng: &mut R,
) -> (GameState, GameState) {
    let mut rolls: Vec<u8> = Vec::with_capacity(300);
    for _ in 0..300 {
        rolls.push(roll_2d6(rng));
    }
    let mirror_rolls: Vec<u8> = rolls.iter().map(|&r| 14 - r).collect();

    let s1 = simulate_with_rolls_state(board, initial_state, policy, &rolls, rng);
    let s2 = simulate_with_rolls_state(board, initial_state, policy, &mirror_rolls, rng);
    (s1, s2)
}

fn simulate_with_rolls_state<P: Policy, R: Rng>(
    board: &BoardLayout,
    initial_state: &GameState,
    policy: &P,
    rolls: &[u8],
    rng: &mut R,
) -> GameState {
    let mut roll_idx = 0usize;
    run_game_inner(board, initial_state, policy, rng, |r| {
        if roll_idx < rolls.len() {
            let v = rolls[roll_idx];
            roll_idx += 1;
            v
        } else {
            roll_2d6(r)
        }
    })
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
        assert!(result.turns <= MAX_TURNS);
    }

    #[test]
    fn test_antithetic_pair() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        let policy = RandomPolicy;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let (r1, r2) = simulate_antithetic_pair(&board, &state, &policy, &mut rng);
        assert!(r1.turns <= MAX_TURNS);
        assert!(r2.turns <= MAX_TURNS);
    }

    #[test]
    fn test_record_path_completes() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        let policy = crate::policy::rule_based::RuleBasedPolicy::default();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let rec = simulate_game_record(&board, &state, &policy, &mut rng);
        assert!(rec.turns <= MAX_TURNS);
        // VP totals are sane: at most 12 (10 to win + slack from hidden VP/army/road).
        for p in 0..PLAYER_COUNT {
            assert!(rec.vp_total[p] <= 14);
        }
    }
}
