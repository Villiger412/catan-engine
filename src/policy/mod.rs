pub mod rule_based;

use crate::actions::Action;
use crate::board::BoardLayout;
use crate::state::GameState;
use rand::Rng;

/// Trait for game-playing policies.
pub trait Policy {
    fn select_action<R: Rng>(
        &self,
        state: &GameState,
        board: &BoardLayout,
        actions: &[Action],
        rng: &mut R,
    ) -> Action;
}

/// Random policy: uniformly selects a random legal action.
pub struct RandomPolicy;

impl Policy for RandomPolicy {
    fn select_action<R: Rng>(
        &self,
        _state: &GameState,
        _board: &BoardLayout,
        actions: &[Action],
        rng: &mut R,
    ) -> Action {
        actions[rng.gen_range(0..actions.len())]
    }
}
