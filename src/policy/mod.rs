pub mod mcts;
pub mod py_strategy;
pub mod rule_based;

use crate::actions::Action;
use crate::board::BoardLayout;
use crate::board::PLAYER_COUNT;
use crate::policy::mcts::MctsPolicy;
use crate::policy::py_strategy::PyStrategy;
use crate::policy::rule_based::RuleBasedPolicy;
use crate::state::GameState;
use rand::Rng;
use std::sync::Arc;

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

/// An owned, concrete policy choice that implements `Policy`. Used as the
/// per-seat entry in `PerSeatPolicy`. Avoids trait-object dispatch (which
/// isn't possible here because `Policy::select_action` is generic over
/// `R: Rng`), while still letting different seats play different policies.
pub enum SeatPolicy {
    Random(RandomPolicy),
    RuleBased(RuleBasedPolicy),
    Mcts(MctsPolicy),
    /// Python-supplied strategy. Held in `Arc` because per-thread `SeatPolicy`
    /// instances are built from a shared spec, and `PyStrategy` itself owns
    /// `Py<PyAny>` handles which are cheap to clone but conceptually shared.
    Py(Arc<PyStrategy>),
}

impl Policy for SeatPolicy {
    fn select_action<R: Rng>(
        &self,
        state: &GameState,
        board: &BoardLayout,
        actions: &[Action],
        rng: &mut R,
    ) -> Action {
        match self {
            SeatPolicy::Random(p) => p.select_action(state, board, actions, rng),
            SeatPolicy::RuleBased(p) => p.select_action(state, board, actions, rng),
            SeatPolicy::Mcts(p) => p.select_action(state, board, actions, rng),
            SeatPolicy::Py(p) => p.select_action(state, board, actions, rng),
        }
    }
}

/// Dispatch a distinct `SeatPolicy` per seat. `select_action` looks up
/// `state.acting_player()` (which resolves to the discarding player during
/// `GamePhase::Discard`, the current player otherwise) and delegates to that
/// seat's policy. Implements `Policy` itself so it is a drop-in for the
/// existing simulation paths.
///
/// This is the kernel of the PSRO / empirical-game-theoretic analysis loop:
/// building payoff cells `M[i][j] = "seat 0 plays policy i, other seats play
/// policy j"` requires exactly this kind of per-seat dispatch.
pub struct PerSeatPolicy {
    pub seats: [SeatPolicy; PLAYER_COUNT],
}

impl PerSeatPolicy {
    pub fn new(seats: [SeatPolicy; PLAYER_COUNT]) -> Self {
        Self { seats }
    }
}

impl Policy for PerSeatPolicy {
    fn select_action<R: Rng>(
        &self,
        state: &GameState,
        board: &BoardLayout,
        actions: &[Action],
        rng: &mut R,
    ) -> Action {
        let seat = state.acting_player() as usize;
        self.seats[seat].select_action(state, board, actions, rng)
    }
}
