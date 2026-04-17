use crate::board::*;
use crate::state::*;
use rand::Rng;

// ── Action Types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Place a settlement at a vertex (setup or normal).
    PlaceSettlement(u8),
    /// Place a road on an edge (setup or normal).
    PlaceRoad(u8),
    /// Upgrade a settlement to a city at a vertex.
    BuildCity(u8),
    /// Buy a development card.
    BuyDevCard,
    /// Play a knight: move robber to hex.
    PlayKnight(u8),
    /// Move robber to a hex (after rolling 7 or playing knight).
    MoveRobber(u8),
    /// Steal from a player at the robber hex.
    StealFrom(u8),
    /// Discard resources (encoded as 5-element cost in a helper).
    Discard([u8; RESOURCE_COUNT]),
    /// Trade with bank (4:1 or port rates): give resource, get resource.
    BankTrade { give: u8, receive: u8 },
    /// Play Year of Plenty: pick a resource.
    YearOfPlentyPick(u8),
    /// Play Monopoly: pick a resource.
    MonopolyPick(u8),
    /// Play Road Building dev card.
    PlayRoadBuilding,
    /// End turn.
    EndTurn,
    /// No action (pass, for discard when player has <= 7).
    Pass,
}

// ── Legal Action Generation ────────────────────────────────────────────────

/// Generate all legal actions for the current game state.
pub fn legal_actions(state: &GameState, board: &BoardLayout) -> Vec<Action> {
    let mut actions = Vec::with_capacity(32);
    let cp = state.current_player as usize;

    match state.phase {
        GamePhase::SetupSettlement { .. } => {
            // Can place settlement on any unoccupied vertex satisfying distance rule
            for v in 0..VERTEX_COUNT {
                if !state.vertex_occupied(v) && state.distance_rule_ok(v) {
                    actions.push(Action::PlaceSettlement(v as u8));
                }
            }
        }

        GamePhase::SetupRoad { .. } => {
            // Must place road adjacent to the just-placed settlement
            // Find the most recent settlement for current player
            let settlements = state.settlements[cp];
            for v in 0..VERTEX_COUNT {
                if (settlements >> v) & 1 == 0 {
                    continue;
                }
                let edges = vertex_edge_ids(v);
                for &e in edges {
                    if e >= 0 {
                        let eidx = e as usize;
                        if (state.all_occupied_edges() >> eidx) & 1 == 0 {
                            actions.push(Action::PlaceRoad(eidx as u8));
                        }
                    }
                }
            }
            // Deduplicate (a vertex may appear for multiple settlements)
            actions.sort_by_key(|a| match a {
                Action::PlaceRoad(e) => *e,
                _ => 255,
            });
            actions.dedup();
        }

        GamePhase::RollDice => {
            // Auto-handled by simulation, but if we need explicit actions:
            // Option to play a knight before rolling
            let p = &state.players[cp];
            if !p.has_played_dev_this_turn && p.dev_cards[DevCardType::Knight as usize] > 0 {
                for hex in 0..HEX_COUNT {
                    if hex != state.robber_hex as usize {
                        actions.push(Action::PlayKnight(hex as u8));
                    }
                }
            }
            // The roll itself is stochastic, not an action. We'll handle it in simulation.
            // But we need at least one action -- "roll" is implicit, represented as Pass.
            if actions.is_empty() {
                actions.push(Action::Pass); // means "roll dice"
            }
        }

        GamePhase::MoveRobber => {
            for hex in 0..HEX_COUNT {
                if hex != state.robber_hex as usize {
                    actions.push(Action::MoveRobber(hex as u8));
                }
            }
        }

        GamePhase::StealResource => {
            // Can steal from any player with buildings on the robber hex
            let vertices = hex_vertex_ids(state.robber_hex as usize);
            let mut targets = Vec::new();
            for &v in vertices.iter() {
                for p in 0..PLAYER_COUNT {
                    if p == cp {
                        continue;
                    }
                    if state.player_has_building(p, v as usize)
                        && state.players[p].total_resources() > 0
                    {
                        targets.push(p as u8);
                    }
                }
            }
            targets.sort();
            targets.dedup();
            if targets.is_empty() {
                actions.push(Action::Pass); // no one to steal from
            } else {
                for t in targets {
                    actions.push(Action::StealFrom(t));
                }
            }
        }

        GamePhase::Discard { player } => {
            let p = &state.players[player as usize];
            if p.total_resources() <= 7 {
                actions.push(Action::Pass);
            } else {
                // Must discard half (rounded down)
                let to_discard = p.total_resources() / 2;
                // Generate a few reasonable discard options (not all combos)
                // For simulation speed, just generate one greedy option
                let discard = greedy_discard(&p.resources, to_discard);
                actions.push(Action::Discard(discard));
            }
        }

        GamePhase::MainAction => {
            let p = &state.players[cp];

            // Build settlement
            if p.can_afford(&SETTLEMENT_COST) && p.settlements_remaining > 0 {
                for v in 0..VERTEX_COUNT {
                    if !state.vertex_occupied(v)
                        && state.distance_rule_ok(v)
                        && state.player_connected_to_vertex(cp, v)
                    {
                        actions.push(Action::PlaceSettlement(v as u8));
                    }
                }
            }

            // Build road
            if p.can_afford(&ROAD_COST) && p.roads_remaining > 0 {
                for e in 0..EDGE_COUNT {
                    if (state.all_occupied_edges() >> e) & 1 == 0
                        && state.player_connected_to_edge(cp, e)
                    {
                        actions.push(Action::PlaceRoad(e as u8));
                    }
                }
            }

            // Build city
            if p.can_afford(&CITY_COST) && p.cities_remaining > 0 {
                for v in 0..VERTEX_COUNT {
                    if state.player_has_settlement(cp, v) {
                        actions.push(Action::BuildCity(v as u8));
                    }
                }
            }

            // Buy dev card
            if p.can_afford(&DEV_CARD_COST) && !state.dev_deck.is_empty() {
                actions.push(Action::BuyDevCard);
            }

            // Play dev cards (one per turn)
            if !p.has_played_dev_this_turn {
                // Knight
                if p.dev_cards[DevCardType::Knight as usize] > 0 {
                    for hex in 0..HEX_COUNT {
                        if hex != state.robber_hex as usize {
                            actions.push(Action::PlayKnight(hex as u8));
                        }
                    }
                }
                // Road building
                if p.dev_cards[DevCardType::RoadBuilding as usize] > 0 && p.roads_remaining > 0 {
                    actions.push(Action::PlayRoadBuilding);
                }
                // Year of plenty
                if p.dev_cards[DevCardType::YearOfPlenty as usize] > 0 {
                    for r in 0..RESOURCE_COUNT {
                        actions.push(Action::YearOfPlentyPick(r as u8));
                    }
                }
                // Monopoly
                if p.dev_cards[DevCardType::Monopoly as usize] > 0 {
                    for r in 0..RESOURCE_COUNT {
                        actions.push(Action::MonopolyPick(r as u8));
                    }
                }
            }

            // Bank trades (4:1 default, 2:1 or 3:1 with ports)
            for give_r in 0..RESOURCE_COUNT {
                let rate = bank_trade_rate(state, board, cp, give_r);
                if p.resources[give_r] >= rate {
                    for recv_r in 0..RESOURCE_COUNT {
                        if recv_r != give_r {
                            actions.push(Action::BankTrade {
                                give: give_r as u8,
                                receive: recv_r as u8,
                            });
                        }
                    }
                }
            }

            // End turn is always an option
            actions.push(Action::EndTurn);
        }

        GamePhase::RoadBuildingPlace { .. } => {
            let p = &state.players[cp];
            if p.roads_remaining > 0 {
                for e in 0..EDGE_COUNT {
                    if (state.all_occupied_edges() >> e) & 1 == 0
                        && state.player_connected_to_edge(cp, e)
                    {
                        actions.push(Action::PlaceRoad(e as u8));
                    }
                }
            }
            if actions.is_empty() {
                actions.push(Action::Pass); // no valid road placement
            }
        }

        GamePhase::YearOfPlentyPick { .. } => {
            for r in 0..RESOURCE_COUNT {
                actions.push(Action::YearOfPlentyPick(r as u8));
            }
        }

        GamePhase::MonopolyPick => {
            for r in 0..RESOURCE_COUNT {
                actions.push(Action::MonopolyPick(r as u8));
            }
        }

        GamePhase::Finished => {}
    }

    actions
}

// ── Action Application ─────────────────────────────────────────────────────

/// Apply an action to the game state. Returns true if the game continues.
pub fn apply_action<R: Rng>(
    state: &mut GameState,
    board: &BoardLayout,
    action: Action,
    rng: &mut R,
) -> bool {
    let cp = state.current_player as usize;

    match action {
        Action::PlaceSettlement(v) => {
            state.settlements[cp] |= 1u64 << v;
            state.players[cp].settlements_remaining -= 1;

            match state.phase {
                GamePhase::SetupSettlement { round } => {
                    // In round 1 (reverse), give initial resources from adjacent hexes
                    if round == 1 {
                        let hexes = vertex_hex_ids(v as usize);
                        for &h in hexes {
                            if h >= 0 {
                                if let Some(res) = board.hex_resource(h as usize) {
                                    state.players[cp].resources[res as usize] += 1;
                                }
                            }
                        }
                    }
                    state.phase = GamePhase::SetupRoad { round };
                }
                GamePhase::MainAction => {
                    state.players[cp].spend(&SETTLEMENT_COST);
                    state.update_longest_road();
                }
                _ => {}
            }
        }

        Action::PlaceRoad(e) => {
            state.roads[cp] |= 1u128 << e;
            state.players[cp].roads_remaining -= 1;

            match state.phase {
                GamePhase::SetupRoad { round: _ } => {
                    advance_setup(state);
                }
                GamePhase::MainAction => {
                    state.players[cp].spend(&ROAD_COST);
                    state.update_longest_road();
                }
                GamePhase::RoadBuildingPlace { remaining } => {
                    if remaining <= 1 {
                        state.phase = GamePhase::MainAction;
                    } else {
                        state.phase = GamePhase::RoadBuildingPlace {
                            remaining: remaining - 1,
                        };
                    }
                    state.update_longest_road();
                }
                _ => {}
            }
        }

        Action::BuildCity(v) => {
            state.settlements[cp] &= !(1u64 << v);
            state.cities[cp] |= 1u64 << v;
            state.players[cp].settlements_remaining += 1; // settlement returned
            state.players[cp].cities_remaining -= 1;
            state.players[cp].spend(&CITY_COST);
        }

        Action::BuyDevCard => {
            state.players[cp].spend(&DEV_CARD_COST);
            if let Some(card) = state.dev_deck.pop() {
                if card == DevCardType::VictoryPoint {
                    state.players[cp].victory_points_hidden += 1;
                } else {
                    state.players[cp].new_dev_cards[card as usize] += 1;
                }
            }
        }

        Action::PlayKnight(hex) => {
            state.players[cp].dev_cards[DevCardType::Knight as usize] -= 1;
            state.players[cp].has_played_dev_this_turn = true;
            state.players[cp].knights_played += 1;
            state.robber_hex = hex;
            state.update_largest_army();
            state.phase = GamePhase::StealResource;
        }

        Action::MoveRobber(hex) => {
            state.robber_hex = hex;
            state.phase = GamePhase::StealResource;
        }

        Action::StealFrom(target) => {
            let target = target as usize;
            let total = state.players[target].total_resources();
            if total > 0 {
                let idx = rng.gen_range(0..total) as u8;
                let mut count = 0u8;
                for r in 0..RESOURCE_COUNT {
                    count += state.players[target].resources[r];
                    if count > idx {
                        state.players[target].resources[r] -= 1;
                        state.players[cp].resources[r] += 1;
                        break;
                    }
                }
            }
            state.phase = GamePhase::MainAction;
        }

        Action::Discard(resources) => {
            if let GamePhase::Discard { player } = state.phase {
                let pidx = player as usize;
                for r in 0..RESOURCE_COUNT {
                    state.players[pidx].resources[r] -= resources[r];
                }
                // Check if more players need to discard
                let next = next_discard_player(state, player + 1);
                if let Some(np) = next {
                    state.phase = GamePhase::Discard { player: np };
                } else {
                    state.phase = GamePhase::MoveRobber;
                }
            }
        }

        Action::BankTrade { give, receive } => {
            let rate = bank_trade_rate(state, board, cp, give as usize);
            state.players[cp].resources[give as usize] -= rate;
            state.players[cp].resources[receive as usize] += 1;
        }

        Action::YearOfPlentyPick(r) => {
            match state.phase {
                GamePhase::MainAction => {
                    // Playing from main action phase
                    state.players[cp].dev_cards[DevCardType::YearOfPlenty as usize] -= 1;
                    state.players[cp].has_played_dev_this_turn = true;
                    state.players[cp].resources[r as usize] += 1;
                    state.phase = GamePhase::YearOfPlentyPick { remaining: 1 };
                }
                GamePhase::YearOfPlentyPick { remaining } => {
                    state.players[cp].resources[r as usize] += 1;
                    if remaining <= 1 {
                        state.phase = GamePhase::MainAction;
                    } else {
                        state.phase = GamePhase::YearOfPlentyPick {
                            remaining: remaining - 1,
                        };
                    }
                }
                _ => {}
            }
        }

        Action::MonopolyPick(r) => {
            match state.phase {
                GamePhase::MainAction => {
                    state.players[cp].dev_cards[DevCardType::Monopoly as usize] -= 1;
                    state.players[cp].has_played_dev_this_turn = true;
                }
                _ => {}
            }
            let r = r as usize;
            let mut total = 0u8;
            for p in 0..PLAYER_COUNT {
                if p != cp {
                    total += state.players[p].resources[r];
                    state.players[p].resources[r] = 0;
                }
            }
            state.players[cp].resources[r] += total;
            state.phase = GamePhase::MainAction;
        }

        Action::PlayRoadBuilding => {
            state.players[cp].dev_cards[DevCardType::RoadBuilding as usize] -= 1;
            state.players[cp].has_played_dev_this_turn = true;
            let remaining = std::cmp::min(2, state.players[cp].roads_remaining);
            state.phase = GamePhase::RoadBuildingPlace { remaining };
        }

        Action::EndTurn => {
            state.next_player();
        }

        Action::Pass => {
            match state.phase {
                GamePhase::RollDice => {} // handled by simulation (dice roll)
                GamePhase::StealResource => {
                    state.phase = GamePhase::MainAction;
                }
                GamePhase::Discard { player } => {
                    let next = next_discard_player(state, player + 1);
                    if let Some(np) = next {
                        state.phase = GamePhase::Discard { player: np };
                    } else {
                        state.phase = GamePhase::MoveRobber;
                    }
                }
                GamePhase::RoadBuildingPlace { .. } => {
                    state.phase = GamePhase::MainAction;
                }
                _ => {}
            }
        }
    }

    // Check for winner
    if state.winner().is_some() {
        state.phase = GamePhase::Finished;
        return false;
    }
    true
}

// ── Helper Functions ───────────────────────────────────────────────────────

/// Get bank trade rate for a resource (4:1 default, lower with ports).
pub fn bank_trade_rate(state: &GameState, board: &BoardLayout, player: usize, resource: usize) -> u8 {
    let mut rate = 4u8;
    let buildings = state.settlements[player] | state.cities[player];
    for v in 0..VERTEX_COUNT {
        if (buildings >> v) & 1 == 0 {
            continue;
        }
        if let Some(port) = board.port_at_vertex(v as u8) {
            match port {
                PortType::ThreeToOne => {
                    if rate > 3 {
                        rate = 3;
                    }
                }
                PortType::TwoToOne(res) if res as usize == resource => {
                    rate = 2;
                    return rate; // can't do better
                }
                _ => {}
            }
        }
    }
    rate
}

/// Greedy discard: discard resources the player has the most of.
fn greedy_discard(resources: &[u8; RESOURCE_COUNT], amount: u8) -> [u8; RESOURCE_COUNT] {
    let mut discard = [0u8; RESOURCE_COUNT];
    let mut remaining = amount;
    let mut available = *resources;

    while remaining > 0 {
        // Find resource with most available
        let mut max_r = 0;
        let mut max_v = 0;
        for r in 0..RESOURCE_COUNT {
            if available[r] > max_v {
                max_v = available[r];
                max_r = r;
            }
        }
        if max_v == 0 {
            break;
        }
        discard[max_r] += 1;
        available[max_r] -= 1;
        remaining -= 1;
    }
    discard
}

/// Find next player who needs to discard (>7 resources).
fn next_discard_player(state: &GameState, start: u8) -> Option<u8> {
    for i in 0..PLAYER_COUNT {
        let p = ((start as usize + i) % PLAYER_COUNT) as u8;
        if state.players[p as usize].total_resources() > 7 {
            return Some(p);
        }
    }
    None
}

/// Advance setup phase to next step.
fn advance_setup(state: &mut GameState) {
    match state.phase {
        GamePhase::SetupRoad { round } => {
            if round == 0 {
                if state.current_player < (PLAYER_COUNT as u8 - 1) {
                    state.current_player += 1;
                    state.phase = GamePhase::SetupSettlement { round: 0 };
                } else {
                    // Start reverse round
                    state.phase = GamePhase::SetupSettlement { round: 1 };
                    // current_player stays at last player
                }
            } else {
                // round == 1, reverse
                if state.current_player > 0 {
                    state.current_player -= 1;
                    state.phase = GamePhase::SetupSettlement { round: 1 };
                } else {
                    // Setup complete, start main game
                    state.current_player = 0;
                    state.phase = GamePhase::RollDice;
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    #[test]
    fn test_setup_actions() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        let actions = legal_actions(&state, &board);
        // In setup, should be able to place settlements on many vertices
        assert!(!actions.is_empty());
        assert!(actions.iter().all(|a| matches!(a, Action::PlaceSettlement(_))));
    }

    #[test]
    fn test_bank_trade_rate_default() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        // No buildings, rate should be 4
        assert_eq!(bank_trade_rate(&state, &board, 0, 0), 4);
    }

    #[test]
    fn test_greedy_discard() {
        let resources = [3, 2, 1, 4, 0]; // total=10, discard 5
        let discard = greedy_discard(&resources, 5);
        let total: u8 = discard.iter().sum();
        assert_eq!(total, 5);
    }
}
