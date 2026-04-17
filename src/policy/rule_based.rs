use crate::actions::*;
use crate::board::*;
use crate::policy::Policy;
use crate::state::*;
use rand::Rng;

/// T1 Rule-Based Expert Policy.
///
/// Encodes strong heuristics for Catan play:
/// - Settlement placement: score by production value, diversity, ports
/// - Action priority: city > settlement > dev card > road > end turn
/// - Trading: goal-directed bank trades
pub struct RuleBasedPolicy;

impl Policy for RuleBasedPolicy {
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

        match state.phase {
            GamePhase::SetupSettlement { .. } => select_setup_settlement(state, board, actions),
            GamePhase::SetupRoad { .. } => select_setup_road(state, board, actions, rng),
            GamePhase::MoveRobber => select_robber_hex(state, board, actions),
            GamePhase::StealResource => select_steal_target(state, actions),
            GamePhase::Discard { .. } => actions[0], // greedy discard already computed
            GamePhase::MainAction => select_main_action(state, board, actions, rng),
            GamePhase::RoadBuildingPlace { .. } => select_road_placement(state, board, actions, rng),
            GamePhase::YearOfPlentyPick { .. } => select_year_of_plenty(state, actions),
            GamePhase::MonopolyPick => select_monopoly(state, actions),
            _ => actions[rng.gen_range(0..actions.len())],
        }
    }
}

// ── Resource Valuation ─────────────────────────────────────────────────────

/// Strategic resource values (tunable).
const RESOURCE_VALUE: [f64; RESOURCE_COUNT] = [
    1.0, // Brick
    1.0, // Wood
    1.1, // Wheat
    1.2, // Ore
    0.8, // Sheep
];

/// Score a vertex for settlement placement.
fn vertex_score(board: &BoardLayout, _state: &GameState, vertex: usize, _player: usize) -> f64 {
    let hexes = vertex_hex_ids(vertex);
    let mut score = 0.0;
    let mut resource_types = 0u8; // bitmask of resource types seen

    for &h in hexes {
        if h < 0 {
            continue;
        }
        let hex = h as usize;
        if let Some(res) = board.hex_resource(hex) {
            let prod = production_weight(board.tile_numbers[hex]);
            score += RESOURCE_VALUE[res as usize] * prod * 36.0; // scale for readability
            resource_types |= 1 << (res as u8);
        }
    }

    // Diversity bonus: +1.5 per unique resource type
    let diversity = resource_types.count_ones() as f64;
    score += diversity * 1.5;

    // Port bonus
    if let Some(port) = board.port_at_vertex(vertex as u8) {
        match port {
            PortType::ThreeToOne => score += 1.0,
            PortType::TwoToOne(_) => score += 2.0,
        }
    }

    score
}

// ── Setup Phase ────────────────────────────────────────────────────────────

fn select_setup_settlement(state: &GameState, board: &BoardLayout, actions: &[Action]) -> Action {
    let cp = state.current_player as usize;
    let mut best_action = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &action in actions {
        if let Action::PlaceSettlement(v) = action {
            let score = vertex_score(board, state, v as usize, cp);
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
    }
    best_action
}

fn select_setup_road<R: Rng>(
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
    _rng: &mut R,
) -> Action {
    // Place road toward the best neighboring unoccupied vertex
    let cp = state.current_player as usize;
    let mut best_action = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &action in actions {
        if let Action::PlaceRoad(e) = action {
            let (v1, v2) = edge_endpoint_ids(e as usize);
            // Score based on what vertices this road leads toward
            let s1 = if !state.vertex_occupied(v1 as usize) {
                vertex_score(board, state, v1 as usize, cp)
            } else {
                0.0
            };
            let s2 = if !state.vertex_occupied(v2 as usize) {
                vertex_score(board, state, v2 as usize, cp)
            } else {
                0.0
            };
            let score = s1.max(s2);
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
    }
    best_action
}

// ── Robber Placement ───────────────────────────────────────────────────────

fn select_robber_hex(state: &GameState, board: &BoardLayout, actions: &[Action]) -> Action {
    let cp = state.current_player as usize;
    let mut best_action = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &action in actions {
        if let Action::MoveRobber(hex) = action {
            // Prefer hexes that hurt leading opponents with high-production numbers
            let vertices = hex_vertex_ids(hex as usize);
            let mut score = 0.0;
            let prod = production_weight(board.tile_numbers[hex as usize]) * 36.0;

            for &v in vertices.iter() {
                for p in 0..PLAYER_COUNT {
                    if p == cp {
                        // Don't block ourselves
                        score -= prod * 5.0;
                        continue;
                    }
                    if state.player_has_building(p, v as usize) {
                        // Higher score for blocking leading players on productive hexes
                        let their_vp = state.victory_points(p) as f64;
                        score += prod * (1.0 + their_vp * 0.3);
                    }
                }
            }
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
    }
    best_action
}

fn select_steal_target(state: &GameState, actions: &[Action]) -> Action {
    let mut best_action = actions[0];
    let mut best_score = 0i32;

    for &action in actions {
        if let Action::StealFrom(target) = action {
            // Prefer stealing from players with more resources and more VP
            let t = target as usize;
            let score = state.players[t].total_resources() as i32
                + state.victory_points(t) as i32 * 2;
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
    }
    best_action
}

// ── Main Action Phase ──────────────────────────────────────────────────────

fn select_main_action<R: Rng>(
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
    rng: &mut R,
) -> Action {
    let cp = state.current_player as usize;

    // Priority ordering: city > settlement > dev card > road building card > bank trade > road > end turn
    // Categorize available actions
    let mut cities: Vec<Action> = Vec::new();
    let mut settlements: Vec<Action> = Vec::new();
    let mut roads: Vec<Action> = Vec::new();
    let mut dev_card_buy = false;
    let mut knights: Vec<Action> = Vec::new();
    let mut road_building = false;
    let mut yop: Vec<Action> = Vec::new();
    let mut monopoly: Vec<Action> = Vec::new();
    let mut bank_trades: Vec<Action> = Vec::new();
    for &action in actions {
        match action {
            Action::BuildCity(_) => cities.push(action),
            Action::PlaceSettlement(_) => settlements.push(action),
            Action::PlaceRoad(_) => roads.push(action),
            Action::BuyDevCard => dev_card_buy = true,
            Action::PlayKnight(_) => knights.push(action),
            Action::PlayRoadBuilding => road_building = true,
            Action::YearOfPlentyPick(_) => yop.push(action),
            Action::MonopolyPick(_) => monopoly.push(action),
            Action::BankTrade { .. } => bank_trades.push(action),
            Action::EndTurn => {}
            _ => {}
        }
    }

    // 1. Build city (highest ROI: +1 VP, doubles production)
    if !cities.is_empty() {
        // Pick city on the highest-production vertex
        return best_by_vertex_score(&cities, board, state, cp);
    }

    // 2. Build settlement
    if !settlements.is_empty() {
        return best_by_vertex_score(&settlements, board, state, cp);
    }

    // 3. Play knight if we're close to largest army
    if !knights.is_empty() {
        let our_knights = state.players[cp].knights_played;
        let need_army = if state.largest_army_player < 0 {
            our_knights >= 2
        } else {
            our_knights >= state.players[state.largest_army_player as usize].knights_played
        };
        if need_army {
            // Pick best robber hex
            return select_robber_hex_from_knights(state, board, &knights, cp);
        }
    }

    // 4. Buy dev card
    if dev_card_buy {
        return Action::BuyDevCard;
    }

    // 5. Bank trade toward a build goal
    if !bank_trades.is_empty() {
        if let Some(trade) = goal_directed_trade(state, board, cp, &bank_trades) {
            return trade;
        }
    }

    // 6. Build road toward a good vertex
    if !roads.is_empty() && state.players[cp].settlements_remaining > 0 {
        return select_road_placement(state, board, &roads, rng);
    }

    // 7. Play road building card
    if road_building {
        return Action::PlayRoadBuilding;
    }

    // 8. Year of plenty / Monopoly (play them rather than waste)
    if !yop.is_empty() {
        return select_year_of_plenty(state, &yop);
    }
    if !monopoly.is_empty() {
        return select_monopoly(state, &monopoly);
    }

    // 9. End turn
    Action::EndTurn
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn best_by_vertex_score(
    actions: &[Action],
    board: &BoardLayout,
    state: &GameState,
    player: usize,
) -> Action {
    let mut best = actions[0];
    let mut best_score = f64::NEG_INFINITY;
    for &a in actions {
        let v = match a {
            Action::PlaceSettlement(v) | Action::BuildCity(v) => v,
            _ => continue,
        };
        let score = vertex_score(board, state, v as usize, player);
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

fn select_robber_hex_from_knights(
    state: &GameState,
    board: &BoardLayout,
    knights: &[Action],
    cp: usize,
) -> Action {
    let mut best = knights[0];
    let mut best_score = f64::NEG_INFINITY;
    for &a in knights {
        if let Action::PlayKnight(hex) = a {
            let vertices = hex_vertex_ids(hex as usize);
            let prod = production_weight(board.tile_numbers[hex as usize]) * 36.0;
            let mut score = 0.0;
            for &v in vertices.iter() {
                for p in 0..PLAYER_COUNT {
                    if p == cp {
                        score -= prod * 5.0;
                    } else if state.player_has_building(p, v as usize) {
                        score += prod * (1.0 + state.victory_points(p) as f64 * 0.3);
                    }
                }
            }
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
    }
    best
}

fn select_road_placement<R: Rng>(
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
    _rng: &mut R,
) -> Action {
    let cp = state.current_player as usize;
    let mut best = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &a in actions {
        if let Action::PlaceRoad(e) = a {
            let (v1, v2) = edge_endpoint_ids(e as usize);
            // Score: how good are the vertices this road leads to?
            let mut score = 0.0;
            for v in [v1, v2] {
                if !state.vertex_occupied(v as usize) && state.distance_rule_ok(v as usize) {
                    score += vertex_score(board, state, v as usize, cp) * 0.5;
                }
            }
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
    }
    best
}

/// Try to find a bank trade that moves us toward building something.
fn goal_directed_trade(
    state: &GameState,
    _board: &BoardLayout,
    player: usize,
    trades: &[Action],
) -> Option<Action> {
    let p = &state.players[player];

    // Check what we're closest to affording
    let goals: [(&[u8; RESOURCE_COUNT], &str); 3] = [
        (&CITY_COST, "city"),
        (&SETTLEMENT_COST, "settlement"),
        (&DEV_CARD_COST, "dev_card"),
    ];

    for (cost, _name) in &goals {
        // Find which resource we're missing
        let mut missing = [0u8; RESOURCE_COUNT];
        let mut total_missing = 0u8;
        for r in 0..RESOURCE_COUNT {
            if p.resources[r] < cost[r] {
                missing[r] = cost[r] - p.resources[r];
                total_missing += missing[r];
            }
        }

        if total_missing == 0 {
            continue; // can already afford
        }
        if total_missing > 2 {
            continue; // too far away
        }

        // Find a trade that provides a missing resource
        for &trade in trades {
            if let Action::BankTrade { give: _, receive } = trade {
                if missing[receive as usize] > 0 {
                    return Some(trade);
                }
            }
        }
    }
    None
}

fn select_year_of_plenty(state: &GameState, actions: &[Action]) -> Action {
    let cp = state.current_player as usize;
    let p = &state.players[cp];

    // Pick the resource we have the least of (that we need for builds)
    let mut best = actions[0];
    let mut best_need = f64::NEG_INFINITY;
    for &a in actions {
        if let Action::YearOfPlentyPick(r) = a {
            // Higher need = more valuable
            let need = RESOURCE_VALUE[r as usize] / (p.resources[r as usize] as f64 + 1.0);
            if need > best_need {
                best_need = need;
                best = a;
            }
        }
    }
    best
}

fn select_monopoly(state: &GameState, actions: &[Action]) -> Action {
    let cp = state.current_player as usize;
    // Pick the resource that opponents have the most of (weighted by value)
    let mut best = actions[0];
    let mut best_score = 0.0;
    for &a in actions {
        if let Action::MonopolyPick(r) = a {
            let r = r as usize;
            let mut total = 0u8;
            for p in 0..PLAYER_COUNT {
                if p != cp {
                    total += state.players[p].resources[r];
                }
            }
            let score = total as f64 * RESOURCE_VALUE[r];
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
    }
    best
}
