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
///
/// `coalition_pressure` (default 1.0) scales how aggressively this policy
/// targets the VP leader with the robber, knight, and steal actions. 0.0 is
/// pure selfish play (no VP weighting); higher values approximate multiplayer
/// Nash behaviour where opponents focus-fire whoever's closest to winning.
/// The "real" GTO win probability for a position lies in the band you get
/// by sweeping this parameter — it isn't a single number for >2-player games.
#[derive(Debug, Clone, Copy)]
pub struct RuleBasedPolicy {
    pub coalition_pressure: f64,
}

impl RuleBasedPolicy {
    pub const DEFAULT_COALITION_PRESSURE: f64 = 1.0;

    pub fn new(coalition_pressure: f64) -> Self {
        Self { coalition_pressure }
    }
}

impl Default for RuleBasedPolicy {
    fn default() -> Self {
        Self { coalition_pressure: Self::DEFAULT_COALITION_PRESSURE }
    }
}

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
            GamePhase::MoveRobber => select_robber_hex(state, board, actions, self.coalition_pressure),
            GamePhase::StealResource => select_steal_target(state, actions, self.coalition_pressure),
            GamePhase::Discard { .. } => actions[0], // greedy discard already computed
            GamePhase::MainAction => select_main_action(state, board, actions, rng, self.coalition_pressure),
            GamePhase::RoadBuildingPlace { .. } => select_road_placement(state, board, actions, rng),
            GamePhase::YearOfPlentyPick { .. } => select_year_of_plenty(state, actions),
            GamePhase::MonopolyPick => select_monopoly(state, actions),
            _ => actions[rng.gen_range(0..actions.len())],
        }
    }
}

// ── Resource Valuation ─────────────────────────────────────────────────────

/// Baseline strategic resource values (tunable). These get scaled per-board
/// by scarcity in `BoardContext` so wheat on a wheat-rich board is worth less
/// than wheat on a wheat-starved one.
const BASE_RESOURCE_VALUE: [f64; RESOURCE_COUNT] = [
    1.0, // Brick
    1.0, // Wood
    1.1, // Wheat
    1.2, // Ore
    0.8, // Sheep
];

/// Per-board scoring context: resource weights scaled by this board's
/// scarcity profile. Cheap to compute (~19 hex iterations).
#[derive(Debug, Clone, Copy)]
pub struct BoardContext {
    /// Final resource weight = BASE_RESOURCE_VALUE[r] × scarcity_factor[r].
    pub weight: [f64; RESOURCE_COUNT],
}

impl BoardContext {
    pub fn from_board(board: &BoardLayout) -> Self {
        // Total production weight per resource across the board.
        let mut total = [0.0f64; RESOURCE_COUNT];
        for hex in 0..HEX_COUNT {
            if let Some(res) = board.hex_resource(hex) {
                total[res as usize] += production_weight(board.tile_numbers[hex]);
            }
        }
        // Mean non-zero production across all resources present.
        let (sum, count): (f64, usize) = total.iter()
            .fold((0.0, 0usize), |(s, c), &t| if t > 0.0 { (s + t, c + 1) } else { (s, c) });
        let mean = if count > 0 { sum / count as f64 } else { 1.0 };

        // scarcity_factor = mean / resource_total, clamped to [0.5, 2.0]
        // so a resource producing half the average gets +up-to-2x weight,
        // and an abundant one gets down-weighted, but never extremely.
        let mut weight = [0.0; RESOURCE_COUNT];
        for r in 0..RESOURCE_COUNT {
            let factor = if total[r] > 1e-9 {
                (mean / total[r]).clamp(0.5, 2.0)
            } else {
                2.0 // resource entirely missing from board → max scarcity
            };
            weight[r] = BASE_RESOURCE_VALUE[r] * factor;
        }
        BoardContext { weight }
    }
}

/// Raw production value of a vertex: sum of resource * production across its
/// adjacent hexes, scaled by board scarcity. Used as a building block by
/// `vertex_score` and by the denial bonus.
fn vertex_production_value(board: &BoardLayout, ctx: &BoardContext, vertex: usize) -> (f64, u8) {
    let hexes = vertex_hex_ids(vertex);
    let mut value = 0.0;
    let mut resource_types = 0u8;
    for &h in hexes {
        if h < 0 {
            continue;
        }
        let hex = h as usize;
        if let Some(res) = board.hex_resource(hex) {
            let prod = production_weight(board.tile_numbers[hex]);
            value += ctx.weight[res as usize] * prod * 36.0;
            resource_types |= 1 << (res as u8);
        }
    }
    (value, resource_types)
}

/// Bitmask of resources produced at a player's currently-placed settlements
/// (plus a hypothetical settlement at `also_vertex` if Some).
fn player_produced_resources(
    board: &BoardLayout,
    state: &GameState,
    player: usize,
    also_vertex: Option<usize>,
) -> u8 {
    let mut mask = 0u8;
    let mut add_vertex = |v: usize| {
        for &h in vertex_hex_ids(v) {
            if h < 0 { continue; }
            if let Some(res) = board.hex_resource(h as usize) {
                mask |= 1 << (res as u8);
            }
        }
    };
    for v in 0..VERTEX_COUNT {
        if state.player_has_building(player, v) {
            add_vertex(v);
        }
    }
    if let Some(v) = also_vertex {
        add_vertex(v);
    }
    mask
}

/// Score a vertex for settlement placement.
///
/// `produced_mask` is the bitmask of resources the player already produces
/// (used so a 2:1 port is only worth full value when we produce that resource).
/// Pass 0 if you don't care (legacy callers).
fn vertex_score(
    board: &BoardLayout,
    ctx: &BoardContext,
    vertex: usize,
    produced_mask: u8,
) -> f64 {
    let (mut score, resource_types) = vertex_production_value(board, ctx, vertex);

    // Diversity bonus: +1.5 per unique resource type on adjacent hexes.
    // Fold the hypothetical settlement at this vertex into the player's set.
    let combined = resource_types | produced_mask;
    let diversity = combined.count_ones() as f64;
    score += diversity * 1.5;

    // Port bonus — conditional on producing the relevant resource.
    if let Some(port) = board.port_at_vertex(vertex as u8) {
        match port {
            PortType::ThreeToOne => score += 0.8,
            PortType::TwoToOne(r) => {
                // 2:1 port is only valuable if we produce that resource
                // (including via this settlement). Otherwise it's worth
                // roughly as much as a 3:1 (generic trade route option).
                let bit = 1u8 << (r as u8);
                if combined & bit != 0 {
                    score += 2.0;
                } else {
                    score += 0.5;
                }
            }
        }
    }

    score
}

/// Denial bonus: by taking `vertex`, we block (via distance rule) its
/// neighboring vertices from being placed on. Value = sum of production
/// value of those blocked neighbor vertices that are currently legal.
/// Caller weights this against own-score.
fn denial_bonus(board: &BoardLayout, ctx: &BoardContext, state: &GameState, vertex: usize) -> f64 {
    let mut bonus = 0.0;
    for &n in vertex_neighbors(vertex) {
        if n < 0 {
            continue;
        }
        let nv = n as usize;
        // Only count neighbors currently legal for placement (unoccupied and distance-rule OK).
        // Since `vertex` is not yet placed, `distance_rule_ok(nv)` reflects the board *before*
        // we place; that's fine — it tells us what an opponent could have taken.
        if state.vertex_occupied(nv) {
            continue;
        }
        if !state.distance_rule_ok(nv) {
            continue;
        }
        let (v, _) = vertex_production_value(board, ctx, nv);
        bonus += v;
    }
    bonus
}

// ── Setup Phase ────────────────────────────────────────────────────────────

/// Denial weight during setup. Only applied in round 0 where many snake-draft
/// placements are still to come; in round 1, denial matters less (the player
/// has no further placements to benefit from blocking) so we taper it.
const DENIAL_WEIGHT_ROUND_0: f64 = 0.30;
const DENIAL_WEIGHT_ROUND_1: f64 = 0.05;

fn select_setup_settlement(state: &GameState, board: &BoardLayout, actions: &[Action]) -> Action {
    let cp = state.current_player as usize;
    let ctx = BoardContext::from_board(board);
    let produced = player_produced_resources(board, state, cp, None);

    // Determine snake-draft round for denial weighting.
    let denial_w = match state.phase {
        GamePhase::SetupSettlement { round: 0 } => DENIAL_WEIGHT_ROUND_0,
        _ => DENIAL_WEIGHT_ROUND_1,
    };

    let mut best_action = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &action in actions {
        if let Action::PlaceSettlement(v) = action {
            let v = v as usize;
            let own = vertex_score(board, &ctx, v, produced);
            let denied = denial_bonus(board, &ctx, state, v);
            let score = own + denial_w * denied;
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
    // Place road toward the best neighboring unoccupied vertex.
    let cp = state.current_player as usize;
    let ctx = BoardContext::from_board(board);
    let produced = player_produced_resources(board, state, cp, None);
    let mut best_action = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &action in actions {
        if let Action::PlaceRoad(e) = action {
            let (v1, v2) = edge_endpoint_ids(e as usize);
            let s1 = if !state.vertex_occupied(v1 as usize) {
                vertex_score(board, &ctx, v1 as usize, produced)
            } else {
                0.0
            };
            let s2 = if !state.vertex_occupied(v2 as usize) {
                vertex_score(board, &ctx, v2 as usize, produced)
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

fn select_robber_hex(
    state: &GameState,
    board: &BoardLayout,
    actions: &[Action],
    coalition_pressure: f64,
) -> Action {
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
                        score += prod * (1.0 + their_vp * 0.3 * coalition_pressure);
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

fn select_steal_target(state: &GameState, actions: &[Action], coalition_pressure: f64) -> Action {
    let mut best_action = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &action in actions {
        if let Action::StealFrom(target) = action {
            // Prefer stealing from players with more resources and (under coalition
            // pressure) from higher-VP players. pressure=0 → pure resource target.
            let t = target as usize;
            let score = state.players[t].total_resources() as f64
                + state.victory_points(t) as f64 * 2.0 * coalition_pressure;
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
    coalition_pressure: f64,
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
            return select_robber_hex_from_knights(state, board, &knights, cp, coalition_pressure);
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
    let ctx = BoardContext::from_board(board);
    let produced = player_produced_resources(board, state, player, None);
    let mut best = actions[0];
    let mut best_score = f64::NEG_INFINITY;
    for &a in actions {
        let v = match a {
            Action::PlaceSettlement(v) | Action::BuildCity(v) => v,
            _ => continue,
        };
        let score = vertex_score(board, &ctx, v as usize, produced);
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
    coalition_pressure: f64,
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
                        score += prod * (1.0 + state.victory_points(p) as f64 * 0.3 * coalition_pressure);
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
    let ctx = BoardContext::from_board(board);
    let produced = player_produced_resources(board, state, cp, None);
    let mut best = actions[0];
    let mut best_score = f64::NEG_INFINITY;

    for &a in actions {
        if let Action::PlaceRoad(e) = a {
            let (v1, v2) = edge_endpoint_ids(e as usize);
            let mut score = 0.0;
            for v in [v1, v2] {
                if !state.vertex_occupied(v as usize) && state.distance_rule_ok(v as usize) {
                    score += vertex_score(board, &ctx, v as usize, produced) * 0.5;
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
            let need = BASE_RESOURCE_VALUE[r as usize] / (p.resources[r as usize] as f64 + 1.0);
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
            let score = total as f64 * BASE_RESOURCE_VALUE[r];
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
    }
    best
}
