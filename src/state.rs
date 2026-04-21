use crate::board::*;
use rand::Rng;

// ── Development Cards ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DevCardType {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

pub const DEV_CARD_TYPES: usize = 5;

/// Standard dev card deck: 14 Knight, 5 VP, 2 Road Building, 2 Year of Plenty, 2 Monopoly
pub fn standard_dev_deck() -> Vec<DevCardType> {
    let mut deck = Vec::with_capacity(25);
    for _ in 0..14 {
        deck.push(DevCardType::Knight);
    }
    for _ in 0..5 {
        deck.push(DevCardType::VictoryPoint);
    }
    for _ in 0..2 {
        deck.push(DevCardType::RoadBuilding);
    }
    for _ in 0..2 {
        deck.push(DevCardType::YearOfPlenty);
    }
    for _ in 0..2 {
        deck.push(DevCardType::Monopoly);
    }
    deck
}

// ── Game Phase ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    /// Setup phase: players place settlements and roads in order.
    /// round=0 is forward (P0,P1,P2,P3), round=1 is reverse (P3,P2,P1,P0).
    SetupSettlement { round: u8 },
    SetupRoad { round: u8 },
    /// Normal turn phases
    RollDice,
    /// Robber triggered (rolled 7 or played knight): must move robber and steal.
    MoveRobber,
    StealResource,
    /// Discard phase: players with >7 cards must discard half.
    Discard { player: u8 },
    /// Main action phase: build, trade, buy dev card, play dev card, or end turn.
    MainAction,
    /// Road building dev card: place 1-2 free roads.
    RoadBuildingPlace { remaining: u8 },
    /// Year of plenty: pick 2 resources from bank.
    YearOfPlentyPick { remaining: u8 },
    /// Monopoly: pick a resource to steal from all players.
    MonopolyPick,
    /// Game over.
    Finished,
}

// ── Player State ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub resources: [u8; RESOURCE_COUNT],
    pub dev_cards: [u8; DEV_CARD_TYPES],
    /// Dev cards bought this turn (can't play same turn).
    pub new_dev_cards: [u8; DEV_CARD_TYPES],
    pub knights_played: u8,
    pub victory_points_hidden: u8, // from VP dev cards
    pub roads_remaining: u8,       // starts at 15
    pub settlements_remaining: u8, // starts at 5
    pub cities_remaining: u8,      // starts at 4
    pub has_played_dev_this_turn: bool,
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            resources: [0; RESOURCE_COUNT],
            dev_cards: [0; DEV_CARD_TYPES],
            new_dev_cards: [0; DEV_CARD_TYPES],
            knights_played: 0,
            victory_points_hidden: 0,
            roads_remaining: 15,
            settlements_remaining: 5,
            cities_remaining: 4,
            has_played_dev_this_turn: false,
        }
    }

    pub fn total_resources(&self) -> u8 {
        self.resources.iter().sum()
    }

    pub fn can_afford(&self, cost: &[u8; RESOURCE_COUNT]) -> bool {
        (0..RESOURCE_COUNT).all(|i| self.resources[i] >= cost[i])
    }

    pub fn spend(&mut self, cost: &[u8; RESOURCE_COUNT]) {
        for i in 0..RESOURCE_COUNT {
            self.resources[i] -= cost[i];
        }
    }

    pub fn gain(&mut self, resources: &[u8; RESOURCE_COUNT]) {
        for i in 0..RESOURCE_COUNT {
            self.resources[i] += resources[i];
        }
    }
}

// ── Build Costs ────────────────────────────────────────────────────────────

/// [Brick, Wood, Wheat, Ore, Sheep]
pub const ROAD_COST: [u8; RESOURCE_COUNT] = [1, 1, 0, 0, 0];
pub const SETTLEMENT_COST: [u8; RESOURCE_COUNT] = [1, 1, 1, 0, 1];
pub const CITY_COST: [u8; RESOURCE_COUNT] = [0, 0, 2, 3, 0];
pub const DEV_CARD_COST: [u8; RESOURCE_COUNT] = [0, 0, 1, 1, 1];

// ── Mid-Game Seed ──────────────────────────────────────────────────────────

/// Full snapshot used to seed `GameState::from_mid_game_full`. Anything you
/// don't know (e.g. opponent hands) can be left as `Default`.
#[derive(Debug, Clone)]
pub struct MidGamePosition {
    pub robber_hex: u8,
    pub settlements: [u64; PLAYER_COUNT],
    pub cities: [u64; PLAYER_COUNT],
    pub roads: [u128; PLAYER_COUNT],
    pub resources: [[u8; RESOURCE_COUNT]; PLAYER_COUNT],
    /// Unplayed dev cards per player, indexed by `DevCardType`.
    pub unplayed_dev: [[u8; DEV_CARD_TYPES]; PLAYER_COUNT],
    pub knights_played: [u8; PLAYER_COUNT],
    pub vp_hidden: [u8; PLAYER_COUNT],
    pub current_player: u8,
}

impl Default for MidGamePosition {
    fn default() -> Self {
        Self {
            robber_hex: 0,
            settlements: [0u64; PLAYER_COUNT],
            cities: [0u64; PLAYER_COUNT],
            roads: [0u128; PLAYER_COUNT],
            resources: [[0u8; RESOURCE_COUNT]; PLAYER_COUNT],
            unplayed_dev: [[0u8; DEV_CARD_TYPES]; PLAYER_COUNT],
            knights_played: [0u8; PLAYER_COUNT],
            vp_hidden: [0u8; PLAYER_COUNT],
            current_player: 0,
        }
    }
}

// ── Game State ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GameState {
    pub players: [PlayerState; PLAYER_COUNT],
    /// Bitboard: bit i set = player has a settlement at vertex i.
    pub settlements: [u64; PLAYER_COUNT],
    /// Bitboard: bit i set = player has a city at vertex i.
    pub cities: [u64; PLAYER_COUNT],
    /// Bitboard: bit i set = player has a road on edge i.
    /// Uses u128 because EDGE_COUNT (72) exceeds u64 capacity.
    pub roads: [u128; PLAYER_COUNT],
    /// Which hex the robber is on.
    pub robber_hex: u8,
    /// Remaining dev card deck (shuffled).
    pub dev_deck: Vec<DevCardType>,
    /// Current player index (0..3).
    pub current_player: u8,
    /// Turn number (increments when current_player wraps).
    pub turn_number: u16,
    /// Current game phase.
    pub phase: GamePhase,
    /// Largest army holder (-1 = none).
    pub largest_army_player: i8,
    /// Longest road holder (-1 = none).
    pub longest_road_player: i8,
    /// Cached longest road lengths per player.
    pub longest_road_len: [u8; PLAYER_COUNT],
}

impl GameState {
    /// Create a new game state ready for setup phase.
    pub fn new(board: &BoardLayout) -> Self {
        let dev_deck = standard_dev_deck();
        // We'll shuffle during simulation with the provided RNG
        Self {
            players: [
                PlayerState::new(),
                PlayerState::new(),
                PlayerState::new(),
                PlayerState::new(),
            ],
            settlements: [0; PLAYER_COUNT],
            cities: [0; PLAYER_COUNT],
            roads: [0u128; PLAYER_COUNT],
            robber_hex: board.desert_hex() as u8,
            dev_deck,
            current_player: 0,
            turn_number: 0,
            phase: GamePhase::SetupSettlement { round: 0 },
            largest_army_player: -1,
            longest_road_player: -1,
            longest_road_len: [0; PLAYER_COUNT],
        }
    }

    /// Shuffle the dev deck with the provided RNG.
    pub fn shuffle_deck<R: Rng>(&mut self, rng: &mut R) {
        use rand::seq::SliceRandom;
        self.dev_deck.shuffle(rng);
    }

    // ── Bitboard Helpers ───────────────────────────────────────────────────

    /// All vertices occupied by any player (settlement or city).
    pub fn all_occupied_vertices(&self) -> u64 {
        let mut mask = 0u64;
        for p in 0..PLAYER_COUNT {
            mask |= self.settlements[p] | self.cities[p];
        }
        mask
    }

    /// All edges occupied by any player (road).
    pub fn all_occupied_edges(&self) -> u128 {
        let mut mask = 0u128;
        for p in 0..PLAYER_COUNT {
            mask |= self.roads[p];
        }
        mask
    }

    /// Check if vertex satisfies distance rule (no adjacent occupied vertex).
    pub fn distance_rule_ok(&self, vertex: usize) -> bool {
        let occupied = self.all_occupied_vertices();
        let neighbors = vertex_neighbors(vertex);
        for &n in neighbors {
            if n >= 0 && (occupied >> n as u64) & 1 == 1 {
                return false;
            }
        }
        true
    }

    /// Check if a vertex is occupied by any player.
    pub fn vertex_occupied(&self, vertex: usize) -> bool {
        (self.all_occupied_vertices() >> vertex) & 1 == 1
    }

    /// Check if player has a road on an edge.
    pub fn player_has_road(&self, player: usize, edge: usize) -> bool {
        (self.roads[player] >> edge) & 1 == 1
    }

    /// Check if player has a settlement at a vertex.
    pub fn player_has_settlement(&self, player: usize, vertex: usize) -> bool {
        (self.settlements[player] >> vertex) & 1 == 1
    }

    /// Check if player has a building (settlement or city) at a vertex.
    pub fn player_has_building(&self, player: usize, vertex: usize) -> bool {
        ((self.settlements[player] | self.cities[player]) >> vertex) & 1 == 1
    }

    /// Check if player has a road or building adjacent to a vertex.
    pub fn player_connected_to_vertex(&self, player: usize, vertex: usize) -> bool {
        // Check adjacent edges for roads
        let edges = vertex_edge_ids(vertex);
        for &e in edges {
            if e >= 0 && self.player_has_road(player, e as usize) {
                return true;
            }
        }
        // Check if player has building at vertex itself
        self.player_has_building(player, vertex)
    }

    /// Check if player has a road or building adjacent to an edge endpoint.
    pub fn player_connected_to_edge(&self, player: usize, edge: usize) -> bool {
        let (v1, v2) = edge_endpoint_ids(edge);
        self.player_connected_to_vertex(player, v1 as usize)
            || self.player_connected_to_vertex(player, v2 as usize)
    }

    // ── Victory Points ─────────────────────────────────────────────────────

    /// Calculate total VP for a player.
    pub fn victory_points(&self, player: usize) -> u8 {
        let p = &self.players[player];
        let settlements = self.settlements[player].count_ones() as u8;
        let cities = self.cities[player].count_ones() as u8;
        let mut vp = settlements + cities * 2 + p.victory_points_hidden;
        if self.largest_army_player == player as i8 {
            vp += 2;
        }
        if self.longest_road_player == player as i8 {
            vp += 2;
        }
        vp
    }

    /// Check if any player has won (>= 10 VP).
    pub fn winner(&self) -> Option<u8> {
        for p in 0..PLAYER_COUNT {
            if self.victory_points(p) >= 10 {
                return Some(p as u8);
            }
        }
        None
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, GamePhase::Finished) || self.winner().is_some()
    }

    // ── Resource Distribution ──────────────────────────────────────────────

    /// Distribute resources for a dice roll.
    pub fn distribute_resources(&mut self, board: &BoardLayout, roll: u8) {
        if roll == 7 {
            return; // Robber, no distribution
        }
        for hex in 0..HEX_COUNT {
            if board.tile_numbers[hex] != roll || hex == self.robber_hex as usize {
                continue;
            }
            if let Some(resource) = board.hex_resource(hex) {
                let vertices = hex_vertex_ids(hex);
                let res_idx = resource as usize;
                for &v in vertices.iter() {
                    for p in 0..PLAYER_COUNT {
                        if self.player_has_settlement(p, v as usize) {
                            self.players[p].resources[res_idx] += 1;
                        } else if (self.cities[p] >> v) & 1 == 1 {
                            self.players[p].resources[res_idx] += 2;
                        }
                    }
                }
            }
        }
    }

    // ── Longest Road Calculation ───────────────────────────────────────────

    /// Compute longest road for a player using DFS on the road graph.
    pub fn compute_longest_road(&self, player: usize) -> u8 {
        let road_mask = self.roads[player];
        if road_mask == 0 {
            return 0;
        }

        let mut best = 0u8;
        // Try starting DFS from each edge the player owns
        for start_edge in 0..EDGE_COUNT {
            if (road_mask >> start_edge) & 1 == 0 {
                continue;
            }
            let (v1, v2) = edge_endpoint_ids(start_edge);
            // DFS from v1 along this edge
            let mut visited = 0u128; // edge visited bitset
            visited |= 1u128 << start_edge;
            let len1 = self.road_dfs(player, v2, &mut visited) + 1;
            if len1 > best {
                best = len1;
            }
            // Also try from v2
            visited = 1u128 << start_edge;
            let len2 = self.road_dfs(player, v1, &mut visited) + 1;
            if len2 > best {
                best = len2;
            }
        }
        best
    }

    fn road_dfs(&self, player: usize, vertex: u8, visited: &mut u128) -> u8 {
        // If another player has a building here, road is broken
        for other in 0..PLAYER_COUNT {
            if other == player {
                continue;
            }
            if self.player_has_building(other, vertex as usize) {
                return 0;
            }
        }

        let edges = vertex_edge_ids(vertex as usize);
        let mut best = 0u8;
        for &e in edges {
            if e < 0 {
                continue;
            }
            let eidx = e as usize;
            if (self.roads[player] >> eidx) & 1 == 0 {
                continue;
            }
            if (*visited >> eidx) & 1 == 1 {
                continue;
            }
            *visited |= 1u128 << eidx;
            let (ev1, ev2) = edge_endpoint_ids(eidx);
            let next_v = if ev1 == vertex { ev2 } else { ev1 };
            let len = self.road_dfs(player, next_v, visited) + 1;
            if len > best {
                best = len;
            }
            *visited &= !(1u128 << eidx);
        }
        best
    }

    /// Update largest army after a knight is played.
    pub fn update_largest_army(&mut self) {
        let cp = self.current_player as usize;
        let knights = self.players[cp].knights_played;
        if knights >= 3 {
            if self.largest_army_player < 0 {
                self.largest_army_player = cp as i8;
            } else {
                let current_holder = self.largest_army_player as usize;
                if knights > self.players[current_holder].knights_played {
                    self.largest_army_player = cp as i8;
                }
            }
        }
    }

    /// Update longest road after a road is built.
    pub fn update_longest_road(&mut self) {
        for p in 0..PLAYER_COUNT {
            self.longest_road_len[p] = self.compute_longest_road(p);
        }
        // Find player with longest road >= 5
        let mut best_player: i8 = -1;
        let mut best_len: u8 = 4; // minimum 5 to qualify
        for p in 0..PLAYER_COUNT {
            if self.longest_road_len[p] > best_len {
                best_len = self.longest_road_len[p];
                best_player = p as i8;
            } else if self.longest_road_len[p] == best_len && best_player >= 0 {
                // Tie: keep current holder (or no one if no current holder)
                if self.longest_road_player == p as i8 {
                    best_player = p as i8;
                }
            }
        }
        self.longest_road_player = best_player;
    }

    // ── Turn Management ────────────────────────────────────────────────────

    /// Seat that should choose the next action. Matches `current_player`
    /// except during `GamePhase::Discard { player }`, where the discarder
    /// (who may not be the roller) is the acting seat. Used by
    /// `PerSeatPolicy` to dispatch actions to the right policy.
    pub fn acting_player(&self) -> u8 {
        match self.phase {
            GamePhase::Discard { player } => player,
            _ => self.current_player,
        }
    }

    /// Advance to next player.
    pub fn next_player(&mut self) {
        // Move new dev cards to playable
        let cp = self.current_player as usize;
        for i in 0..DEV_CARD_TYPES {
            self.players[cp].dev_cards[i] += self.players[cp].new_dev_cards[i];
            self.players[cp].new_dev_cards[i] = 0;
        }
        self.players[cp].has_played_dev_this_turn = false;

        self.current_player = (self.current_player + 1) % PLAYER_COUNT as u8;
        if self.current_player == 0 {
            self.turn_number += 1;
        }
        self.phase = GamePhase::RollDice;
    }

    /// Build a mid-game state from known piece placements.
    ///
    /// `settlements[p]` and `cities[p]` are bitmasks over vertex indices 0..53.
    /// Phase is set to `RollDice`; turn_number is estimated from total pieces placed.
    /// This lets the engine simulate from a real-game snapshot rather than fresh setup.
    pub fn from_mid_game(
        robber_hex: u8,
        settlements: [u64; PLAYER_COUNT],
        cities: [u64; PLAYER_COUNT],
        current_player: u8,
    ) -> Self {
        Self::from_mid_game_full(&MidGamePosition {
            robber_hex,
            settlements,
            cities,
            current_player,
            ..Default::default()
        })
    }

    /// Build a mid-game state from a full snapshot (pieces, roads, resources,
    /// dev cards, knight counts, robber position). Any fields left at their
    /// `Default` are treated as empty/unknown. Cached longest-road and
    /// largest-army holders are recomputed from the seeded state.
    pub fn from_mid_game_full(init: &MidGamePosition) -> Self {
        // Account for cards already accounted for elsewhere (in hands, played as
        // knights, or face-down VP cards). Remove those from the 25-card deck
        // so duplicates can't be re-drawn.
        let mut accounted = [0u8; DEV_CARD_TYPES];
        for p in 0..PLAYER_COUNT {
            for i in 0..DEV_CARD_TYPES {
                accounted[i] += init.unplayed_dev[p][i];
            }
            accounted[DevCardType::Knight as usize] += init.knights_played[p];
            accounted[DevCardType::VictoryPoint as usize] += init.vp_hidden[p];
        }
        let dev_deck: Vec<DevCardType> = standard_dev_deck()
            .into_iter()
            .filter(|card| {
                let i = *card as usize;
                if accounted[i] > 0 {
                    accounted[i] -= 1;
                    false
                } else {
                    true
                }
            })
            .collect();

        let mut players: [PlayerState; PLAYER_COUNT] = [
            PlayerState::new(),
            PlayerState::new(),
            PlayerState::new(),
            PlayerState::new(),
        ];

        // A city replaces the settlement on the same vertex; enforce invariant.
        let mut settlements = init.settlements;
        for p in 0..PLAYER_COUNT {
            settlements[p] &= !init.cities[p];
        }

        for p in 0..PLAYER_COUNT {
            players[p].resources = init.resources[p];
            players[p].dev_cards = init.unplayed_dev[p];
            players[p].knights_played = init.knights_played[p];
            players[p].victory_points_hidden = init.vp_hidden[p];

            let n_s = settlements[p].count_ones() as u8;
            let n_c = init.cities[p].count_ones() as u8;
            let n_r = init.roads[p].count_ones() as u8;
            // A city was built on a settlement slot, so settlements_remaining covers both.
            players[p].settlements_remaining = 5u8.saturating_sub(n_s + n_c);
            players[p].cities_remaining = 4u8.saturating_sub(n_c);
            players[p].roads_remaining = 15u8.saturating_sub(n_r);
        }

        let mut state = Self {
            players,
            settlements,
            cities: init.cities,
            roads: init.roads,
            robber_hex: init.robber_hex,
            dev_deck,
            current_player: init.current_player % PLAYER_COUNT as u8,
            turn_number: 0,
            phase: GamePhase::RollDice,
            largest_army_player: -1,
            longest_road_player: -1,
            longest_road_len: [0; PLAYER_COUNT],
        };

        // Seed largest-army from knights_played (>=3 required to qualify).
        let mut la_player: i8 = -1;
        let mut la_count: u8 = 2;
        for p in 0..PLAYER_COUNT {
            let k = init.knights_played[p];
            if k >= 3 && k > la_count {
                la_count = k;
                la_player = p as i8;
            }
        }
        state.largest_army_player = la_player;

        // Recompute longest-road cache and holder from seeded road bitboards.
        state.update_longest_road();

        // Rough progress estimate for draw detection.
        let total_pieces: u32 = (0..PLAYER_COUNT)
            .map(|p| state.settlements[p].count_ones() + state.cities[p].count_ones())
            .sum();
        state.turn_number = (total_pieces / PLAYER_COUNT as u32) as u16;

        state
    }

    /// Get setup turn order player for a given step (0..7).
    pub fn setup_player(step: usize) -> u8 {
        // Forward: 0,1,2,3 then reverse: 3,2,1,0
        if step < PLAYER_COUNT {
            step as u8
        } else {
            (2 * PLAYER_COUNT - 1 - step) as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game_state() {
        let board = BoardLayout::beginner();
        let state = GameState::new(&board);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.turn_number, 0);
        assert!(matches!(
            state.phase,
            GamePhase::SetupSettlement { round: 0 }
        ));
        assert_eq!(state.dev_deck.len(), 25);
    }

    #[test]
    fn test_build_costs() {
        let mut player = PlayerState::new();
        assert!(!player.can_afford(&SETTLEMENT_COST));
        player.resources = [1, 1, 1, 0, 1]; // brick, wood, wheat, ore, sheep
        assert!(player.can_afford(&SETTLEMENT_COST));
        player.spend(&SETTLEMENT_COST);
        assert_eq!(player.resources, [0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_victory_points() {
        let board = BoardLayout::beginner();
        let mut state = GameState::new(&board);
        // Place a settlement for player 0 at vertex 0
        state.settlements[0] |= 1u64 << 0;
        assert_eq!(state.victory_points(0), 1);
        // Add a city at vertex 5
        state.cities[0] |= 1u64 << 5;
        assert_eq!(state.victory_points(0), 3); // 1 settlement + 1 city*2
    }

    #[test]
    fn test_setup_player_order() {
        assert_eq!(GameState::setup_player(0), 0);
        assert_eq!(GameState::setup_player(1), 1);
        assert_eq!(GameState::setup_player(2), 2);
        assert_eq!(GameState::setup_player(3), 3);
        assert_eq!(GameState::setup_player(4), 3);
        assert_eq!(GameState::setup_player(5), 2);
        assert_eq!(GameState::setup_player(6), 1);
        assert_eq!(GameState::setup_player(7), 0);
    }
}
