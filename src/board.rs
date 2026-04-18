use serde::{Deserialize, Serialize};

// ── Resource & Port Types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Resource {
    Brick = 0,
    Wood = 1,
    Wheat = 2,
    Ore = 3,
    Sheep = 4,
}

pub const RESOURCE_COUNT: usize = 5;
pub const ALL_RESOURCES: [Resource; 5] = [
    Resource::Brick,
    Resource::Wood,
    Resource::Wheat,
    Resource::Ore,
    Resource::Sheep,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TileType {
    Brick = 0,
    Wood = 1,
    Wheat = 2,
    Ore = 3,
    Sheep = 4,
    Desert = 5,
}

impl TileType {
    pub fn resource(self) -> Option<Resource> {
        match self {
            TileType::Brick => Some(Resource::Brick),
            TileType::Wood => Some(Resource::Wood),
            TileType::Wheat => Some(Resource::Wheat),
            TileType::Ore => Some(Resource::Ore),
            TileType::Sheep => Some(Resource::Sheep),
            TileType::Desert => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortType {
    ThreeToOne,
    TwoToOne(Resource),
}

// ── Hex Grid Topology ──────────────────────────────────────────────────────
//
// Standard Catan board: 19 hexes in a radius-2 hexagonal arrangement.
// We use axial coordinates (q, r) with the constraint |q| <= 2, |r| <= 2, |q+r| <= 2.
//
// The 19 hex positions in a fixed order:
pub const HEX_COUNT: usize = 19;
pub const VERTEX_COUNT: usize = 54;
pub const EDGE_COUNT: usize = 72;
pub const PORT_COUNT: usize = 9;
pub const PLAYER_COUNT: usize = 4;

/// Axial hex coordinates. The 19 valid hexes satisfy |q| <= 2, |r| <= 2, |q+r| <= 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub q: i8,
    pub r: i8,
}

impl HexCoord {
    pub fn new(q: i8, r: i8) -> Self {
        Self { q, r }
    }

    pub fn is_valid(self) -> bool {
        self.q.abs() <= 2 && self.r.abs() <= 2 && (self.q + self.r).abs() <= 2
    }
}

/// The 19 hexes in canonical order (spiral from center).
pub const HEX_COORDS: [HexCoord; HEX_COUNT] = [
    // Center
    HexCoord { q: 0, r: 0 },
    // Ring 1 (6 hexes)
    HexCoord { q: 1, r: 0 },
    HexCoord { q: 1, r: -1 },
    HexCoord { q: 0, r: -1 },
    HexCoord { q: -1, r: 0 },
    HexCoord { q: -1, r: 1 },
    HexCoord { q: 0, r: 1 },
    // Ring 2 (12 hexes)
    HexCoord { q: 2, r: 0 },
    HexCoord { q: 2, r: -1 },
    HexCoord { q: 2, r: -2 },
    HexCoord { q: 1, r: -2 },
    HexCoord { q: 0, r: -2 },
    HexCoord { q: -1, r: -1 },
    HexCoord { q: -2, r: 0 },
    HexCoord { q: -2, r: 1 },
    HexCoord { q: -2, r: 2 },
    HexCoord { q: -1, r: 2 },
    HexCoord { q: 0, r: 2 },
    HexCoord { q: 1, r: 1 },
];

/// Map from HexCoord to hex index (0..19). Returns None for invalid coords.
pub fn hex_index(coord: HexCoord) -> Option<usize> {
    HEX_COORDS.iter().position(|&c| c == coord)
}

// ── Vertex & Edge Addressing ───────────────────────────────────────────────
//
// Each vertex is the intersection of up to 3 hexes.
// Each edge is shared between up to 2 hexes.
//
// We enumerate vertices 0..53 and edges 0..71 using precomputed tables.
// For each hex, there are 6 vertices (numbered 0=N, 1=NE, 2=SE, 3=S, 4=SW, 5=NW)
// and 6 edges (numbered 0=NE, 1=E, 2=SE, 3=SW, 4=W, 5=NW).

/// For each hex (0..19), the 6 vertex indices (N, NE, SE, S, SW, NW).
/// Precomputed so that shared vertices between adjacent hexes map to the same index.
static HEX_VERTICES: [[u8; 6]; HEX_COUNT] = precompute_hex_vertices();

/// For each hex (0..19), the 6 edge indices.
static HEX_EDGES: [[u8; 6]; HEX_COUNT] = precompute_hex_edges();

/// For each vertex (0..54), the list of adjacent hex indices (up to 3).
static VERTEX_HEXES: [[i8; 3]; VERTEX_COUNT] = precompute_vertex_hexes();

/// For each vertex (0..54), the list of adjacent vertex indices (up to 3).
static VERTEX_ADJACENT: [[i8; 3]; VERTEX_COUNT] = precompute_vertex_adjacent();

/// For each edge (0..72), the two endpoint vertex indices.
static EDGE_VERTICES: [[u8; 2]; EDGE_COUNT] = precompute_edge_vertices();

/// For each vertex (0..54), the adjacent edges (up to 3).
static VERTEX_EDGES: [[i8; 3]; VERTEX_COUNT] = precompute_vertex_edges();

// ── Precomputation (const fn) ──────────────────────────────────────────────
//
// The Catan board topology is fixed. We compute all adjacency tables at compile time.
// Strategy: assign vertex/edge IDs by iterating hex vertices/edges and deduplicating
// based on coordinate equality.

/// Vertex position in "doubled" coordinates for deduplication.
/// Each hex center is at (2*q + r, 2*r) in a doubled-coordinate system.
/// Vertices offset from center by fixed deltas.
const fn vertex_pos(hex_idx: usize, local: usize) -> (i16, i16) {
    let q = HEX_COORDS[hex_idx].q as i16;
    let r = HEX_COORDS[hex_idx].r as i16;
    // Hex center in a fine grid: (3*q, 3*r) with offsets for vertices
    // Using a grid where hex centers are at (3q, 3r) and vertices at integer offsets
    let cx = 3 * q;
    let cy = 3 * r;
    match local {
        0 => (cx + 1, cy - 2), // N  (top-right of top)
        1 => (cx + 2, cy - 1), // NE
        2 => (cx + 1, cy + 1), // SE
        3 => (cx - 1, cy + 2), // S
        4 => (cx - 2, cy + 1), // SW
        5 => (cx - 1, cy - 1), // NW
        _ => (127, 127),       // unreachable
    }
}

/// Edge midpoint in fine grid for deduplication.
const fn edge_pos(hex_idx: usize, local: usize) -> (i16, i16) {
    let q = HEX_COORDS[hex_idx].q as i16;
    let r = HEX_COORDS[hex_idx].r as i16;
    let cx = 3 * q;
    let cy = 3 * r;
    // Edge midpoints: average of the two endpoint vertex positions
    // local 0=NE edge (between N and NE vertices), 1=E (NE-SE), 2=SE (SE-S),
    // 3=SW (S-SW), 4=W (SW-NW), 5=NW (NW-N)
    match local {
        0 => (cx * 2 + 3, cy * 2 - 3), // NE: avg of N(+1,-2) and NE(+2,-1)
        1 => (cx * 2 + 3, cy * 2),      // E:  avg of NE(+2,-1) and SE(+1,+1)
        2 => (cx * 2, cy * 2 + 3),      // SE: avg of SE(+1,+1) and S(-1,+2)
        3 => (cx * 2 - 3, cy * 2 + 3),  // SW: avg of S(-1,+2) and SW(-2,+1)
        4 => (cx * 2 - 3, cy * 2),      // W:  avg of SW(-2,+1) and NW(-1,-1)
        5 => (cx * 2, cy * 2 - 3),      // NW: avg of NW(-1,-1) and N(+1,-2)
        _ => (127, 127),
    }
}

const fn precompute_hex_vertices() -> [[u8; 6]; HEX_COUNT] {
    let mut result = [[0u8; 6]; HEX_COUNT];
    // Collect all unique vertex positions and assign IDs
    let mut positions = [(0i16, 0i16); 120]; // upper bound: 19*6=114
    let mut count = 0usize;

    let mut h = 0;
    while h < HEX_COUNT {
        let mut v = 0;
        while v < 6 {
            let pos = vertex_pos(h, v);
            // Check if this position already exists
            let mut found = 255u8;
            let mut i = 0;
            while i < count {
                if positions[i].0 == pos.0 && positions[i].1 == pos.1 {
                    found = i as u8;
                    break;
                }
                i += 1;
            }
            if found == 255 {
                found = count as u8;
                positions[count] = pos;
                count += 1;
            }
            result[h][v] = found;
            v += 1;
        }
        h += 1;
    }
    // count should equal VERTEX_COUNT (54)
    result
}

const fn precompute_hex_edges() -> [[u8; 6]; HEX_COUNT] {
    let mut result = [[0u8; 6]; HEX_COUNT];
    let mut positions = [(0i16, 0i16); 120];
    let mut count = 0usize;

    let mut h = 0;
    while h < HEX_COUNT {
        let mut e = 0;
        while e < 6 {
            let pos = edge_pos(h, e);
            let mut found = 255u8;
            let mut i = 0;
            while i < count {
                if positions[i].0 == pos.0 && positions[i].1 == pos.1 {
                    found = i as u8;
                    break;
                }
                i += 1;
            }
            if found == 255 {
                found = count as u8;
                positions[count] = pos;
                count += 1;
            }
            result[h][e] = found;
            e += 1;
        }
        h += 1;
    }
    result
}

const fn precompute_vertex_hexes() -> [[i8; 3]; VERTEX_COUNT] {
    let mut result = [[-1i8; 3]; VERTEX_COUNT];
    let mut h = 0;
    while h < HEX_COUNT {
        let mut v = 0;
        while v < 6 {
            let vid = HEX_VERTICES[h][v] as usize;
            // Find empty slot in result[vid]
            let mut s = 0;
            while s < 3 {
                if result[vid][s] == -1 {
                    result[vid][s] = h as i8;
                    break;
                }
                s += 1;
            }
            v += 1;
        }
        h += 1;
    }
    result
}

const fn precompute_vertex_adjacent() -> [[i8; 3]; VERTEX_COUNT] {
    // Two vertices are adjacent if they share an edge.
    // An edge connects vertex local i to vertex local (i+1)%6 within a hex.
    let mut result = [[-1i8; 3]; VERTEX_COUNT];

    let mut h = 0;
    while h < HEX_COUNT {
        let mut v = 0;
        while v < 6 {
            let v1 = HEX_VERTICES[h][v] as usize;
            let v2 = HEX_VERTICES[h][(v + 1) % 6] as usize;
            // Add v2 as adjacent to v1 (if not already present)
            add_adjacent(&mut result[v1], v2 as i8);
            add_adjacent(&mut result[v2], v1 as i8);
            v += 1;
        }
        h += 1;
    }
    result
}

const fn add_adjacent(arr: &mut [i8; 3], val: i8) {
    let mut i = 0;
    while i < 3 {
        if arr[i] == val {
            return;
        }
        if arr[i] == -1 {
            arr[i] = val;
            return;
        }
        i += 1;
    }
}

const fn precompute_edge_vertices() -> [[u8; 2]; EDGE_COUNT] {
    let mut result = [[255u8; 2]; EDGE_COUNT];
    let mut h = 0;
    while h < HEX_COUNT {
        let mut e = 0;
        while e < 6 {
            let eid = HEX_EDGES[h][e] as usize;
            if result[eid][0] == 255 {
                // Edge e connects vertex local e to vertex local (e+1)%6
                // Wait -- edge local 0 (NE) connects vertex 0 (N) and vertex 1 (NE)
                // Edge local i connects vertex i and vertex (i+1)%6
                result[eid][0] = HEX_VERTICES[h][e];
                result[eid][1] = HEX_VERTICES[h][(e + 1) % 6];
            }
            e += 1;
        }
        h += 1;
    }
    result
}

const fn precompute_vertex_edges() -> [[i8; 3]; VERTEX_COUNT] {
    let mut result = [[-1i8; 3]; VERTEX_COUNT];
    let mut e = 0;
    while e < EDGE_COUNT {
        if EDGE_VERTICES[e][0] != 255 {
            let v1 = EDGE_VERTICES[e][0] as usize;
            let v2 = EDGE_VERTICES[e][1] as usize;
            add_adjacent(&mut result[v1], e as i8);
            add_adjacent(&mut result[v2], e as i8);
        }
        e += 1;
    }
    result
}

// ── Public Accessors ───────────────────────────────────────────────────────

/// Get the 6 vertex indices for a hex.
pub fn hex_vertex_ids(hex: usize) -> &'static [u8; 6] {
    &HEX_VERTICES[hex]
}

/// Get the 6 edge indices for a hex.
pub fn hex_edge_ids(hex: usize) -> &'static [u8; 6] {
    &HEX_EDGES[hex]
}

/// Get adjacent hex indices for a vertex (up to 3, -1 = none).
pub fn vertex_hex_ids(vertex: usize) -> &'static [i8; 3] {
    &VERTEX_HEXES[vertex]
}

/// Get adjacent vertices (up to 3, -1 = none).
pub fn vertex_neighbors(vertex: usize) -> &'static [i8; 3] {
    &VERTEX_ADJACENT[vertex]
}

/// Get the two endpoint vertices of an edge.
pub fn edge_endpoint_ids(edge: usize) -> (u8, u8) {
    (EDGE_VERTICES[edge][0], EDGE_VERTICES[edge][1])
}

/// Get adjacent edges for a vertex (up to 3, -1 = none).
pub fn vertex_edge_ids(vertex: usize) -> &'static [i8; 3] {
    &VERTEX_EDGES[vertex]
}

// ── Dice Probability ───────────────────────────────────────────────────────

/// Number of ways to roll each sum (index 0-12, indices 0 and 1 unused).
pub const DICE_WAYS: [u8; 13] = [0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1];

/// Probability of each dice sum.
pub fn dice_probability(sum: u8) -> f64 {
    DICE_WAYS[sum as usize] as f64 / 36.0
}

/// Production weight for settlement placement scoring.
/// Higher = more likely to produce. Normalized so max (6 or 8) = 5/36.
pub fn production_weight(number: u8) -> f64 {
    if number == 0 || number == 7 {
        return 0.0;
    }
    DICE_WAYS[number as usize] as f64 / 36.0
}

// ── Board Layout ───────────────────────────────────────────────────────────

/// Returns true if any two adjacent hexes both carry a 6 or 8.
fn has_adjacent_red_numbers(numbers: &[u8; HEX_COUNT]) -> bool {
    const DIRS: [(i8, i8); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];
    for h in 0..HEX_COUNT {
        if numbers[h] == 6 || numbers[h] == 8 {
            let q = HEX_COORDS[h].q;
            let r = HEX_COORDS[h].r;
            for (dq, dr) in DIRS {
                let neighbor = HexCoord::new(q + dq, r + dr);
                if let Some(nh) = hex_index(neighbor) {
                    if numbers[nh] == 6 || numbers[nh] == 8 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// A specific board setup: which resource and number token is on each hex,
/// plus port configuration. This is fixed for an entire simulation batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardLayout {
    pub tile_types: [TileType; HEX_COUNT],
    pub tile_numbers: [u8; HEX_COUNT], // 0 for desert
    pub ports: [(PortType, u8, u8); PORT_COUNT], // (type, vertex1, vertex2)
}

impl BoardLayout {
    /// Create the standard beginner board layout.
    pub fn beginner() -> Self {
        // Standard beginner layout tiles (spiral order matching HEX_COORDS)
        let tile_types = [
            TileType::Desert, // center
            TileType::Wheat,
            TileType::Sheep,
            TileType::Wood,
            TileType::Brick,
            TileType::Ore,
            TileType::Wheat,
            TileType::Wood,
            TileType::Sheep,
            TileType::Brick,
            TileType::Ore,
            TileType::Wood,
            TileType::Wheat,
            TileType::Sheep,
            TileType::Brick,
            TileType::Ore,
            TileType::Wheat,
            TileType::Sheep,
            TileType::Wood,
        ];

        let tile_numbers = [
            0,  // desert
            2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 3, 4, 5, 6, 8, 9, 10, 11,
        ];

        // Port vertex IDs — outer boundary vertex pairs, going clockwise from right.
        // Vertices 24-53 are the 30 outer boundary vertices (ring-2 only).
        // Each pair forms one edge of the outer hexagonal perimeter.
        let ports = [
            (PortType::ThreeToOne,           24u8, 25u8), // right
            (PortType::TwoToOne(Resource::Brick),  27u8, 53u8), // lower-right
            (PortType::ThreeToOne,           48u8, 52u8), // bottom-right
            (PortType::TwoToOne(Resource::Wood),   45u8, 46u8), // bottom-left
            (PortType::ThreeToOne,           43u8, 44u8), // left
            (PortType::TwoToOne(Resource::Wheat),  38u8, 42u8), // upper-left
            (PortType::ThreeToOne,           36u8, 37u8), // top-left
            (PortType::TwoToOne(Resource::Ore),    33u8, 34u8), // top
            (PortType::TwoToOne(Resource::Sheep),  28u8, 31u8), // upper-right
        ];

        Self {
            tile_types,
            tile_numbers,
            ports,
        }
    }

    /// Which resource a hex produces (None for desert).
    pub fn hex_resource(&self, hex: usize) -> Option<Resource> {
        self.tile_types[hex].resource()
    }

    /// Which hexes produce on a given dice roll.
    pub fn producing_hexes(&self, roll: u8) -> impl Iterator<Item = usize> + '_ {
        (0..HEX_COUNT).filter(move |&h| self.tile_numbers[h] == roll)
    }

    /// Check if a vertex is on a port, and if so which type.
    pub fn port_at_vertex(&self, vertex: u8) -> Option<PortType> {
        for &(port_type, v1, v2) in &self.ports {
            if vertex == v1 || vertex == v2 {
                return Some(port_type);
            }
        }
        None
    }

    /// Get the desert hex index (where robber starts).
    pub fn desert_hex(&self) -> usize {
        self.tile_types
            .iter()
            .position(|&t| t == TileType::Desert)
            .unwrap_or(0)
    }

    /// Generate a random legal Catan board.
    ///
    /// Tiles are shuffled with the standard distribution (4W/3B/4Wh/3O/4S/1D).
    /// Number tokens are shuffled onto non-desert hexes ensuring no two adjacent
    /// hexes both carry a 6 or 8 (the standard "no adjacent red" rule).
    /// Port types are shuffled across the 9 fixed outer-edge port slots.
    pub fn random(rng: &mut impl rand::Rng) -> Self {
        use rand::seq::SliceRandom;

        // Standard tile counts: 4 Wood, 3 Brick, 4 Wheat, 3 Ore, 4 Sheep, 1 Desert
        let mut tiles = [
            TileType::Wood,  TileType::Wood,  TileType::Wood,  TileType::Wood,
            TileType::Brick, TileType::Brick, TileType::Brick,
            TileType::Wheat, TileType::Wheat, TileType::Wheat, TileType::Wheat,
            TileType::Ore,   TileType::Ore,   TileType::Ore,
            TileType::Sheep, TileType::Sheep, TileType::Sheep, TileType::Sheep,
            TileType::Desert,
        ];
        tiles.shuffle(rng);

        // Standard number tokens: one 2, two each of 3-6 and 8-11, one 12 (18 total)
        let numbers_pool: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

        // Retry until no two adjacent hexes share a 6 or 8
        let tile_numbers = loop {
            let mut shuffled = numbers_pool;
            shuffled.shuffle(rng);

            let mut tile_numbers = [0u8; HEX_COUNT];
            let mut idx = 0;
            for i in 0..HEX_COUNT {
                if tiles[i] != TileType::Desert {
                    tile_numbers[i] = shuffled[idx];
                    idx += 1;
                }
            }

            if !has_adjacent_red_numbers(&tile_numbers) {
                break tile_numbers;
            }
        };

        // Fixed port slot vertex-pairs on the outer boundary (clockwise from right)
        let port_slots: [(u8, u8); PORT_COUNT] = [
            (24, 25), (27, 53), (48, 52), (45, 46), (43, 44),
            (38, 42), (36, 37), (33, 34), (28, 31),
        ];

        let mut port_types = [
            PortType::ThreeToOne, PortType::ThreeToOne,
            PortType::ThreeToOne, PortType::ThreeToOne,
            PortType::TwoToOne(Resource::Brick),
            PortType::TwoToOne(Resource::Wood),
            PortType::TwoToOne(Resource::Wheat),
            PortType::TwoToOne(Resource::Ore),
            PortType::TwoToOne(Resource::Sheep),
        ];
        port_types.shuffle(rng);

        let mut ports = [(PortType::ThreeToOne, 0u8, 0u8); PORT_COUNT];
        for i in 0..PORT_COUNT {
            ports[i] = (port_types[i], port_slots[i].0, port_slots[i].1);
        }

        Self { tile_types: tiles, tile_numbers, ports }
    }

    /// Parse a board from the JSON format produced by `get_board_layout()`.
    /// Hexes are matched by their `id` field (canonical spiral index 0..18).
    /// Ports are consumed in order from the array.
    pub fn from_frontend_json(json: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;

        let hexes = v["hexes"].as_array().ok_or("missing hexes")?;
        if hexes.len() != HEX_COUNT {
            return Err(format!("expected {} hexes, got {}", HEX_COUNT, hexes.len()));
        }

        let mut tile_types = [TileType::Desert; HEX_COUNT];
        let mut tile_numbers = [0u8; HEX_COUNT];

        for hex in hexes {
            let id = hex["id"].as_u64().ok_or("missing hex id")? as usize;
            if id >= HEX_COUNT {
                return Err(format!("hex id {} out of range", id));
            }
            tile_types[id] = match hex["resource"].as_str().ok_or("missing resource")? {
                "wood"   => TileType::Wood,
                "brick"  => TileType::Brick,
                "wheat"  => TileType::Wheat,
                "ore"    => TileType::Ore,
                "sheep"  => TileType::Sheep,
                "desert" => TileType::Desert,
                r => return Err(format!("unknown resource '{}'", r)),
            };
            tile_numbers[id] = hex["number"].as_u64().unwrap_or(0) as u8;
        }

        let ports_json = v["ports"].as_array().ok_or("missing ports")?;
        if ports_json.len() != PORT_COUNT {
            return Err(format!("expected {} ports, got {}", PORT_COUNT, ports_json.len()));
        }

        let mut ports = [(PortType::ThreeToOne, 0u8, 0u8); PORT_COUNT];
        for (i, p) in ports_json.iter().enumerate() {
            let type_str = p["type"].as_str().ok_or("missing port type")?;
            let v1 = p["v1"].as_u64().ok_or("missing v1")? as u8;
            let v2 = p["v2"].as_u64().ok_or("missing v2")? as u8;
            let port_type = match type_str {
                "3:1" => PortType::ThreeToOne,
                "2:1:brick" => PortType::TwoToOne(Resource::Brick),
                "2:1:wood"  => PortType::TwoToOne(Resource::Wood),
                "2:1:wheat" => PortType::TwoToOne(Resource::Wheat),
                "2:1:ore"   => PortType::TwoToOne(Resource::Ore),
                "2:1:sheep" => PortType::TwoToOne(Resource::Sheep),
                t => return Err(format!("unknown port type '{}'", t)),
            };
            ports[i] = (port_type, v1, v2);
        }

        Ok(Self { tile_types, tile_numbers, ports })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_count() {
        assert_eq!(HEX_COORDS.len(), HEX_COUNT);
        for coord in &HEX_COORDS {
            assert!(coord.is_valid(), "Invalid hex coord: {:?}", coord);
        }
    }

    #[test]
    fn test_vertex_count() {
        // Verify we get exactly 54 unique vertices
        let mut max_vertex = 0u8;
        for h in 0..HEX_COUNT {
            for v in 0..6 {
                if HEX_VERTICES[h][v] > max_vertex {
                    max_vertex = HEX_VERTICES[h][v];
                }
            }
        }
        // max_vertex + 1 should equal VERTEX_COUNT
        assert_eq!(
            max_vertex as usize + 1,
            VERTEX_COUNT,
            "Expected {} vertices, got {}",
            VERTEX_COUNT,
            max_vertex + 1
        );
    }

    #[test]
    fn test_edge_count() {
        let mut max_edge = 0u8;
        for h in 0..HEX_COUNT {
            for e in 0..6 {
                if HEX_EDGES[h][e] > max_edge {
                    max_edge = HEX_EDGES[h][e];
                }
            }
        }
        assert_eq!(
            max_edge as usize + 1,
            EDGE_COUNT,
            "Expected {} edges, got {}",
            EDGE_COUNT,
            max_edge + 1
        );
    }

    #[test]
    fn test_vertex_adjacency() {
        // Each vertex should have 2 or 3 neighbors
        for v in 0..VERTEX_COUNT {
            let neighbors = vertex_neighbors(v);
            let count = neighbors.iter().filter(|&&n| n >= 0).count();
            assert!(
                count == 2 || count == 3,
                "Vertex {} has {} neighbors (expected 2 or 3)",
                v,
                count
            );
        }
    }

    #[test]
    fn test_dice_probabilities() {
        let total: f64 = (2..=12).map(|i| dice_probability(i)).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
}
