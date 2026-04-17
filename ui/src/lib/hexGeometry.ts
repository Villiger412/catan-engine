/**
 * Hex board geometry for pointy-top hexagons using axial coordinates (q, r).
 *
 * Matches the coordinate system in board.rs:
 *   - 19 hexes in spiral order, ring 0 → ring 2
 *   - Vertices labeled N(0), NE(1), SE(2), S(3), SW(4), NW(5)
 *   - Abstract deduplication via (3q + local_dx, 3r + local_dy)
 */

export const HEX_COORDS_QR: [number, number][] = [
  // Center
  [0, 0],
  // Ring 1
  [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1],
  // Ring 2
  [2, 0], [2, -1], [2, -2], [1, -2], [0, -2], [-1, -1],
  [-2, 0], [-2, 1], [-2, 2], [-1, 2], [0, 2], [1, 1],
]

/** Abstract grid offsets for each of the 6 local vertices (N, NE, SE, S, SW, NW). */
const LOCAL_OFFSETS: [number, number][] = [
  [1, -2],  // N
  [2, -1],  // NE
  [1, 1],   // SE
  [-1, 2],  // S
  [-2, 1],  // SW
  [-1, -1], // NW
]

export interface Point { x: number; y: number }

/** Convert axial (q,r) to pixel center for pointy-top hexagons. */
export function hexCenter(q: number, r: number, size: number): Point {
  return {
    x: size * Math.sqrt(3) * q + size * (Math.sqrt(3) / 2) * r,
    y: size * 1.5 * r,
  }
}

/**
 * Convert abstract grid position to pixel coordinates.
 * Abstract grid: hex center at (3q, 3r); pixel center via hexCenter.
 * Linear transform:  gx,gy → pixel uses the same basis.
 */
function abstractToPixel(gx: number, gy: number, size: number): Point {
  return {
    x: gx * (size * Math.sqrt(3) / 3) + gy * (size * Math.sqrt(3) / 6),
    y: gy * (size / 2),
  }
}

export interface BoardGeometry {
  /** vertexPixels[vertexId] = pixel position */
  vertexPixels: Point[]
  /** hexVertices[hexIdx][localIdx 0..5] = vertexId */
  hexVertices: number[][]
  /** hexEdges[hexIdx][localIdx 0..5] = edgeId */
  hexEdges: number[][]
  /** edgeVertices[edgeId] = [v1, v2] */
  edgeVertices: [number, number][]
}

/**
 * Build the full board geometry from scratch, replicating the Rust const fn
 * deduplication so vertex / edge IDs match the engine exactly.
 */
export function buildBoardGeometry(size: number): BoardGeometry {
  // ── Vertex deduplication ───────────────────────────────────────────────
  const abstractKey = (gx: number, gy: number) => `${gx},${gy}`
  const keyToId = new Map<string, number>()
  const vertexPixels: Point[] = []
  const hexVertices: number[][] = []

  for (const [q, r] of HEX_COORDS_QR) {
    const cx = 3 * q, cy = 3 * r
    const verts: number[] = []
    for (const [da, db] of LOCAL_OFFSETS) {
      const gx = cx + da, gy = cy + db
      const key = abstractKey(gx, gy)
      let id = keyToId.get(key)
      if (id === undefined) {
        id = vertexPixels.length
        keyToId.set(key, id)
        vertexPixels.push(abstractToPixel(gx, gy, size))
      }
      verts.push(id)
    }
    hexVertices.push(verts)
  }

  // ── Edge deduplication ─────────────────────────────────────────────────
  // Edge local i connects vertex local i → vertex local (i+1)%6
  const edgeKeyToId = new Map<string, number>()
  const edgeVertices: [number, number][] = []
  const hexEdges: number[][] = []

  for (const verts of hexVertices) {
    const edges: number[] = []
    for (let i = 0; i < 6; i++) {
      const v1 = verts[i], v2 = verts[(i + 1) % 6]
      const key = v1 < v2 ? `${v1},${v2}` : `${v2},${v1}`
      let id = edgeKeyToId.get(key)
      if (id === undefined) {
        id = edgeVertices.length
        edgeKeyToId.set(key, id)
        edgeVertices.push([v1, v2])
      }
      edges.push(id)
    }
    hexEdges.push(edges)
  }

  return { vertexPixels, hexVertices, hexEdges, edgeVertices }
}

/** Polygon points string for an SVG <polygon> centered at (cx, cy). */
export function hexPolygonPoints(cx: number, cy: number, size: number): string {
  // Pointy-top: vertices at angles 90°, 30°, -30°, -90°, -150°, 150°
  return Array.from({ length: 6 }, (_, i) => {
    const angle = Math.PI / 2 - (Math.PI / 3) * i
    return `${cx + size * Math.cos(angle)},${cy - size * Math.sin(angle)}`
  }).join(' ')
}

/** Midpoint between two points (for edge/road rendering). */
export function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
}
