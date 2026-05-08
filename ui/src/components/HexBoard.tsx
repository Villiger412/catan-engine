import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { buildBoardGeometry, hexCenter, hexPolygonPoints } from '../lib/hexGeometry'
import type { BoardData, GamePosition, HexData } from '../types'
import { PLAYER_COLORS } from '../types'

// ── Colours ───────────────────────────────────────────────────────────────────
const TILE_STYLE: Record<string, { fill: string; stroke: string; icon: string }> = {
  wood:   { fill: '#2c6e1e', stroke: '#1e4d10', icon: '🌲' },
  brick:  { fill: '#b84820', stroke: '#883418', icon: '🧱' },
  wheat:  { fill: '#c8940e', stroke: '#9c7008', icon: '🌾' },
  ore:    { fill: '#6e8090', stroke: '#4e606e', icon: '⛰️' },
  sheep:  { fill: '#62b430', stroke: '#449020', icon: '🐑' },
  desert: { fill: '#c8a050', stroke: '#a07830', icon: '🏜️' },
}

const PORT_COLOR: Record<string, string> = {
  '3:1':       '#e8d5a0',
  '2:1:brick': '#b84820',
  '2:1:wood':  '#2c6e1e',
  '2:1:wheat': '#c8940e',
  '2:1:ore':   '#6e8090',
  '2:1:sheep': '#62b430',
}

const PORT_LABEL: Record<string, string> = {
  '3:1':       '3:1',
  '2:1:brick': 'Brick',
  '2:1:wood':  'Wood',
  '2:1:wheat': 'Grain',
  '2:1:ore':   'Ore',
  '2:1:sheep': 'Wool',
}

const PORT_ICON: Record<string, string> = {
  '3:1':       '⚓',
  '2:1:brick': '🧱',
  '2:1:wood':  '🌲',
  '2:1:wheat': '🌾',
  '2:1:ore':   '⛰️',
  '2:1:sheep': '🐑',
}

const DOT_COUNT: Record<number, number> = {
  2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}

// Cycle order for resource editing
const RESOURCE_CYCLE: HexData['resource'][] = ['wood', 'brick', 'wheat', 'ore', 'sheep', 'desert']
// Valid number tokens (no 7)
const NUMBER_CYCLE = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

// ── Layout constants ─────────────────────────────────────────────────────
// Every length on the board derives from HEX_SIZE. If you find yourself
// adding a new fixed pixel value here, ask whether it should be
// HEX_SIZE * <ratio> instead — that's what keeps hexes, ports and the
// viewBox in sync when any one of them changes. The viewBox itself is
// computed from the rendered content (see `viewBox` useMemo below) so
// nothing can ever drift out of frame.
const HEX_SIZE = 60
const BEACH_SIZE = HEX_SIZE * 3.42
const PORT_PUSH = HEX_SIZE * 0.7        // edge-midpoint → port badge centre
const VIEWBOX_PAD = HEX_SIZE * 0.45     // breathing room around the outermost content
const PORT_BADGE_W = 44
const PORT_BADGE_H = 52
const HINT_BAR_W = HEX_SIZE * 7

// Geometry origin is the centre hex (q=0, r=0). No CX/CY translation —
// every render site uses raw geometry coordinates and the viewBox is
// shifted to include them.
const BEACH_POINTS = hexPolygonPoints(0, 0, BEACH_SIZE)

// Piece cycle: none → P0 settlement → P1 → P2 → P3 → P0 city → P1 → P2 → P3 → none
type PieceState = { player: number; kind: 'settlement' | 'city' } | null

function getVertexPiece(vid: number, pos: GamePosition): PieceState {
  for (let p = 0; p < 4; p++) {
    if (pos.cities[p]?.includes(vid)) return { player: p, kind: 'city' }
    if (pos.settlements[p]?.includes(vid)) return { player: p, kind: 'settlement' }
  }
  return null
}

interface Props {
  board: BoardData
  /** When provided, the board becomes interactive (edit mode). */
  onBoardChange?: (board: BoardData) => void
  /** When provided, vertex and/or edge clicks mutate the position. */
  position?: GamePosition
  onPositionChange?: (p: GamePosition) => void
  /** If true, vertex clicks cycle piece ownership. */
  piecesMode?: boolean
  /** If true, edge clicks cycle road ownership. */
  roadsMode?: boolean
  /** If true, hex clicks move the robber. */
  robberMode?: boolean
}

function getEdgeOwner(eid: number, pos: GamePosition): number | null {
  for (let p = 0; p < 4; p++) {
    if (pos.roads[p]?.includes(eid)) return p
  }
  return null
}

export default function HexBoard({
  board, onBoardChange, position, onPositionChange,
  piecesMode = false, roadsMode = false, robberMode = false,
}: Props) {
  const editable = Boolean(onBoardChange)
  const geo = useMemo(() => buildBoardGeometry(HEX_SIZE), [])

  // Robber defaults to the desert unless caller specified.
  const desertHex = board.hexes.find(h => h.resource === 'desert')?.id ?? 0
  const robberHex = position?.robber_hex ?? desertHex

  function moveRobber(hexId: number) {
    if (!onPositionChange || !position) return
    if (hexId === robberHex) return
    onPositionChange({ ...position, robber_hex: hexId })
  }

  function cycleVertex(vid: number) {
    if (!onPositionChange || !position) return
    const current = getVertexPiece(vid, position)
    // Remove from all
    const s = position.settlements.map(arr => arr.filter(v => v !== vid))
    const c = position.cities.map(arr => arr.filter(v => v !== vid))
    if (current === null) {
      s[0] = [...s[0], vid]
    } else if (current.kind === 'settlement' && current.player < 3) {
      s[current.player + 1] = [...s[current.player + 1], vid]
    } else if (current.kind === 'settlement' && current.player === 3) {
      c[0] = [...c[0], vid]
    } else if (current.kind === 'city' && current.player < 3) {
      c[current.player + 1] = [...c[current.player + 1], vid]
    }
    // else city P3 → none (already removed)
    onPositionChange({ ...position, settlements: s, cities: c })
  }

  function cycleEdge(eid: number) {
    if (!onPositionChange || !position) return
    const owner = getEdgeOwner(eid, position)
    const roads = position.roads.map(arr => arr.filter(e => e !== eid))
    // none → P0 → P1 → P2 → P3 → none
    const next = owner === null ? 0 : owner === 3 ? -1 : owner + 1
    if (next >= 0) {
      roads[next] = [...roads[next], eid]
    }
    onPositionChange({ ...position, roads })
  }

  const portPositions = useMemo(() => {
    return board.ports.flatMap(port => {
      const v1 = geo.vertexPixels[port.v1]
      const v2 = geo.vertexPixels[port.v2]
      if (!v1 || !v2) return []
      const mx = (v1.x + v2.x) / 2
      const my = (v1.y + v2.y) / 2
      // Push along perpendicular bisector of v1-v2 so badge is equidistant from both vertices
      let px = -(v2.y - v1.y), py = v2.x - v1.x
      if (px * mx + py * my < 0) { px = -px; py = -py }
      const plen = Math.hypot(px, py) || 1
      const angle = Math.atan2(py, px) * 180 / Math.PI
      return [{
        x: mx + (px / plen) * PORT_PUSH, y: my + (py / plen) * PORT_PUSH, type: port.type, angle,
        v1x: v1.x, v1y: v1.y,
        v2x: v2.x, v2y: v2.y,
      }]
    })
  }, [board.ports, geo])

  // viewBox = bbox of every visible thing + padding. Computed from the
  // actual geometry so adding ports, resizing hexes, or changing PORT_PUSH
  // can never push content out of frame.
  const viewBox = useMemo(() => {
    const xs: number[] = []
    const ys: number[] = []

    // Beach hex corners (BEACH_POINTS in pointy-top form, centred at origin)
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 2 - (Math.PI / 3) * i
      xs.push(BEACH_SIZE * Math.cos(a))
      ys.push(-BEACH_SIZE * Math.sin(a))
    }
    // Vertex pixels
    for (const v of geo.vertexPixels) { xs.push(v.x); ys.push(v.y) }
    // Port badges (rect extents)
    for (const p of portPositions) {
      xs.push(p.x - PORT_BADGE_W / 2, p.x + PORT_BADGE_W / 2)
      ys.push(p.y - PORT_BADGE_H / 2, p.y + PORT_BADGE_H / 2)
    }

    const minX = Math.min(...xs) - VIEWBOX_PAD
    const maxX = Math.max(...xs) + VIEWBOX_PAD
    const minY = Math.min(...ys) - VIEWBOX_PAD
    const maxY = Math.max(...ys) + VIEWBOX_PAD
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY }
  }, [geo, portPositions])

  // ── Zoom / pan ─────────────────────────────────────────────────────────
  // The "natural" viewBox above is a property of the geometry. Zoom is a
  // transform applied on top: zoom shrinks the viewBox (showing less of the
  // world), pan slides it. All of this stays in the SVG coordinate system
  // so rendering remains crisp at any zoom level.
  const MIN_ZOOM = 0.6
  const MAX_ZOOM = 5
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const svgRef = useRef<SVGSVGElement>(null)
  const dragRef = useRef({ active: false, startX: 0, startY: 0, startPan: { x: 0, y: 0 }, moved: 0 })
  const pointersRef = useRef(new Map<number, { x: number; y: number }>())
  const pinchRef = useRef<{ dist: number; zoom: number; pan: { x: number; y: number }; cx: number; cy: number } | null>(null)

  const effectiveViewBox = useMemo(() => {
    const w = viewBox.w / zoom
    const h = viewBox.h / zoom
    return {
      x: viewBox.x + (viewBox.w - w) / 2 + pan.x,
      y: viewBox.y + (viewBox.h - h) / 2 + pan.y,
      w, h,
    }
  }, [viewBox, zoom, pan])

  const clampZoom = useCallback((z: number) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z)), [])

  // Pan that places `anchorSvg` at relative screen position (relX, relY) at zoom z.
  const panForAnchor = useCallback(
    (anchorSvgX: number, anchorSvgY: number, relX: number, relY: number, z: number) => {
      const w = viewBox.w / z
      const h = viewBox.h / z
      return {
        x: anchorSvgX - relX * w - viewBox.x - (viewBox.w - w) / 2,
        y: anchorSvgY - relY * h - viewBox.y - (viewBox.h - h) / 2,
      }
    },
    [viewBox],
  )

  const zoomAt = useCallback(
    (newZoomRaw: number, clientX: number, clientY: number) => {
      const svg = svgRef.current
      if (!svg) return
      const newZoom = clampZoom(newZoomRaw)
      if (newZoom === zoom) return
      const rect = svg.getBoundingClientRect()
      const relX = (clientX - rect.left) / rect.width
      const relY = (clientY - rect.top) / rect.height
      const svgX = effectiveViewBox.x + relX * effectiveViewBox.w
      const svgY = effectiveViewBox.y + relY * effectiveViewBox.h
      setZoom(newZoom)
      setPan(panForAnchor(svgX, svgY, relX, relY, newZoom))
    },
    [zoom, effectiveViewBox, clampZoom, panForAnchor],
  )

  // Wheel needs a non-passive listener so we can preventDefault and not scroll the page.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const factor = Math.exp(-e.deltaY * 0.0015)
      zoomAt(zoom * factor, e.clientX, e.clientY)
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [zoom, zoomAt])

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* untrusted/synthetic event */ }
    pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (pointersRef.current.size === 1) {
      dragRef.current = {
        active: true,
        startX: e.clientX,
        startY: e.clientY,
        startPan: { ...pan },
        moved: 0,
      }
    } else if (pointersRef.current.size === 2) {
      const pts = [...pointersRef.current.values()]
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1
      pinchRef.current = {
        dist,
        zoom,
        pan: { ...pan },
        cx: (pts[0].x + pts[1].x) / 2,
        cy: (pts[0].y + pts[1].y) / 2,
      }
      dragRef.current.active = false
    }
  }

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!pointersRef.current.has(e.pointerId)) return
    pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (pointersRef.current.size === 2 && pinchRef.current) {
      const pts = [...pointersRef.current.values()]
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) || 1
      const factor = dist / pinchRef.current.dist
      const newZoom = clampZoom(pinchRef.current.zoom * factor)

      // Anchor the pinch midpoint: the SVG point that was originally under it stays under it.
      const rect = e.currentTarget.getBoundingClientRect()
      const relX = (pinchRef.current.cx - rect.left) / rect.width
      const relY = (pinchRef.current.cy - rect.top) / rect.height
      const startW = viewBox.w / pinchRef.current.zoom
      const startH = viewBox.h / pinchRef.current.zoom
      const startX = viewBox.x + (viewBox.w - startW) / 2 + pinchRef.current.pan.x
      const startY = viewBox.y + (viewBox.h - startH) / 2 + pinchRef.current.pan.y
      const anchorSvgX = startX + relX * startW
      const anchorSvgY = startY + relY * startH

      setZoom(newZoom)
      setPan(panForAnchor(anchorSvgX, anchorSvgY, relX, relY, newZoom))
    } else if (pointersRef.current.size === 1 && dragRef.current.active) {
      const dx = e.clientX - dragRef.current.startX
      const dy = e.clientY - dragRef.current.startY
      dragRef.current.moved = Math.max(dragRef.current.moved, Math.hypot(dx, dy))
      const rect = e.currentTarget.getBoundingClientRect()
      const scaleX = effectiveViewBox.w / rect.width
      const scaleY = effectiveViewBox.h / rect.height
      setPan({
        x: dragRef.current.startPan.x - dx * scaleX,
        y: dragRef.current.startPan.y - dy * scaleY,
      })
    }
  }

  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    pointersRef.current.delete(e.pointerId)
    if (pointersRef.current.size < 2) pinchRef.current = null
    if (pointersRef.current.size === 0) dragRef.current.active = false
  }

  // If a drag occurred, swallow the click so it doesn't fire hex/vertex/edge handlers.
  const onClickCapture = (e: React.MouseEvent) => {
    if (dragRef.current.moved > 5) {
      e.stopPropagation()
      e.preventDefault()
      dragRef.current.moved = 0
    }
  }

  const zoomFromCenter = (factor: number) => {
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    zoomAt(zoom * factor, rect.left + rect.width / 2, rect.top + rect.height / 2)
  }
  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }) }
  const isReset = zoom === 1 && pan.x === 0 && pan.y === 0

  function cycleResource(hexId: number) {
    if (!onBoardChange) return
    const hex = board.hexes[hexId]
    const idx = RESOURCE_CYCLE.indexOf(hex.resource)
    const next = RESOURCE_CYCLE[(idx + 1) % RESOURCE_CYCLE.length]
    const newHexes = board.hexes.map(h =>
      h.id === hexId
        ? { ...h, resource: next, number: next === 'desert' ? 0 : (h.number || 5) }
        : h
    )
    onBoardChange({ ...board, hexes: newHexes })
  }

  function cycleNumber(hexId: number) {
    if (!onBoardChange) return
    const hex = board.hexes[hexId]
    if (hex.resource === 'desert') return
    const idx = NUMBER_CYCLE.indexOf(hex.number)
    const next = NUMBER_CYCLE[(idx + 1) % NUMBER_CYCLE.length]
    const newHexes = board.hexes.map(h => h.id === hexId ? { ...h, number: next } : h)
    onBoardChange({ ...board, hexes: newHexes })
  }

  return (
    <div className="hex-board">
    <svg
      ref={svgRef}
      viewBox={`${effectiveViewBox.x} ${effectiveViewBox.y} ${effectiveViewBox.w} ${effectiveViewBox.h}`}
      width="100%"
      height="100%"
      preserveAspectRatio="xMidYMid meet"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onClickCapture={onClickCapture}
      onDoubleClick={(e) => zoomAt(zoom * 1.6, e.clientX, e.clientY)}
      style={{
        display: 'block',
        overflow: 'visible',
        touchAction: 'none',
        userSelect: 'none',
        cursor: (editable || piecesMode || roadsMode || robberMode) ? 'pointer' : 'grab',
      }}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id="oceanGrad" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor="#175a8c" />
          <stop offset="60%" stopColor="#0a2e54" />
          <stop offset="100%" stopColor="#061a36" />
        </radialGradient>
        <radialGradient id="beachGrad" cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor="#e8d5a0" />
          <stop offset="100%" stopColor="#d4b96e" />
        </radialGradient>
        {Object.entries(TILE_STYLE).map(([res, s]) => (
          <linearGradient key={res} id={`tileGrad-${res}`} x1="0%" y1="0%" x2="60%" y2="100%">
            <stop offset="0%" stopColor={lighten(s.fill, 30)} />
            <stop offset="100%" stopColor={s.fill} />
          </linearGradient>
        ))}
        <filter id="tokenShadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="#00000055" />
        </filter>
        <filter id="hexInset" x="-5%" y="-5%" width="110%" height="110%">
          <feDropShadow dx="0" dy="3" stdDeviation="3" floodColor="#00000040" />
        </filter>
        <filter id="hexHover" x="-5%" y="-5%" width="110%" height="110%">
          <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#ffffff60" />
        </filter>
        <clipPath id="islandClip">
          <polygon points={BEACH_POINTS} />
        </clipPath>
      </defs>

      {/* ── Ocean background ── extends well past the viewBox so any container
           aspect ratio (wide desktop, tall mobile) gets a seamless ocean fill. */}
      <rect
        x={viewBox.x - viewBox.w}
        y={viewBox.y - viewBox.h}
        width={viewBox.w * 3}
        height={viewBox.h * 3}
        fill="url(#oceanGrad)"
      />


      {/* ── Beach island ── */}
      <polygon points={BEACH_POINTS} fill="url(#beachGrad)" />


      {/* ── Resource hexes ── */}
      {board.hexes.map(hex => {
        const c = hexCenter(hex.q, hex.r, HEX_SIZE)
        const cx = c.x, cy = c.y
        const style = TILE_STYLE[hex.resource]
        const pts = hexPolygonPoints(cx, cy, HEX_SIZE - 2)
        const isDesert = hex.resource === 'desert'
        const isHot = hex.number === 6 || hex.number === 8
        const dots = DOT_COUNT[hex.number] ?? 0

        const hexClick = editable
          ? () => cycleResource(hex.id)
          : robberMode ? () => moveRobber(hex.id) : undefined
        const hexStroke = robberMode && hex.id === robberHex ? '#f5a623' : style.stroke
        const hexStrokeWidth = robberMode && hex.id === robberHex ? 3 : 2
        return (
          <g key={hex.id} filter="url(#hexInset)">
            {/* Hex tile — clicking cycles resource or moves robber */}
            <polygon
              points={pts}
              fill={`url(#tileGrad-${hex.resource})`}
              stroke={hexStroke}
              strokeWidth={hexStrokeWidth}
              onClick={hexClick}
              style={(editable || robberMode) ? { cursor: 'pointer' } : undefined}
            />

            {/* Number token — clicking cycles number */}
            {!isDesert && <>
              <circle
                cx={cx} cy={cy} r={22}
                fill="#f5f0e0" stroke="#d4c080" strokeWidth="2"
                filter="url(#tokenShadow)"
                onClick={editable ? (e) => { e.stopPropagation(); cycleNumber(hex.id) } : undefined}
                style={editable ? { cursor: 'pointer' } : undefined}
              />
              <text
                x={cx} y={cy + 1}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={dots === 5 ? 16 : 18} fontWeight="800" fontFamily="Nunito, sans-serif"
                fill={isHot ? '#c0200a' : '#2a2010'}
                onClick={editable ? (e) => { e.stopPropagation(); cycleNumber(hex.id) } : undefined}
                style={editable ? { cursor: 'pointer', userSelect: 'none' } : { userSelect: 'none' }}
              >
                {hex.number}
              </text>
              <g transform={`translate(${cx}, ${cy + 14})`}>
                {Array.from({ length: dots }, (_, d) => (
                  <circle key={d} cx={(d - (dots - 1) / 2) * 5} cy={0} r={2}
                    fill={isHot ? '#c0200a' : '#2a2010'} />
                ))}
              </g>
            </>}

            {/* Resource icon */}
            <text
              x={cx} y={cy + (isDesert ? 8 : 36)}
              textAnchor="middle" dominantBaseline="middle"
              fontSize={isDesert ? 28 : 14}
              style={{ userSelect: 'none', pointerEvents: 'none' }}
            >
              {style.icon}
            </text>
          </g>
        )
      })}

      {/* ── Robber token ── */}
      {(() => {
        const hex = board.hexes[robberHex]
        if (!hex) return null
        const c = hexCenter(hex.q, hex.r, HEX_SIZE)
        const rx = c.x
        // Offset the robber away from the number token so both stay visible.
        const ry = c.y - HEX_SIZE * 0.55
        return (
          <g transform={`translate(${rx}, ${ry})`} style={{ pointerEvents: 'none' }}>
            <ellipse cx={0} cy={16} rx={12} ry={3} fill="#00000060" />
            <path d="M 0,-14 C 7,-14 9,-8 9,-2 L 9,8 L -9,8 L -9,-2 C -9,-8 -7,-14 0,-14 Z"
              fill="#1a1a1a" stroke="#f5a623" strokeWidth="1.5"
              style={{ filter: 'drop-shadow(0 2px 3px #000000a0)' }} />
            <circle cx={0} cy={-6} r={2} fill="#f5a623" />
          </g>
        )
      })()}

      {/* ── Edges: roads (and invisible click targets when in roads mode) ── */}
      {geo.edgeVertices.map(([v1, v2], eid) => {
        const a = geo.vertexPixels[v1]
        const b = geo.vertexPixels[v2]
        if (!a || !b) return null
        const x1 = a.x, y1 = a.y
        const x2 = b.x, y2 = b.y
        const owner = position ? getEdgeOwner(eid, position) : null
        const color = owner !== null ? PLAYER_COLORS[owner] : null
        return (
          <g key={`edge-${eid}`}
             onClick={roadsMode ? () => cycleEdge(eid) : undefined}
             style={roadsMode ? { cursor: 'pointer' } : undefined}>
            {/* Fat invisible hit target to make edge clicking comfortable */}
            {roadsMode && (
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#ffffff01" strokeWidth={16} strokeLinecap="round" />
            )}
            {/* Road rendering when owned */}
            {color && (
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={color} strokeWidth={6} strokeLinecap="round"
                style={{ filter: 'drop-shadow(0 1px 2px #00000080)' }} />
            )}
          </g>
        )
      })}

      {/* ── Vertices: dots in view mode, clickable pieces in pieces mode ── */}
      {geo.vertexPixels.map((vp, vid) => {
        const vx = vp.x
        const vy = vp.y
        const piece = position ? getVertexPiece(vid, position) : null
        const color = piece ? PLAYER_COLORS[piece.player] : null
        return (
          <g key={vid} transform={`translate(${vx}, ${vy})`}
             onClick={piecesMode ? () => cycleVertex(vid) : undefined}
             style={piecesMode ? { cursor: 'pointer' } : undefined}>
            {/* Empty vertex dot */}
            {!piece && (
              <circle r={piecesMode ? 7 : 4}
                fill={piecesMode ? '#ffffff18' : '#ffffff30'}
                stroke={piecesMode ? '#ffffff55' : '#ffffff50'}
                strokeWidth="1" />
            )}
            {/* Settlement: house pentagon */}
            {piece?.kind === 'settlement' && (
              <>
                <path d="M 0,-11 L 8,-4 L 8,5 L -8,5 L -8,-4 Z"
                  fill={color!} stroke="#fff" strokeWidth="1.5" />
                {piecesMode && <circle r={12} fill="transparent" />}
              </>
            )}
            {/* City: castle silhouette */}
            {piece?.kind === 'city' && (
              <>
                <path d="M -10,5 L -10,-1 L -6,-7 L -2,-7 L -2,-11 L 2,-11 L 2,-7 L 6,-7 L 10,-1 L 10,5 Z"
                  fill={color!} stroke="#fff" strokeWidth="1.5" />
                {piecesMode && <circle r={14} fill="transparent" />}
              </>
            )}
          </g>
        )
      })}

      {/* ── Port badges ── */}
      {portPositions.map((port, i) => {
        const fill = PORT_COLOR[port.type] ?? '#e8d5a0'
        const icon = PORT_ICON[port.type] ?? '⚓'
        const label = PORT_LABEL[port.type] ?? port.type
        const ratio = port.type === '3:1' ? '' : '2:1'
        return (
          <g key={i}>
            {/* Pier planks from edge vertices to port */}
            <line x1={port.v1x} y1={port.v1y} x2={port.x} y2={port.y}
              stroke="#6b4c1e" strokeWidth="4" strokeLinecap="round" opacity="0.85" />
            <line x1={port.v2x} y1={port.v2y} x2={port.x} y2={port.y}
              stroke="#6b4c1e" strokeWidth="4" strokeLinecap="round" opacity="0.85" />
            {/* Dock end cap */}
            <line x1={port.v1x} y1={port.v1y} x2={port.v2x} y2={port.v2y}
              stroke="#6b4c1e" strokeWidth="3" strokeLinecap="round" opacity="0.6" />

            {/* Badge */}
            <g transform={`translate(${port.x}, ${port.y})`}
              style={{ filter: 'drop-shadow(0 3px 6px #00000070)' }}>
              {/* Badge body */}
              <rect x={-22} y={-26} width={44} height={52} rx={10}
                fill={fill} stroke="#ffffffaa" strokeWidth="1.5" />
              {/* Darker bottom strip for ratio text */}
              <rect x={-22} y={12} width={44} height={14} rx={0}
                fill="#00000030" />
              <rect x={-22} y={16} width={44} height={10} rx={10}
                fill="#00000030" />
              {/* Icon */}
              <text x={0} y={-8}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={16} style={{ userSelect: 'none' }}>
                {icon}
              </text>
              {/* Resource label */}
              <text x={0} y={7}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={7.5} fontWeight="800"
                fontFamily="Nunito, sans-serif" fill="#fff"
                style={{ userSelect: 'none' }}>
                {label}
              </text>
              {/* Ratio text */}
              <text x={0} y={20}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={7} fontWeight="700"
                fontFamily="Nunito, sans-serif" fill="#ffffffcc"
                style={{ userSelect: 'none' }}>
                {ratio || label}
              </text>
            </g>
          </g>
        )
      })}

      {/* ── Mode hints — rendered last so they always appear above port badges ── */}
      {(editable || piecesMode || roadsMode || robberMode) && (() => {
        const hint =
          editable   ? 'Click tile → cycle resource · Click number → cycle token' :
          piecesMode ? 'Click vertex → cycle: Red S → Blue S → Green S → Orange S → cities → clear' :
          roadsMode  ? 'Click edge → cycle road owner: Red → Blue → Green → Orange → clear' :
                       'Click a hex → move the robber there (blocks that tile\'s production)'
        const W = Math.min(HINT_BAR_W, viewBox.w - HEX_SIZE * 0.4)
        const barY = viewBox.y + viewBox.h - HEX_SIZE * 0.35
        return (
          <g style={{ pointerEvents: 'none' }}>
            <rect
              x={-W / 2} y={barY}
              width={W} height={16} rx={6}
              fill="#000000bb"
            />
            <text
              x={0} y={barY + 11}
              textAnchor="middle" dominantBaseline="middle"
              fontSize={9} fill="#ffffffcc"
              fontFamily="Nunito, sans-serif" fontWeight="700"
            >
              {hint}
            </text>
          </g>
        )
      })()}
    </svg>
    <div className="board-zoom-controls">
      <button type="button" className="board-zoom-btn" onClick={() => zoomFromCenter(1.3)}
        aria-label="Zoom in" title="Zoom in (+)" disabled={zoom >= MAX_ZOOM - 1e-3}>+</button>
      <button type="button" className="board-zoom-btn" onClick={() => zoomFromCenter(1 / 1.3)}
        aria-label="Zoom out" title="Zoom out (−)" disabled={zoom <= MIN_ZOOM + 1e-3}>−</button>
      <button type="button" className="board-zoom-btn board-zoom-reset" onClick={resetView}
        aria-label="Reset view" title="Reset view" disabled={isReset}>↺</button>
    </div>
    </div>
  )
}

function lighten(hex: string, amount: number): string {
  const n = parseInt(hex.slice(1), 16)
  const r = Math.min(255, ((n >> 16) & 0xff) + amount)
  const g = Math.min(255, ((n >> 8) & 0xff) + amount)
  const b = Math.min(255, (n & 0xff) + amount)
  return `#${((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1)}`
}
