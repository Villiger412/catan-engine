import { describe, it, expect } from 'vitest'
import { buildRecordsCsv } from './csv'
import type { RecordsResult } from '../types'

const MOCK: RecordsResult = {
  games_run: 200,
  elapsed_ms: 1234.5,
  games_per_sec: 162.0,
  policy: 'rule_based',
  turns_min: 30,
  turns_p25: 48.0,
  turns_median: 54.0,
  turns_mean: 55.2,
  turns_p75: 62.0,
  turns_max: 98,
  seat_stats: [
    { win_rate: 0.31, avg_vp: 8.2, avg_cities: 1.2, avg_settlements: 2.1, avg_roads: 8.3, avg_knights: 1.3, lr_rate: 0.34, la_rate: 0.35 },
    { win_rate: 0.25, avg_vp: 7.9, avg_cities: 1.1, avg_settlements: 2.0, avg_roads: 7.9, avg_knights: 1.2, lr_rate: 0.28, la_rate: 0.27 },
    { win_rate: 0.22, avg_vp: 7.8, avg_cities: 1.0, avg_settlements: 2.1, avg_roads: 8.1, avg_knights: 1.1, lr_rate: 0.22, la_rate: 0.21 },
    { win_rate: 0.22, avg_vp: 7.7, avg_cities: 1.0, avg_settlements: 2.0, avg_roads: 7.8, avg_knights: 1.0, lr_rate: 0.16, la_rate: 0.17 },
  ],
}

describe('buildRecordsCsv', () => {
  it('produces a header row as the first line', () => {           // 17
    const csv = buildRecordsCsv(MOCK)
    const firstLine = csv.split('\n')[0]
    expect(firstLine).toContain('seat')
    expect(firstLine).toContain('win_rate')
    expect(firstLine).toContain('turns_median')
  })

  it('produces exactly 4 data rows (one per seat)', () => {       // 18
    const csv = buildRecordsCsv(MOCK)
    const lines = csv.split('\n').filter(Boolean)
    expect(lines.length).toBe(5) // 1 header + 4 seats
  })

  it('quotes values that contain commas', () => {                 // 19
    const tricky: RecordsResult = {
      ...MOCK,
      policy: 'rule,based',  // comma in field
    }
    const csv = buildRecordsCsv(tricky)
    expect(csv).toContain('"rule,based"')
  })

  it('zero win_rate seats round-trip correctly', () => {          // 20
    const zeroed: RecordsResult = {
      ...MOCK,
      seat_stats: MOCK.seat_stats.map(s => ({ ...s, win_rate: 0.0 })),
    }
    const csv = buildRecordsCsv(zeroed)
    const dataRows = csv.split('\n').slice(1)
    for (const row of dataRows) {
      const cols = row.split(',')
      const winRateIdx = csv.split('\n')[0].split(',').indexOf('win_rate')
      expect(parseFloat(cols[winRateIdx])).toBe(0)
    }
  })
})
