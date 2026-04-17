/**
 * Static beginner board data — mirrors board.rs BoardLayout::beginner().
 * Used as fallback when /api/board isn't available yet.
 */

import type { BoardData } from '../types'

export const BEGINNER_BOARD: BoardData = {
  hexes: [
    { id: 0,  resource: 'desert', number: 0,  q: 0,  r: 0  },
    { id: 1,  resource: 'wheat',  number: 2,  q: 1,  r: 0  },
    { id: 2,  resource: 'sheep',  number: 3,  q: 1,  r: -1 },
    { id: 3,  resource: 'wood',   number: 4,  q: 0,  r: -1 },
    { id: 4,  resource: 'brick',  number: 5,  q: -1, r: 0  },
    { id: 5,  resource: 'ore',    number: 6,  q: -1, r: 1  },
    { id: 6,  resource: 'wheat',  number: 8,  q: 0,  r: 1  },
    { id: 7,  resource: 'wood',   number: 9,  q: 2,  r: 0  },
    { id: 8,  resource: 'sheep',  number: 10, q: 2,  r: -1 },
    { id: 9,  resource: 'brick',  number: 11, q: 2,  r: -2 },
    { id: 10, resource: 'ore',    number: 12, q: 1,  r: -2 },
    { id: 11, resource: 'wood',   number: 3,  q: 0,  r: -2 },
    { id: 12, resource: 'wheat',  number: 4,  q: -1, r: -1 },
    { id: 13, resource: 'sheep',  number: 5,  q: -2, r: 0  },
    { id: 14, resource: 'brick',  number: 6,  q: -2, r: 1  },
    { id: 15, resource: 'ore',    number: 8,  q: -2, r: 2  },
    { id: 16, resource: 'wheat',  number: 9,  q: -1, r: 2  },
    { id: 17, resource: 'sheep',  number: 10, q: 0,  r: 2  },
    { id: 18, resource: 'wood',   number: 11, q: 1,  r: 1  },
  ],
  ports: [
    { type: '3:1',       v1: 0,  v2: 1  },
    { type: '2:1:brick', v1: 3,  v2: 4  },
    { type: '3:1',       v1: 7,  v2: 8  },
    { type: '2:1:wood',  v1: 11, v2: 12 },
    { type: '3:1',       v1: 15, v2: 16 },
    { type: '2:1:wheat', v1: 19, v2: 20 },
    { type: '3:1',       v1: 23, v2: 24 },
    { type: '2:1:ore',   v1: 27, v2: 28 },
    { type: '2:1:sheep', v1: 31, v2: 32 },
  ],
}
