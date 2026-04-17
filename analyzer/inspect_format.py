#!/usr/bin/env python3
"""
inspect_format.py
=================
Downloads the first game from the dataset and pretty-prints its structure
so we can verify the coordinate system and adapt the parser if needed.

Run BEFORE analyze.py to sanity-check the coordinate mapping.
"""

import json, tarfile, sys
from pathlib import Path
import requests

DATASET_URL = (
    "https://github.com/Catan-data/dataset/releases/download/"
    "v1.0.0/games.tar.gz"
)
DATA_DIR = Path(__file__).parent / "data"
TARBALL  = DATA_DIR / "games.tar.gz"
OUT      = DATA_DIR / "sample_game.json"


def download_first_game():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Streaming tarball from:\n  {DATASET_URL}\n")
    print("Will stop as soon as the first .json file is extracted...\n")

    with requests.get(DATASET_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        # Write to disk incrementally so tarfile can read it
        buf_path = DATA_DIR / "_stream.tar.gz"
        with open(buf_path, "wb") as f_buf:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                f_buf.write(chunk)
                # Try to open tarball and find first json
                try:
                    with tarfile.open(buf_path, "r:gz") as tf:
                        for member in tf:
                            if member.name.endswith(".json"):
                                fobj = tf.extractfile(member)
                                if fobj:
                                    data = json.loads(fobj.read())
                                    return data
                except (tarfile.TarError, EOFError):
                    pass  # not enough data yet, keep downloading
    return None


def inspect(data):
    """Print the fields we need for the coordinate mapping."""
    print("=" * 70)
    print("TOP-LEVEL KEYS:", list(data.keys()))

    d = data.get("data", {})
    print("\ndata KEYS:", list(d.keys()))
    print("playOrder:", d.get("playOrder"))

    eh = d.get("eventHistory", {})
    print("\neventHistory KEYS:", list(eh.keys()))
    print("totalTurnCount:", eh.get("totalTurnCount"))

    init = eh.get("initialState", {})
    print("\ninitialState KEYS:", list(init.keys()))

    tiles = (init.get("tiles") or init.get("hexes") or
             init.get("board", {}).get("tiles"))
    if tiles:
        print(f"\nTILES ({len(tiles)} total) — first 4 examples:")
        for t in tiles[:4]:
            print(" ", json.dumps(t))

    ports = (init.get("harbors") or init.get("ports") or
             init.get("board", {}).get("ports"))
    if ports:
        print(f"\nPORTS ({len(ports)} total) — first 3 examples:")
        for p in (ports or [])[:3]:
            print(" ", json.dumps(p))

    # Scan events for building placements (type 4)
    events = eh.get("events", [])
    print(f"\nTotal events: {len(events)}")
    print("\nFIRST 20 EVENTS — looking for type-4 (building placement):")
    placed = 0
    for ev in events[:30]:
        sc = ev.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        for _, log in gls.items():
            txt = log.get("text", {})
            if txt.get("type") == 4:
                corners = sc.get("mapState", {}).get("tileCornerStates", {})
                print(f"  BUILD event — player {txt.get('playerColor')}")
                print(f"  ALL tileCornerStates keys (settlements/cities present):")
                for k, v in corners.items():
                    if v.get("owner") is not None:
                        print(f"    corner_id={k!r}  → {v}")
                placed += 1
                if placed >= 8:
                    break
        if placed >= 8:
            break

    # End game
    end = eh.get("endGameState", {})
    print("\nendGameState KEYS:", list(end.keys()))
    players = end.get("players", {})
    print("players:")
    for color, pd in players.items():
        print(f"  color={color}: rank={pd.get('rank')}, winner={pd.get('winningPlayer')}, "
              f"vp={pd.get('victoryPoints')}")

    print("\n" + "=" * 70)
    print("\nFull initialState (for detailed inspection):")
    print(json.dumps(init, indent=2)[:4000])


def main():
    if OUT.exists():
        print(f"Using cached sample: {OUT}")
        with open(OUT) as f:
            data = json.load(f)
    else:
        data = download_first_game()
        if not data:
            print("Failed to extract game from tarball.")
            sys.exit(1)
        with open(OUT, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved to: {OUT}\n")

    inspect(data)


if __name__ == "__main__":
    main()
