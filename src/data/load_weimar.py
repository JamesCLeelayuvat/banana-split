"""Load solos from the Weimar Jazz Database (data/raw/wjazzd.db).

Output per solo (one dict per solo, onsets/durations in beats, floats, not yet quantized):
    {
      "solo_id": "weimar_<melid>",
      "notes":   [{"pitch": int, "onset": float, "dur": float}, ...],  # sorted by onset
      "chords":  [str, ...],      # one raw Weimar symbol per beat, e.g. "C-7", "Bbj7", "NC"
      "meta":    {"performer", "instrument", "style", "tempo", "tempoclass", "rhythmfeel",
                  "key", "title", "signature", "melid", "source"},
    }

Beat 0 is the downbeat of the solo's first bar, so bar b starts at beat 4*b. If the beat grid
starts mid-bar (a pickup), the missing beats before it are padded with "NC" chords.

How the database is turned into that:
- Onsets come from the annotated metrical position, not the performed time: a note on
  (bar, beat, tatum) of a beat split into `division` tatums sits at
  beat_index + (tatum - 1) / division. This drops micro-timing (swing ratio, laying back), so
  onsets land on the grid the transcriber heard (8ths, 16ths, triplets, ...).
- Durations are performed lengths: the note's start and end in seconds are mapped to beats by
  linear interpolation on the beat grid (`beats` table), and dur = end - start.
- Chords in `beats` are only written where they change; they're forward-filled to every beat.
  Beats before the first chord are "NC". Symbols are left raw for src/data/chords.parse().
- Only solos in regular 4/4 are kept: every bar except the first and last must have 4 beats.
  This drops solos in other meters and the few "4/4" solos with odd-length bars, since those
  would shift the bar lines that phrases are cut on.
"""
from __future__ import annotations

import sqlite3
from bisect import bisect_right
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "raw" / "wjazzd.db"
BEATS_PER_BAR = 4
NO_CHORD = "NC"


def _seconds_to_beats(t: float, grid: list[float], offset: int) -> float:
    """Map a time in seconds to a beat position, interpolating between grid beats.
    Times outside the grid are extrapolated from the nearest beat interval."""
    i = bisect_right(grid, t) - 1
    i = min(max(i, 0), len(grid) - 2)
    span = grid[i + 1] - grid[i]
    return offset + i + (t - grid[i]) / span


def _is_regular(bars: list[int]) -> bool:
    """True if every bar except the first and last has BEATS_PER_BAR beats and bar numbers
    are consecutive."""
    counts = Counter(bars)
    order = sorted(counts)
    if order != list(range(order[0], order[-1] + 1)):
        return False
    return all(counts[b] == BEATS_PER_BAR for b in order[1:-1])


def _load_solo(conn: sqlite3.Connection, melid: int, info: dict) -> dict | None:
    beat_rows = conn.execute(
        "SELECT onset, bar, beat, chord FROM beats WHERE melid = ? ORDER BY onset", (melid,)
    ).fetchall()
    if len(beat_rows) < 2 or not _is_regular([r[1] for r in beat_rows]):
        return None

    grid = [r[0] for r in beat_rows]
    offset = beat_rows[0][2] - 1  # pad a pickup so beat 0 is a downbeat
    beat_index = {(bar, beat): offset + i for i, (_, bar, beat, _) in enumerate(beat_rows)}

    chords = [NO_CHORD] * offset
    current = NO_CHORD
    for _, _, _, symbol in beat_rows:
        if symbol:
            current = symbol.strip()
        chords.append(current)

    notes = []
    for onset, pitch, dur, bar, beat, tatum, division in conn.execute(
        "SELECT onset, pitch, duration, bar, beat, tatum, division FROM melody "
        "WHERE melid = ? ORDER BY onset", (melid,)
    ):
        if (bar, beat) not in beat_index:
            return None  # note outside the beat grid; can't place it metrically
        length = _seconds_to_beats(onset + dur, grid, offset) - _seconds_to_beats(onset, grid, offset)
        start = beat_index[(bar, beat)] + (tatum - 1) / division
        notes.append({"pitch": int(round(pitch)), "onset": start, "dur": length})
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))

    return {"solo_id": f"weimar_{melid}", "notes": notes, "chords": chords, "meta": info}


def load_solos(db_path: str | Path = DEFAULT_DB, verbose: bool = True) -> list[dict]:
    """Return a list of solos: {"solo_id", "notes", "chords", "meta"}."""
    db_path = Path(db_path)
    if not db_path.is_absolute() and not db_path.exists():
        db_path = ROOT / db_path
    if not db_path.exists():
        raise FileNotFoundError(f"Weimar database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT melid, performer, title, instrument, style, avgtempo, tempoclass, "
            "rhythmfeel, key, signature FROM solo_info ORDER BY melid"
        ).fetchall()
        solos, skipped = [], []
        for melid, performer, title, instrument, style, tempo, tempoclass, feel, key, sig in rows:
            info = {
                "performer": performer,
                "instrument": instrument,
                "style": style,
                "tempo": tempo,
                "tempoclass": tempoclass,
                "rhythmfeel": feel,
                "key": key,
                "title": title,
                "signature": sig,
                "melid": melid,
                "source": "weimar",
            }
            solo = _load_solo(conn, melid, info) if sig == "4/4" else None
            if solo is None or not solo["notes"]:
                skipped.append((melid, title, sig))
            else:
                solos.append(solo)
    finally:
        conn.close()

    if verbose and skipped:
        print(f"load_weimar: skipped {len(skipped)} solos not in regular 4/4:")
        for melid, title, sig in skipped:
            print(f"  {melid:4d}  {sig or '?':4s}  {title}")
    return solos


if __name__ == "__main__":
    solos = load_solos()
    n_notes = sum(len(s["notes"]) for s in solos)
    print(f"{len(solos)} solos, {n_notes} notes")
    s = solos[0]
    print(s["solo_id"], s["meta"]["performer"], "-", s["meta"]["title"])
    print("  first notes:", s["notes"][:3])
    print("  first chords:", s["chords"][:8])
