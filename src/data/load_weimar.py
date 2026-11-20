"""Load solos from the Weimar Jazz Database (data/raw/wjazzd.db).

Output per solo: notes (pitch, onset in beats, duration in beats), per-beat chords, metadata.
"""
import sqlite3
from pathlib import Path


def load_solos(db_path: str | Path = "data/raw/wjazzd.db") -> list[dict]:
    """Return a list of solos: {"solo_id", "notes", "chords", "meta"}."""
    raise NotImplementedError


if __name__ == "__main__":
    solos = load_solos()
    print(f"{len(solos)} solos")
