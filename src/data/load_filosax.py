"""Load Filosax improvisations via mirdata (data/raw/filosax/).

If chord symbols don't ship with it, the 48 lead-sheet progressions are typed in by hand.
Output format matches load_weimar.load_solos().
"""
from pathlib import Path


def load_solos(data_home: str | Path = "data/raw/filosax") -> list[dict]:
    raise NotImplementedError
