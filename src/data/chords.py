"""Chord-symbol normalizer: any symbol in the data -> (root, quality), 12 roots x 9 qualities.

Handles Weimar notation (C-7, Bbj7, Ebj7911#, C79b, Fsus7913, Bo7, G+7, F/A) as well as common
lead-sheet spellings (Cmaj7, Dm7, Bm7b5, Bdim7, G7alt, A7sus4). Slash basses are dropped.
"""
from __future__ import annotations

import logging
import re

ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
QUALITIES = ["maj7", "6", "7", "m7", "m6", "m7b5", "dim7", "7alt", "sus"]
NO_CHORD = "NC"

log = logging.getLogger(__name__)

_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ROOT_RE = re.compile(r"^([A-G])([b#]*)(.*)$")

# Semitones above the root for each quality's chord tones.
INTERVALS = {
    "maj7": (0, 4, 7, 11),
    "6": (0, 4, 7, 9),
    "7": (0, 4, 7, 10),
    "m7": (0, 3, 7, 10),
    "m6": (0, 3, 7, 9),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "7alt": (0, 4, 8, 10),  # 3, #5, b7 (the b9/#9 are colour tones, not chord tones)
    "sus": (0, 5, 7, 10),
}


def _quality(q: str) -> tuple[str, bool]:
    """Suffix after the root -> (quality, exact). exact=False means a rare quality was mapped."""
    q = q.replace("maj", "j").replace("M7", "j7").replace("^", "j").replace("min", "-")
    q = q.replace("dim", "o").replace("ø", "m7b5").replace("h7", "m7b5").replace("aug", "+")
    if q.startswith("m") and not q.startswith("m7b5"):
        q = "-" + q[1:]
    if q.startswith("-7b5") or q.startswith("m7b5"):
        return "m7b5", True
    if q.startswith("o"):
        return "dim7", q in ("o", "o7")
    if q.startswith("-"):
        rest = q[1:]
        if "j" in rest:  # minor-major: closest tonic-minor quality
            return "m6", False
        if rest.startswith("6"):
            return "m6", rest in ("6", "69")
        return "m7", rest in ("", "7")
    if q.startswith("sus") or "sus" in q:
        return "sus", True
    if "alt" in q:
        return "7alt", True
    if q.startswith("+"):
        if "j" in q:
            return "maj7", False
        return "7alt", False
    if q.startswith("j") or q == "":
        return "maj7", q in ("", "j7")
    if q.startswith("6"):
        return "6", q in ("6", "69")
    if q[:1] in ("7", "9", "1"):  # 7, 9, 11, 13 and their tensions
        if any(t in q for t in ("9b", "9#", "13b", "b9", "#9", "b13", "#5", "+")):
            return "7alt", False
        return "7", True
    return "maj7", False


def parse(symbol: str) -> tuple[int, str] | None:
    """Return (root pitch class 0-11, quality) or None for no-chord.

    Rare qualities map to the nearest of the 9; mapped symbols are logged at DEBUG level.
    Raises ValueError for a symbol with no recognisable root.
    """
    s = symbol.strip()
    if s in ("", NO_CHORD, "N.C.", "N.C", "NC.", "X"):
        return None
    s = s.split("/")[0]  # drop slash bass
    m = _ROOT_RE.match(s)
    if not m:
        raise ValueError(f"unparseable chord symbol: {symbol!r}")
    letter, accidentals, suffix = m.groups()
    root = (_LETTER[letter] + accidentals.count("#") - accidentals.count("b")) % 12
    quality, exact = _quality(suffix)
    if not exact:
        log.debug("mapped chord %r -> %s%s", symbol, ROOTS[root], quality)
    return root, quality


def chord_tones(root: int, quality: str) -> set[int]:
    """Pitch classes of the chord (used by features and chord-fit eval)."""
    return {(root + i) % 12 for i in INTERVALS[quality]}


def transpose(chord: tuple[int, str] | None, semitones: int) -> tuple[int, str] | None:
    if chord is None:
        return None
    root, quality = chord
    return ((root + semitones) % 12, quality)
