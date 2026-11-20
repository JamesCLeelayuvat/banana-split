"""Chord -> token id. 12 roots x 9 qualities = 108, plus NO_CHORD and PAD."""
from src.data.chords import QUALITIES, ROOTS

PAD_ID = 0
NO_CHORD_ID = 1
_OFFSET = 2
VOCAB_SIZE = _OFFSET + len(ROOTS) * len(QUALITIES)


def chord_to_id(chord: tuple[int, str] | None) -> int:
    if chord is None:
        return NO_CHORD_ID
    root, quality = chord
    return _OFFSET + root * len(QUALITIES) + QUALITIES.index(quality)


def id_to_chord(idx: int) -> tuple[int, str] | None:
    if idx in (PAD_ID, NO_CHORD_ID):
        return None
    root, q = divmod(idx - _OFFSET, len(QUALITIES))
    return (root, QUALITIES[q])
