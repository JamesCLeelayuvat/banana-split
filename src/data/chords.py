"""Chord-symbol normalizer: any symbol in the data -> (root, quality), 12 roots x 9 qualities."""

ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
QUALITIES = ["maj7", "6", "7", "m7", "m6", "m7b5", "dim7", "7alt", "sus"]
NO_CHORD = "NC"


def parse(symbol: str) -> tuple[int, str] | None:
    """Return (root pitch class 0-11, quality) or None for no-chord.

    Rare qualities map to the nearest of the 9; log what was mapped.
    """
    raise NotImplementedError


def chord_tones(root: int, quality: str) -> set[int]:
    """Pitch classes of the chord (used by features and chord-fit eval)."""
    raise NotImplementedError


def transpose(chord: tuple[int, str] | None, semitones: int) -> tuple[int, str] | None:
    if chord is None:
        return None
    root, quality = chord
    return ((root + semitones) % 12, quality)
