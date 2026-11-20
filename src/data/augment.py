"""Transposition augmentation (-5..+6 semitones) with instrument-range checks. Train split only."""


def transpose_phrase(phrase: dict, semitones: int, pitch_range=(46, 96)) -> dict | None:
    """Return the transposed phrase, or None if any note falls outside pitch_range."""
    raise NotImplementedError


def augment(phrase: dict, lo: int = -5, hi: int = 6, pitch_range=(46, 96)) -> list[dict]:
    out = []
    for k in range(lo, hi + 1):
        p = transpose_phrase(phrase, k, pitch_range)
        if p is not None:
            out.append(p)
    return out
