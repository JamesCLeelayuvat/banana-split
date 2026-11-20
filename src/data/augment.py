"""Transposition augmentation (-5..+6 semitones). Train split only.

No instrument-range filtering: the only limit is the tokenizer's pitch range 0-126, which a transposition
of real solo data never reaches in practice.
"""


def transpose_phrase(phrase: dict, semitones: int, pitch_range=(0, 126)) -> dict | None:
    """Return the transposed phrase, or None if any note falls outside the tokenizer's pitch range."""
    raise NotImplementedError


def augment(phrase: dict, lo: int = -5, hi: int = 6, pitch_range=(0, 126)) -> list[dict]:
    out = []
    for k in range(lo, hi + 1):
        p = transpose_phrase(phrase, k, pitch_range)
        if p is not None:
            out.append(p)
    return out
