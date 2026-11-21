"""Transposition augmentation (-5..+6 semitones). Train split only.

No instrument-range filtering: the only limit is the tokenizer's pitch range 0-126, which a transposition
of real solo data never reaches in practice.
"""
from src.data.chords import NO_CHORD, ROOTS, parse, transpose


def transpose_phrase(phrase: dict, semitones: int, pitch_range=(0, 126)) -> dict | None:
    """Return the transposed phrase, or None if any note falls outside the tokenizer's pitch range.

    Notes shift by `semitones`; chords are re-spelled as normalized symbols (root + one of the 9
    qualities, e.g. "D-7" up 2 -> "Em7"), which parse() reads back to the same chord.
    """
    lo, hi = pitch_range
    notes = [{**n, "pitch": n["pitch"] + semitones} for n in phrase["notes"]]
    if any(not lo <= n["pitch"] <= hi for n in notes):
        return None
    out = {**phrase, "notes": notes}
    if "chords" in phrase:
        out["chords"] = [_transpose_symbol(c, semitones) for c in phrase["chords"]]
    return out


def _transpose_symbol(symbol: str, semitones: int) -> str:
    chord = transpose(parse(symbol), semitones)
    return NO_CHORD if chord is None else ROOTS[chord[0]] + chord[1]


def augment(phrase: dict, lo: int = -5, hi: int = 6, pitch_range=(0, 126)) -> list[dict]:
    out = []
    for k in range(lo, hi + 1):
        p = transpose_phrase(phrase, k, pitch_range)
        if p is not None:
            out.append(p)
    return out
