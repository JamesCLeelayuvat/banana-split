"""Sampling: temperature, top-p, grammar token masks, classifier-free guidance on text.

Usage: python -m src.generate --ckpt outputs/checkpoints/best.pt --chords "Dm7 G7 | Cmaj7 | ..." \
           --tempo 180 --text "a dense descending bebop line" --out outputs/samples/
"""
import argparse


def parse_changes(s: str, beats_per_bar: int = 4) -> list[str]:
    """'Dm7 G7 | Cmaj7' -> one chord symbol per beat.

    Bars are separated by '|'. The chords in a bar share its beats evenly, with any remainder
    going to the earlier chords (3 chords in 4/4 -> 2+1+1). A bar holding only '%' repeats the
    previous bar; an empty bar holds the previous chord.
    """
    beats: list[str] = []
    prev_bar = ["NC"]
    for bar in s.strip().strip("|").split("|"):
        syms = bar.split()
        if syms == ["%"]:
            syms = prev_bar
        elif not syms:
            syms = [beats[-1] if beats else "NC"]
        if len(syms) > beats_per_bar:
            raise ValueError(f"bar {bar!r} has more chords than beats")
        base, extra = divmod(beats_per_bar, len(syms))
        for i, sym in enumerate(syms):
            beats += [sym] * (base + (1 if i < extra else 0))
        prev_bar = syms
    return beats


def allowed_mask(prev_tokens: list[int]):
    """POS must increase within a bar, PITCH follows POS, DUR follows PITCH, pitch in range."""
    raise NotImplementedError


def sample(model, chords, text, bars=4, temperature=0.9, top_p=0.9, guidance=2.0):
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chords", required=True)
    ap.add_argument("--tempo", type=int, default=160)
    ap.add_argument("--text", default="")
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out", default="outputs/samples")
    args = ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
