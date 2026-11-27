"""Caption features per phrase: numbers -> tercile words -> data/features/.

Each measurement is its own function so the evaluation (src/eval/control_test.py, chord_fit.py)
measures generated phrases with exactly the definition the captions were written from.

Numeric features (compute_features):
    mean_pitch      mean MIDI pitch (absolute, not instrument-relative)          -> register
    range           highest - lowest pitch, semitones                            -> range
    density         notes per beat                                               -> density
    slope           least-squares pitch slope, semitones per bar                 -> contour
    arch            middle third of the notes minus the higher outer third, semitones -> contour
    sixteenth_share share of onsets on the 2nd/4th 16th of a beat (ticks 3, 9)   -> rhythm
    triplet_share   share of onsets on a triplet partial: tick 4, or tick 8 when the same beat
                    also has a tick-4 onset (a lone 8 is read as a swung eighth)   -> rhythm
    triplet_notes   count of those onsets; "triplets" needs at least MIN_TRIPLET_NOTES
    mean_interval   mean |interval| between consecutive notes, semitones        -> intervals
    rest_fraction   share of the phrase's ticks with no note sounding            -> space
    chord_tone_rate share of on-beat notes that are tones of the beat's chord    -> harmony
    chromatic_rate  share of notes outside the chord scale that resolve by half step to a chord tone
    outside_rate    share of notes outside the chord scale that don't resolve that way
Context words (style, tempo, instrument) come straight from metadata.

Words (to_words): cut-offs are the training split's thirds (fit_bins). Three-word features always
get a word; two-word features (range, intervals, space) get one only in the outer thirds and
None in the middle, so captions don't describe a phrase as "medium-ranged" etc.

Only register changes under transposition (every other feature is pitch-class relative to the
chords, or pitch-free). The training dataset restricts shifts to ones that keep the register
word (allowed_shifts).

Run:  python -m src.captions.features   (writes data/features/features.jsonl and bins.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.chords import chord_tones, parse

ROOT = Path(__file__).resolve().parents[2]
TICKS_PER_BEAT = 12
BEATS_PER_BAR = 4

# Chord scales: pitch classes above the root that are "inside" for each quality. Deliberately
# generous (common tensions included), so "outside" means really outside the harmony.
SCALES = {
    "maj7": (0, 2, 4, 5, 6, 7, 9, 11),   # ionian + #11
    "6": (0, 2, 4, 5, 6, 7, 9, 11),
    "7": (0, 2, 4, 5, 6, 7, 9, 10),      # mixolydian + #11
    "m7": (0, 2, 3, 5, 7, 9, 10),        # dorian
    "m6": (0, 2, 3, 5, 7, 9, 10, 11),    # dorian + melodic minor 7
    "m7b5": (0, 1, 2, 3, 5, 6, 8, 10),   # locrian + natural 9
    "dim7": (0, 2, 3, 5, 6, 8, 9, 11),   # whole-half diminished
    "7alt": (0, 1, 3, 4, 6, 8, 10),      # altered
    "sus": (0, 2, 4, 5, 7, 9, 10),       # mixolydian
}

# Numeric feature -> words for its low / middle / high third (None = say nothing).
WORDS = {
    "register": ("mean_pitch", ("low", "middle", "high")),
    "density": ("density", ("sparse", "moderate", "dense")),
    "range": ("range", ("narrow", None, "wide")),
    "intervals": ("mean_interval", ("stepwise", None, "intervallic")),
    "space": ("rest_fraction", ("busy", None, "spacious")),
}
BINNED = ("mean_pitch", "density", "range", "mean_interval", "rest_fraction", "slope",
          "sixteenth_share", "triplet_share", "chromatic_rate", "outside_rate")
ARCH_SEMITONES = 3.0  # the middle third must sit this far above the higher outer third
MIN_TRIPLET_NOTES = 3  # fewer triplet onsets than this never makes a phrase "triplets"

STYLE = {"POSTBOP": "post-bop", "HARDBOP": "hard bop", "COOL": "cool jazz", "SWING": "swing",
         "BEBOP": "bebop", "FUSION": "fusion", "TRADITIONAL": "traditional"}
TEMPO = {"UP": "up-tempo", "MEDIUM UP": "medium-up", "MEDIUM": "medium tempo",
         "MEDIUM SLOW": "medium-slow", "SLOW": "slow"}
INSTRUMENT = {"ts": "tenor sax", "ts-c": "tenor sax", "as": "alto sax", "ss": "soprano sax",
              "bs": "baritone sax", "tp": "trumpet", "cor": "cornet", "tb": "trombone",
              "cl": "clarinet", "bcl": "bass clarinet", "p": "piano", "g": "guitar",
              "vib": "vibraphone"}


# ---------------------------------------------------------------- measurements

def _chord_at(phrase: dict, onset: int):
    chords = phrase["chords"]
    return parse(chords[min(onset // TICKS_PER_BEAT, len(chords) - 1)])


def _in_scale(pitch: int, chord) -> bool:
    root, quality = chord
    return (pitch - root) % 12 in SCALES[quality]


def mean_pitch(phrase: dict) -> float:
    return float(np.mean([n["pitch"] for n in phrase["notes"]]))


def density(phrase: dict) -> float:
    return len(phrase["notes"]) / len(phrase["chords"])


def contour(phrase: dict) -> tuple[float, float]:
    """(slope in semitones per bar, arch in semitones). Arch > 0 means the middle third of the
    notes sits above both outer thirds (by at least that much)."""
    notes = phrase["notes"]
    pitch = np.array([n["pitch"] for n in notes], float)
    if len(notes) < 3:
        return 0.0, 0.0
    t = np.array([n["onset"] for n in notes], float) / (TICKS_PER_BEAT * BEATS_PER_BAR)
    slope = float(np.polyfit(t, pitch, 1)[0]) if np.ptp(t) > 0 else 0.0
    first, middle, last = np.array_split(pitch, 3)
    return slope, float(middle.mean() - max(first.mean(), last.mean()))


def triplet_onsets(phrase: dict) -> int:
    """Onsets on a triplet partial: tick 4 of a beat, or tick 8 when that beat also has a
    tick-4 onset. A tick-8 onset without one ({0, 8} or a lone 8) is the swung off-beat eighth
    that some transcriptions kept, so it doesn't count."""
    onsets = [n["onset"] for n in phrase["notes"]]
    beats_with_4 = {o // TICKS_PER_BEAT for o in onsets if o % TICKS_PER_BEAT == 4}
    return sum(o % TICKS_PER_BEAT == 4
               or (o % TICKS_PER_BEAT == 8 and o // TICKS_PER_BEAT in beats_with_4)
               for o in onsets)


def rhythm_shares(phrase: dict) -> tuple[float, float]:
    """(sixteenth share, triplet share) of onsets, by position within the beat."""
    pos = np.array([n["onset"] % TICKS_PER_BEAT for n in phrase["notes"]])
    return float(np.isin(pos, (3, 9)).mean()), triplet_onsets(phrase) / len(pos)


def mean_interval(phrase: dict) -> float | None:
    pitch = [n["pitch"] for n in phrase["notes"]]
    return float(np.mean(np.abs(np.diff(pitch)))) if len(pitch) > 1 else None


def rest_fraction(phrase: dict) -> float:
    length = len(phrase["chords"]) * TICKS_PER_BEAT
    sounding = np.zeros(length, bool)
    for n in phrase["notes"]:
        sounding[n["onset"]:n["onset"] + n["dur"]] = True
    return float(1 - sounding.mean())


def chord_tone_rate(phrase: dict, beats=(0, 1, 2, 3)) -> float | None:
    """Share of notes starting exactly on one of `beats` (beat-in-bar indices) that are chord
    tones of that beat's chord. None if no such note has a chord."""
    hits = []
    for n in phrase["notes"]:
        on_beat = n["onset"] % TICKS_PER_BEAT == 0
        if on_beat and (n["onset"] // TICKS_PER_BEAT) % BEATS_PER_BAR in beats:
            chord = _chord_at(phrase, n["onset"])
            if chord is not None:
                hits.append(n["pitch"] % 12 in chord_tones(*chord))
    return float(np.mean(hits)) if hits else None


def harmonic_tension(phrase: dict) -> tuple[float | None, float | None]:
    """(chromatic_rate, outside_rate) over notes that have a chord.

    A note outside its chord's scale is a chromatic approach if the next note is a half step
    away and is a chord tone (of the next note's chord); otherwise it counts as outside."""
    notes = phrase["notes"]
    approach = outside = total = 0
    for i, n in enumerate(notes):
        chord = _chord_at(phrase, n["onset"])
        if chord is None:
            continue
        total += 1
        if _in_scale(n["pitch"], chord):
            continue
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        nxt_chord = _chord_at(phrase, nxt["onset"]) if nxt else None
        if (nxt_chord is not None and abs(nxt["pitch"] - n["pitch"]) == 1
                and nxt["pitch"] % 12 in chord_tones(*nxt_chord)):
            approach += 1
        else:
            outside += 1
    if not total:
        return None, None
    return approach / total, outside / total


def compute_features(phrase: dict) -> dict:
    """Numeric features + context for one phrase (see module docstring). None = not measurable."""
    pitch = [n["pitch"] for n in phrase["notes"]]
    slope, arch = contour(phrase)
    sixteenth, triplet = rhythm_shares(phrase)
    chromatic, outside = harmonic_tension(phrase)
    meta = phrase.get("meta", {})
    return {
        "mean_pitch": mean_pitch(phrase),
        "range": max(pitch) - min(pitch),
        "density": density(phrase),
        "slope": slope,
        "arch": arch,
        "sixteenth_share": sixteenth,
        "triplet_share": triplet,
        "triplet_notes": triplet_onsets(phrase),
        "mean_interval": mean_interval(phrase),
        "rest_fraction": rest_fraction(phrase),
        "chord_tone_rate": chord_tone_rate(phrase),
        "chromatic_rate": chromatic,
        "outside_rate": outside,
        "style": meta.get("style"),
        "tempoclass": meta.get("tempoclass"),
        "instrument": meta.get("instrument"),
    }


# ---------------------------------------------------------------- words

def fit_bins(train_features: list[dict]) -> dict:
    """Tercile cut-offs [low, high] per numeric feature, from the training split only."""
    bins = {}
    for key in BINNED:
        values = np.array([f[key] for f in train_features if f[key] is not None], float)
        bins[key] = [float(np.percentile(values, 100 / 3)), float(np.percentile(values, 200 / 3))]
    return bins


def third(value: float | None, cuts: list[float]) -> int | None:
    """0 / 1 / 2 for the low / middle / high third; None if unmeasured."""
    if value is None:
        return None
    return 0 if value <= cuts[0] else (2 if value > cuts[1] else 1)


def register_word(pitch: float, bins: dict) -> str:
    return WORDS["register"][1][third(pitch, bins["mean_pitch"])]


def allowed_shifts(phrase: dict, bins: dict, lo: int, hi: int) -> list[int]:
    """Transpositions in [lo, hi] that keep the phrase's register word (always includes 0)."""
    pitch = mean_pitch(phrase)
    word = register_word(pitch, bins)
    return [s for s in range(lo, hi + 1) if s == 0 or register_word(pitch + s, bins) == word]


def to_words(features: dict, bins: dict) -> dict:
    """e.g. {"register": "high", "density": "dense", "contour": "descending", ...}.
    None means "don't mention it"."""
    words = {name: options[third(features[key], bins[key])] if features[key] is not None else None
             for name, (key, options) in WORDS.items()}

    if features["arch"] >= ARCH_SEMITONES:
        words["contour"] = "arching"
    else:
        words["contour"] = ("descending", "level", "ascending")[third(features["slope"], bins["slope"])]

    sixteenth = third(features["sixteenth_share"], bins["sixteenth_share"])
    triplet = third(features["triplet_share"], bins["triplet_share"])
    if (triplet == 2 and features["triplet_share"] > features["sixteenth_share"]
            and features["triplet_notes"] >= MIN_TRIPLET_NOTES):
        words["rhythm"] = "triplets"
    elif sixteenth == 2:
        words["rhythm"] = "sixteenth runs"
    else:
        words["rhythm"] = "eighth notes"

    if third(features["outside_rate"], bins["outside_rate"]) == 2:
        words["harmony"] = "outside"
    elif third(features["chromatic_rate"], bins["chromatic_rate"]) == 2:
        words["harmony"] = "chromatic"
    elif features["outside_rate"] is not None:
        words["harmony"] = "inside"
    else:
        words["harmony"] = None

    words["style"] = STYLE.get(features["style"])
    words["tempo"] = TEMPO.get(features["tempoclass"])
    words["instrument"] = INSTRUMENT.get(features["instrument"])
    return words


# ---------------------------------------------------------------- build

def build(verbose: bool = True) -> dict:
    """Features + words for every phrase -> data/features/features.jsonl; cut-offs -> bins.json."""
    import yaml

    from src.data.phrases import SPLITS, load_phrases

    with open(ROOT / "configs" / "data.yaml") as f:
        out_dir = ROOT / yaml.safe_load(f)["paths"]["features"]
    by_split = {s: [(p["phrase_id"], compute_features(p)) for p in load_phrases(s)] for s in SPLITS}
    bins = fit_bins([f for _, f in by_split["train"]])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bins.json").write_text(json.dumps(bins, indent=2))
    with open(out_dir / "features.jsonl", "w", encoding="utf-8") as f:
        for split, rows in by_split.items():
            for pid, feats in rows:
                row = {"phrase_id": pid, "split": split, "features": feats,
                       "words": to_words(feats, bins)}
                f.write(json.dumps(row) + "\n")
    if verbose:
        n = sum(len(r) for r in by_split.values())
        print(f"{n} phrases -> {out_dir / 'features.jsonl'}")
        for key, cuts in bins.items():
            print(f"  {key:16s} thirds at {cuts[0]:.3f} / {cuts[1]:.3f}")
    return bins


def load_bins() -> dict:
    import yaml

    with open(ROOT / "configs" / "data.yaml") as f:
        out_dir = ROOT / yaml.safe_load(f)["paths"]["features"]
    return json.loads((out_dir / "bins.json").read_text())


if __name__ == "__main__":
    build()
