"""Caption features: hand-built phrases with known answers, words, and register-safe shifts."""
import pytest

from src.captions import features as F
from src.data.dataset import PhraseDataset


def phrase(pitches, onsets=None, dur=6, chords=("C7",) * 16, meta=None):
    onsets = onsets if onsets is not None else [6 * i for i in range(len(pitches))]
    return {"phrase_id": "p", "bars": 4, "chords": list(chords), "meta": meta or {},
            "notes": [{"pitch": p, "onset": o, "dur": dur} for p, o in zip(pitches, onsets)]}


def test_basic_measurements():
    p = phrase([60, 62, 64, 65], dur=6)  # four eighth notes from beat 0
    f = F.compute_features(p)
    assert f["mean_pitch"] == pytest.approx(62.75)
    assert f["range"] == 5
    assert f["density"] == pytest.approx(4 / 16)
    assert f["mean_interval"] == pytest.approx(5 / 3)
    assert f["rest_fraction"] == pytest.approx(1 - 24 / 192)
    assert f["sixteenth_share"] == 0 and f["triplet_share"] == 0


def test_contour_slope_and_arch():
    up = phrase(list(range(60, 72)))
    down = phrase(list(range(72, 60, -1)))
    assert F.contour(up)[0] > 0 > F.contour(down)[0]
    arch = phrase([60, 61, 62, 70, 71, 72, 61, 60, 59])
    assert F.contour(arch)[1] >= F.ARCH_SEMITONES


def test_rhythm_shares():
    p = phrase([60] * 4, onsets=[0, 3, 4, 8], dur=1)
    assert F.rhythm_shares(p) == (pytest.approx(0.25), pytest.approx(0.5))


def test_swung_eighths_are_not_triplets():
    # Beat 0: {0, 8} long-short pair; beat 1: lone 8; beat 2: {0, 4}; beat 3: full triplet.
    p = phrase([60] * 8, onsets=[0, 8, 20, 24, 28, 36, 40, 44], dur=1)
    assert F.triplet_onsets(p) == 3  # tick 4 in beat 2, ticks 4 and 8 in beat 3


def test_triplets_word_needs_enough_triplet_notes():
    bins = {k: [0.0, 0.0] for k in F.BINNED}
    few = F.compute_features(phrase([60, 62, 64, 65], onsets=[0, 4, 12, 24], dur=1))
    assert few["triplet_share"] > few["sixteenth_share"]
    assert F.to_words(few, bins)["rhythm"] != "triplets"  # only one triplet onset
    many = F.compute_features(phrase([60] * 6, onsets=[0, 4, 8, 12, 16, 20], dur=1))
    assert F.to_words(many, bins)["rhythm"] == "triplets"


def test_harmony():
    # Over C7: E on beat 0 (chord tone), Db -> C (chromatic approach), F# leaping away (in
    # C7's scale: #11), Ab -> D (outside: not in mixolydian+#11, doesn't resolve by half step).
    p = phrase([64, 61, 60, 66, 56, 62], onsets=[0, 12, 18, 24, 36, 42])
    assert F.chord_tone_rate(p) == pytest.approx(1 / 4)  # on-beat notes: E, Db, F#, Ab
    chromatic, outside = F.harmonic_tension(p)
    assert chromatic == pytest.approx(1 / 6) and outside == pytest.approx(1 / 6)


def test_no_chord_notes_are_unmeasured():
    p = phrase([60, 62], chords=("NC",) * 16)
    assert F.chord_tone_rate(p) is None
    assert F.harmonic_tension(p) == (None, None)


def test_words_follow_bins():
    p = phrase([60, 62, 64, 65], meta={"style": "BEBOP", "tempoclass": "UP", "instrument": "ts"})
    f = F.compute_features(p)
    bins = {k: [0.0, 0.0] for k in F.BINNED}  # everything lands in the top third
    bins["mean_pitch"] = [70.0, 80.0]         # ...except register: low
    w = F.to_words(f, bins)
    assert w["register"] == "low" and w["density"] == "dense" and w["range"] == "wide"
    assert w["harmony"] == "inside"  # C D E F are all in C7's scale: outside_rate 0
    assert (w["style"], w["tempo"], w["instrument"]) == ("bebop", "up-tempo", "tenor sax")


def test_allowed_shifts_keep_register_word():
    bins = {"mean_pitch": [63.0, 68.0]}
    p = phrase([65, 66])  # mean 65.5: middle
    shifts = F.allowed_shifts(p, bins, -5, 6)
    assert shifts == [-2, -1, 0, 1, 2]
    assert all(F.register_word(65.5 + s, bins) == "middle" for s in shifts)


def test_dataset_uses_allowed_shifts():
    bins = {"mean_pitch": [63.0, 68.0]}
    p = phrase([65, 66])
    ds = PhraseDataset([p], max_len=512, transpose=(-5, 6), register_bins=bins, seed=1)
    assert ds.shifts == [[-2, -1, 0, 1, 2]]
    unrestricted = PhraseDataset([p], max_len=512, transpose=(-5, 6))
    assert unrestricted.shifts == [list(range(-5, 7))]
