"""Swing, backing and changes parsing (no FluidSynth needed)."""
from src import render
from src.generate import parse_changes
from src.tokens import remi

TPQ = render.TPQ


def test_parse_changes():
    assert parse_changes("Dm7 G7 | Cmaj7") == ["Dm7"] * 2 + ["G7"] * 2 + ["Cmaj7"] * 4
    assert parse_changes("| A7 | % |") == ["A7"] * 8
    assert parse_changes("C7 F7 G7") == ["C7", "C7", "F7", "G7"]


def test_swing_moves_offbeats_only():
    sw = lambda t: render._swing_tick(t, 2 / 3)
    assert [sw(x) for x in (0, TPQ // 2, TPQ)] == [0, 2 * TPQ // 3, TPQ]  # 8ths
    assert sw(TPQ // 3) == TPQ // 3                                        # triplet untouched
    assert render._swing_tick(TPQ // 2, 0.5) == TPQ // 2                   # 0.5 = straight


def test_backing_covers_solo_and_follows_chords():
    phrase = {"notes": [{"pitch": 67, "onset": i * 6, "dur": 6} for i in range(16)], "bars": 2}
    solo = remi.phrase_to_score(phrase, 160)
    chords = parse_changes("Dm7 G7 | Cmaj7")
    score = render.add_backing(solo, chords, 160, count_in=1)
    names = [t.name for t in score.tracks]
    assert names == ["Solo", "Bass", "Piano", "Drums"] and score.tracks[3].is_drum
    bass = score.tracks[1].notes
    # one bass note per beat of the 2 bars, plus the tag; downbeats are chord roots
    assert len(bass) == 2 * 4 + 1
    assert bass[0].pitch % 12 == 2 and bass[2].pitch % 12 == 7 and bass[4].pitch % 12 == 0
    assert bass[0].time == render.BAR  # after the count-in bar
    assert score.tracks[0].notes[0].time == render.BAR
