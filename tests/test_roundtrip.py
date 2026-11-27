"""Tokens -> MIDI -> tokens must give the same thing."""
import random

import pytest

from src.tokens import remi

PHRASE = {
    "notes": [
        {"pitch": 67, "onset": 0, "dur": 6},
        {"pitch": 65, "onset": 6, "dur": 6},
        {"pitch": 64, "onset": 12, "dur": 3},
        {"pitch": 62, "onset": 60, "dur": 12},  # second bar, after a rest
    ]
}


def random_phrase(rng: random.Random, bars: int = 4) -> dict:
    """Monophonic phrase on the 12-tick grid, like the phrase cutter produces."""
    notes, t, end = [], rng.randrange(0, 24), bars * remi.TICKS_PER_BAR
    while t < end:
        dur = rng.choice([rng.randint(1, 12), rng.randint(1, 24), rng.randint(1, remi.MAX_DUR)])
        notes.append({"pitch": rng.randint(remi.PITCH_MIN, remi.PITCH_MAX), "onset": t, "dur": dur})
        t += dur + rng.choice([0, 0, rng.randint(1, 30)])  # next note starts after this one ends
    return {"notes": notes, "bars": max(bars, (notes[-1]["onset"] // remi.TICKS_PER_BAR) + 1)}


def test_encode_decode_roundtrip():
    tokens = remi.encode(PHRASE)
    assert remi.decode(tokens)["notes"] == PHRASE["notes"]
    assert remi.encode(remi.decode(tokens)) == tokens


def test_token_layout_matches_design_doc():
    s = remi.ids_to_strings(remi.encode(PHRASE))
    assert s[:8] == ["BOS_None", "Bar_None", "Position_0", "Pitch_67", "Duration_0.6.12",
                     "Position_6", "Pitch_65", "Duration_0.6.12"]
    assert s.count("Bar_None") == 2
    assert s[-1] == "EOS_None"
    assert not any(x.startswith("Velocity") for x in s)


def test_vocab_size():
    names = set(remi.vocab())
    assert sum(n.startswith("Position_") for n in names) == remi.TICKS_PER_BAR
    n_pitch = remi.PITCH_MAX - remi.PITCH_MIN + 2  # + the unused Pitch_{PITCH_MAX + 1}
    assert sum(n.startswith("Pitch_") for n in names) == n_pitch
    assert sum(n.startswith("Duration_") for n in names) == remi.MAX_DUR
    assert {"PAD_None", "BOS_None", "EOS_None", "Bar_None"} <= names
    assert len(names) == remi.TICKS_PER_BAR + n_pitch + remi.MAX_DUR + 4  # + PAD BOS EOS Bar


@pytest.mark.parametrize("seed", range(50))
def test_random_phrases_roundtrip(seed):
    phrase = random_phrase(random.Random(seed))
    tokens = remi.encode(phrase)
    out = remi.decode(tokens)
    assert out["notes"] == sorted(phrase["notes"], key=lambda n: (n["onset"], n["pitch"]))
    assert out["bars"] == phrase["bars"]
    assert remi.encode(out) == tokens


def test_midi_file_roundtrip(tmp_path):
    phrase = random_phrase(random.Random(123))
    path = tmp_path / "phrase.mid"
    remi.to_midi(phrase, path)
    back = remi.from_midi(path)
    assert back["notes"] == phrase["notes"]
    assert remi.encode(back) == remi.encode({"notes": phrase["notes"]})


def test_trailing_empty_bars_are_padded():
    tokens = remi.encode({**PHRASE, "bars": 4})
    assert remi.ids_to_strings(tokens).count("Bar_None") == 4
    assert remi.decode(tokens)["bars"] == 4


def test_empty_phrase():
    tokens = remi.encode({"notes": [], "bars": 4})
    assert remi.ids_to_strings(tokens) == ["BOS_None"] + ["Bar_None"] * 4 + ["EOS_None"]
    assert remi.decode(tokens) == {"notes": [], "bars": 4}


@pytest.mark.parametrize("bad", [
    {"pitch": remi.PITCH_MIN - 1, "onset": 0, "dur": 6},
    {"pitch": remi.PITCH_MAX + 1, "onset": 0, "dur": 6},
    {"pitch": 60, "onset": 0, "dur": 0},
    {"pitch": 60, "onset": 0, "dur": remi.MAX_DUR + 1},
    {"pitch": 60, "onset": -1, "dur": 6},
    {"pitch": 60, "onset": 0.5, "dur": 6},
])
def test_unrepresentable_notes_raise(bad):
    with pytest.raises(remi.TokenizeError):
        remi.encode({"notes": [bad]})


def test_too_many_bars_raises():
    with pytest.raises(remi.TokenizeError):
        remi.encode({**PHRASE, "bars": 1})


def test_context_alignment():
    out = remi.encode_with_context(PHRASE)
    assert len(out["ids"]) == len(out["bar_pos"]) == len(out["beat"])
    strings = remi.ids_to_strings(out["ids"])
    rows = list(zip(strings, out["bar_pos"], out["beat"]))
    # third note: Position_12 in bar 0 -> beat 1
    i = strings.index("Position_12")
    assert rows[i:i + 3] == [("Position_12", 12, 1), ("Pitch_64", 12, 1), ("Duration_0.3.12", 12, 1)]
    # last note: onset 60 = bar 1, pos 12 -> beat 5
    j = strings.index("Pitch_62")
    assert rows[j][1:] == (12, 5)
    # every Bar token resets position
    assert all(p == 0 for s, p, _ in rows if s == "Bar_None")


def test_decode_stops_at_eos_and_ignores_pad():
    tokens = remi.encode(PHRASE)
    junk = tokens + remi.encode({"notes": [{"pitch": 70, "onset": 0, "dur": 3}]})
    assert remi.decode(junk) == remi.decode(tokens)
    assert remi.decode(tokens + [remi.pad_id()] * 5) == remi.decode(tokens)
