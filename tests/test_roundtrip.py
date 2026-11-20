"""Tokens -> MIDI -> tokens must give the same thing."""
from src.tokens import remi

PHRASE = {
    "notes": [
        {"pitch": 67, "onset": 0, "dur": 6},
        {"pitch": 65, "onset": 6, "dur": 6},
        {"pitch": 64, "onset": 12, "dur": 3},
        {"pitch": 62, "onset": 60, "dur": 12},  # second bar, after a rest
    ]
}


def test_encode_decode_roundtrip():
    tokens = remi.encode(PHRASE)
    assert remi.decode(tokens)["notes"] == PHRASE["notes"]
    assert remi.encode(remi.decode(tokens)) == tokens
