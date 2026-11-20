"""The chord parser must handle every symbol in the data."""
import pytest

from src.data.chords import QUALITIES, ROOTS
from src.tokens.chord_vocab import VOCAB_SIZE, chord_to_id, id_to_chord

SAMPLE_SYMBOLS = ["Cmaj7", "C6", "F7", "Dm7", "Ebm6", "Bm7b5", "Bo7", "G7alt", "A7sus4", "NC"]


def test_vocab_roundtrip():
    for r in range(len(ROOTS)):
        for q in QUALITIES:
            assert id_to_chord(chord_to_id((r, q))) == (r, q)
    assert VOCAB_SIZE == 2 + 12 * 9


@pytest.mark.parametrize("symbol", SAMPLE_SYMBOLS)
def test_parse_sample_symbols(symbol):
    from src.data.chords import parse
    out = parse(symbol)
    assert out is None or (0 <= out[0] < 12 and out[1] in QUALITIES)


def test_parse_every_symbol_in_data():
    """Fill in once loaders exist: collect all symbols from Weimar (+ Filosax) and parse each."""
    pytest.skip("needs data/raw")
