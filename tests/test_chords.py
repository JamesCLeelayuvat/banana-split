"""The chord parser must handle every symbol in the data."""
import sqlite3
from pathlib import Path

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


WEIMAR_DB = Path(__file__).resolve().parents[1] / "data" / "raw" / "wjazzd.db"


@pytest.mark.skipif(not WEIMAR_DB.exists(), reason="needs data/raw/wjazzd.db")
def test_parse_every_symbol_in_data():
    """Every chord symbol in Weimar must parse to one of the 9 qualities (or NC).

    TODO: add Filosax symbols once load_filosax exists.
    """
    from src.data.chords import parse

    con = sqlite3.connect(WEIMAR_DB)
    symbols = [s for (s,) in con.execute("select distinct chord from beats where chord != ''")]
    con.close()
    assert symbols, "no chord symbols found in the database"

    failures = {}
    for s in symbols:
        try:
            out = parse(s)
        except ValueError as e:
            failures[s] = str(e)
            continue
        if not (out is None or (0 <= out[0] < 12 and out[1] in QUALITIES)):
            failures[s] = f"bad output {out}"
    assert not failures, f"{len(failures)} unparseable symbols: {failures}"
