"""Transposition augmentation and the training dataset."""
import random

from src.data.augment import augment, transpose_phrase
from src.data.chords import parse
from src.data.dataset import PhraseDataset, collate
from src.tokens import chord_vocab

PHRASE = {"phrase_id": "p0", "bars": 1,
          "notes": [{"pitch": 60, "onset": 0, "dur": 12}, {"pitch": 64, "onset": 12, "dur": 12}],
          "chords": ["D-7", "D-7", "G79b", "NC"]}


def test_transpose_moves_notes_and_chords():
    up = transpose_phrase(PHRASE, 2)
    assert [n["pitch"] for n in up["notes"]] == [62, 66]
    assert [parse(c) for c in up["chords"]] == [(4, "m7"), (4, "m7"), (9, "7alt"), None]
    assert PHRASE["notes"][0]["pitch"] == 60  # original untouched


def test_transpose_out_of_range_is_none():
    assert transpose_phrase(PHRASE, 70) is None
    assert len(augment(PHRASE)) == 12


def test_dataset_filters_transposes_and_drops_captions():
    long = {**PHRASE, "phrase_id": "p1",
            "notes": [{"pitch": 60, "onset": i, "dur": 1} for i in range(48)]}
    caps = {"p0": ["a", "b"], "p1": ["c"]}
    ds = PhraseDataset([PHRASE, long], max_len=40, transpose=(-5, 6), captions=caps,
                       caption_dropout=0.5, seed=1)
    assert len(ds) == 1 and ds.n_dropped == 1
    items = [ds[0] for _ in range(200)]
    roots = {chord_vocab.id_to_chord(it["chord_ids"][0])[0] for it in items}
    assert len(roots) == 12  # every key shows up
    assert {it["caption"] for it in items} == {"", "a", "b"}
    ev = PhraseDataset([PHRASE], max_len=40, captions=caps, random_caption=False)
    assert {ev[0]["caption"] for _ in range(20)} == {"a"}
    batch = collate(items[:4])
    assert batch["ids"].shape[0] == 4 and batch["chord_ids"].shape == (4, 4)
