"""Phrases -> model batches: tokenization, on-the-fly transposition, captions with dropout, collate.

Batches are what SoloModel expects (see src/model/solo_model.py):
    ids, bar_pos, beat   [B, L]  right-padded with PAD / 0 / 0
    chord_ids            [B, N]  one chord id per beat, right-padded with chord PAD (0)
    captions             list[str]
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.augment import transpose_phrase
from src.data.chords import parse
from src.tokens import chord_vocab, remi


def encode_item(phrase: dict, caption: str = "") -> dict:
    enc = remi.encode_with_context(phrase)
    enc["chord_ids"] = [chord_vocab.chord_to_id(parse(c)) for c in phrase["chords"]]
    enc["caption"] = caption
    return enc


def collate(items: list[dict]) -> dict:
    L = max(len(it["ids"]) for it in items)
    N = max(len(it["chord_ids"]) for it in items)

    def pad(key, length, value):
        return torch.tensor([it[key] + [value] * (length - len(it[key])) for it in items])

    return {
        "ids": pad("ids", L, remi.pad_id()),
        "bar_pos": pad("bar_pos", L, 0),
        "beat": pad("beat", L, 0),
        "chord_ids": pad("chord_ids", N, chord_vocab.PAD_ID),
        "captions": [it["caption"] for it in items],
    }


def make_batch(phrases: list[dict], captions: list[str] | None = None) -> dict:
    """Phrase dicts -> one batch, no augmentation (tests, generation, evaluation)."""
    captions = captions or [""] * len(phrases)
    return collate([encode_item(p, c) for p, c in zip(phrases, captions)])


def load_captions(path: str | Path) -> dict[str, list[str]]:
    """captions.jsonl (H2) -> {phrase_id: [caption, ...]}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["phrase_id"]] = row["captions"]
    return out


class PhraseDataset(Dataset):
    """Phrases for training or evaluation.

    transpose=(lo, hi) picks a random shift in [lo, hi] every time a phrase is drawn (train split
    only), so each epoch sees the data in fresh keys. captions maps phrase_id -> captions; with
    random_caption one is picked at random per draw (training), otherwise always the first
    (evaluation, so val loss is comparable across epochs). Either is replaced by "" with
    probability caption_dropout. Phrases longer
    than max_len tokens are dropped (count in .n_dropped).

    register_bins (src.captions.features.load_bins()) limits each phrase's shifts to those that
    keep its register word, so a caption saying "high register" stays true after transposition.
    Pass it whenever captions are used with transpose.
    """

    def __init__(self, phrases: list[dict], max_len: int, transpose: tuple[int, int] | None = None,
                 captions: dict[str, list[str]] | None = None, caption_dropout: float = 0.0,
                 random_caption: bool = True, seed: int = 0, register_bins: dict | None = None):
        lengths = [len(remi.encode(p)) for p in phrases]  # transposition doesn't change length
        self.phrases = [p for p, n in zip(phrases, lengths) if n <= max_len]
        self.n_dropped = len(phrases) - len(self.phrases)
        self.transpose = transpose
        self.shifts = None
        if transpose is not None:
            if register_bins is not None:
                from src.captions.features import allowed_shifts
                self.shifts = [allowed_shifts(p, register_bins, *transpose) for p in self.phrases]
            else:
                self.shifts = [list(range(transpose[0], transpose[1] + 1))] * len(self.phrases)
        self.captions = captions
        self.caption_dropout = caption_dropout
        self.random_caption = random_caption
        self.rng = random.Random(seed)
        if captions is not None:
            missing = sum(p["phrase_id"] not in captions for p in self.phrases)
            if missing:
                raise KeyError(f"{missing} phrases have no captions")

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, i: int) -> dict:
        phrase = self.phrases[i]
        if self.shifts is not None:
            shifted = transpose_phrase(phrase, self.rng.choice(self.shifts[i]))
            phrase = shifted if shifted is not None else phrase
        caption = ""
        if self.captions is not None and self.rng.random() >= self.caption_dropout:
            options = self.captions[phrase["phrase_id"]]
            caption = self.rng.choice(options) if self.random_caption else options[0]
        return encode_item(phrase, caption)
