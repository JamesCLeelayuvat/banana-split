"""Full model: text encoder + chord encoder -> shared memory (with segment embedding) -> decoder.

Batch format (what the training collate and generation both produce):
    ids       [B, L] long   solo token ids (decoder input; PAD on the right)
    bar_pos   [B, L] long   position in the bar of each token, from remi.token_context
    beat      [B, L] long   beat index of each token in the phrase, from remi.token_context
    chord_ids [B, N] long   one chord id per beat (src.tokens.chord_vocab; PAD=0 on the right)
    captions  list[str]     B captions, "" for none (only read when conditioning='full')

The decoder's current-chord embedding is chord_ids at each token's beat (clamped to the last
beat). The unconditioned baseline ('none') sees no chords at all: no memory, and NO_CHORD as
every token's current chord, so it is a true floor. chord_ids may then be omitted.
"""
from __future__ import annotations

import torch
from torch import nn

from src.model.chord_encoder import ChordEncoder
from src.model.decoder import SoloDecoder
from src.tokens import chord_vocab, remi

CONDITIONINGS = ("none", "chords", "full")
CHORD_SEG, TEXT_SEG = 0, 1


class SoloModel(nn.Module):
    def __init__(self, cfg: dict, conditioning: str = "full"):
        """cfg: a configs/model_*.yaml dict. conditioning: 'none' | 'chords' | 'full'."""
        super().__init__()
        if conditioning not in CONDITIONINGS:
            raise ValueError(f"conditioning must be one of {CONDITIONINGS}, got {conditioning!r}")
        self.conditioning = conditioning
        self.pad_id = remi.pad_id()
        d, dropout = cfg["d_model"], cfg.get("dropout", 0.1)
        self.decoder = SoloDecoder(
            vocab_size=len(remi.vocab()),
            chord_vocab_size=chord_vocab.VOCAB_SIZE,
            d_model=d,
            layers=cfg["decoder"]["layers"],
            heads=cfg["decoder"]["heads"],
            ff_dim=cfg["decoder"]["ff_dim"],
            max_len=cfg["max_len"],
            dropout=dropout,
            ticks_per_bar=remi.TICKS_PER_BAR,
        )
        self.chord_encoder = self.text_encoder = self.segment = None
        if conditioning in ("chords", "full"):
            ce = cfg["chord_encoder"]
            self.chord_encoder = ChordEncoder(chord_vocab.VOCAB_SIZE, d, ce["layers"], ce["heads"],
                                              dropout)
        if conditioning == "full":
            from src.model.text_encoder import TextEncoder
            self.text_encoder = TextEncoder(cfg["text_encoder"]["name"], d)
            self.segment = nn.Embedding(2, d)

    def encode(self, batch: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Conditioning memory [B, T, d] and its padding mask [B, T], or (None, None).
        Generation calls this once and reuses it for every sampling step."""
        if self.conditioning == "none":
            return None, None
        chord_ids = batch["chord_ids"]
        mem, mask = self.chord_encoder(chord_ids), chord_ids.eq(chord_vocab.PAD_ID)
        if self.conditioning == "full":
            text, text_mask = self.text_encoder(batch["captions"])
            mem = torch.cat([mem + self.segment.weight[CHORD_SEG], text + self.segment.weight[TEXT_SEG]], 1)
            mask = torch.cat([mask, text_mask], 1)
        return mem, mask

    def current_chords(self, batch: dict) -> torch.Tensor:
        """Chord id on each token's beat, [B, L]."""
        beat = batch["beat"]
        if self.conditioning == "none":
            return torch.full_like(beat, chord_vocab.NO_CHORD_ID)
        chord_ids = batch["chord_ids"]
        return chord_ids.gather(1, beat.clamp(0, chord_ids.size(1) - 1))

    def forward(self, batch: dict, memory: tuple | None = None) -> torch.Tensor:
        """Return logits [B, L, vocab]. `memory` is an optional precomputed encode(batch)."""
        mem, mem_mask = self.encode(batch) if memory is None else memory
        ids = batch["ids"]
        return self.decoder(ids, batch["bar_pos"], self.current_chords(batch), mem, mem_mask,
                            pad_mask=ids.eq(self.pad_id))
