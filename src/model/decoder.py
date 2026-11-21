"""Decoder: token + bar-position + current-chord embeddings -> nn.TransformerDecoder -> logits.

Every decoder input token gets four embeddings summed: the token itself, its position in the bar
(0..47 ticks), the chord on the beat it sits at, and its index in the sequence. Each layer does
causal self-attention over the notes so far, then cross-attention into the conditioning memory
(chords and/or text, built by SoloModel).

Cross-attention always reads a memory that starts with one learned "null" vector. With
memory=None (the unconditioned baseline) that vector is the whole memory, so all three models
share one architecture and checkpoint format. When memory is given, the null vector works as an
attention sink: a layer that has nothing to look up can attend to it, and a memory whose
positions are all padding never produces NaNs.

Pre-LayerNorm layers (more stable on small data) and output weights tied to the token embedding.
"""
from __future__ import annotations

import torch
from torch import nn


class SoloDecoder(nn.Module):
    def __init__(self, vocab_size: int, chord_vocab_size: int, d_model: int, layers: int, heads: int,
                 ff_dim: int, max_len: int, dropout: float = 0.1, ticks_per_bar: int = 48):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.bar_pos = nn.Embedding(ticks_per_bar, d_model)
        self.cur_chord = nn.Embedding(chord_vocab_size, d_model)
        self.seq_pos = nn.Embedding(max_len, d_model)
        self.null_memory = nn.Parameter(torch.zeros(1, 1, d_model))
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff_dim, dropout, batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.tok.weight  # weight tying
        for emb in (self.tok, self.bar_pos, self.cur_chord, self.seq_pos):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.null_memory, std=0.02)

    def _memory(self, memory: torch.Tensor | None, memory_mask: torch.Tensor | None, batch: int):
        """Prepend the null vector; return (memory [B, 1+T, d], key padding mask [B, 1+T])."""
        null = self.null_memory.expand(batch, -1, -1)
        if memory is None:
            return null, torch.zeros(batch, 1, dtype=torch.bool, device=null.device)
        if memory_mask is None:
            memory_mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
        keep = torch.zeros(batch, 1, dtype=torch.bool, device=memory.device)
        return torch.cat([null, memory], 1), torch.cat([keep, memory_mask], 1)

    def forward(self, tokens, bar_pos, cur_chord, memory=None, memory_mask=None, pad_mask=None):
        """Logits [B, L, vocab] for the next token at every position.

        tokens, bar_pos, cur_chord: [B, L] long (bar_pos in ticks 0..47, cur_chord a chord id).
        memory: [B, T, d_model] conditioning, or None -> unconditioned baseline (decoder-only).
        memory_mask: [B, T] bool, True where memory is padding.
        pad_mask: [B, L] bool, True where tokens are padding (optional; causal masking already
            stops real tokens from seeing right-padding, this only skips wasted attention).
        """
        B, L = tokens.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len {self.max_len}")
        pos = torch.arange(L, device=tokens.device)
        x = self.tok(tokens) + self.bar_pos(bar_pos) + self.cur_chord(cur_chord) + self.seq_pos(pos)
        mem, mem_mask = self._memory(memory, memory_mask, B)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device,
                                                                dtype=torch.bool)
        h = self.decoder(self.drop(x), mem, tgt_mask=causal, tgt_is_causal=True,
                         tgt_key_padding_mask=pad_mask, memory_key_padding_mask=mem_mask)
        return self.out(h)
