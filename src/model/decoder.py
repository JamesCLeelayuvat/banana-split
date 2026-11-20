"""Decoder: token + bar-position + current-chord embeddings -> nn.TransformerDecoder -> logits."""
import torch
from torch import nn


class SoloDecoder(nn.Module):
    def __init__(self, vocab_size: int, chord_vocab_size: int, d_model: int, layers: int, heads: int,
                 ff_dim: int, max_len: int, dropout: float = 0.1, ticks_per_bar: int = 48):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.bar_pos = nn.Embedding(ticks_per_bar, d_model)
        self.cur_chord = nn.Embedding(chord_vocab_size, d_model)
        self.seq_pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, bar_pos, cur_chord, memory=None, memory_mask=None):
        """memory=None -> unconditioned baseline (decoder-only)."""
        raise NotImplementedError
