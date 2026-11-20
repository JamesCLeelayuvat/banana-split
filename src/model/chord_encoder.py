"""Chord encoder: per-beat chord tokens -> contextual embeddings (small TransformerEncoder)."""
import torch
from torch import nn


class ChordEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, layers: int, heads: int, dropout: float = 0.1, max_beats: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_beats, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, chord_ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(chord_ids.size(1), device=chord_ids.device)
        x = self.embed(chord_ids) + self.pos(pos)
        return self.encoder(x, src_key_padding_mask=chord_ids.eq(0))
