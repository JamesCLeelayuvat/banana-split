"""Full model: text encoder + chord encoder -> shared memory (with segment embedding) -> decoder."""
from torch import nn


class SoloModel(nn.Module):
    def __init__(self, cfg: dict, conditioning: str = "full"):
        """conditioning: 'none' | 'chords' | 'full'."""
        super().__init__()
        self.conditioning = conditioning
        raise NotImplementedError

    def forward(self, batch: dict):
        """Return logits [B, L, vocab]."""
        raise NotImplementedError
