"""Frozen pretrained text encoder (T5 / sentence-embedding model) + trainable projection to d_model."""
import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, name: str, d_model: int):
        super().__init__()
        self.name = name
        self.proj = None  # nn.Linear(encoder_hidden, d_model), built once the encoder is loaded
        raise NotImplementedError

    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (memory [B, T, d_model], padding_mask [B, T]). Empty caption -> null embedding."""
        raise NotImplementedError
