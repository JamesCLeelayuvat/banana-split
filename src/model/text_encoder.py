"""Frozen pretrained text encoder (T5 / sentence-embedding model) + trainable projection to d_model.

captions -> HF tokenizer -> frozen encoder -> [mean-pooled vector, per-token vectors] -> Linear
-> memory [B, 1+T, d_model]. The pooled vector comes first because sentence-embedding models
(MiniLM) are trained so that the mean over tokens is the meaningful summary; the per-token
vectors let cross-attention pick out single words ("high", "sparse").

An empty caption becomes one learned null vector instead (and nothing else), so caption dropout
in training and the unconditional pass of classifier-free guidance share one embedding.

The frozen encoder is deliberately not a registered submodule: its weights stay out of
state_dict() and parameters(), so checkpoints hold only the projection and null vector and
optimizers never see it. It follows the projection's device on every call.
"""
from __future__ import annotations

import torch
from torch import nn

MAX_TOKENS = 64  # captions are one or two sentences; longer ones are truncated


def _load_encoder(name: str):
    from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(name)
    config = AutoConfig.from_pretrained(name)
    if getattr(config, "is_encoder_decoder", False):
        encoder = T5EncoderModel.from_pretrained(name)  # encoder half only (t5-*, flan-t5-*)
    else:
        encoder = AutoModel.from_pretrained(name)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return tokenizer, encoder, config.hidden_size if hasattr(config, "hidden_size") else config.d_model


class TextEncoder(nn.Module):
    def __init__(self, name: str, d_model: int):
        super().__init__()
        self.name = name
        tokenizer, encoder, hidden = _load_encoder(name)
        object.__setattr__(self, "tokenizer", tokenizer)  # not submodules: frozen, not saved
        object.__setattr__(self, "encoder", encoder)
        self.proj = nn.Linear(hidden, d_model)
        self.null = nn.Parameter(torch.randn(1, d_model) * 0.02)

    @torch.no_grad()
    def embed(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen features for non-empty captions: ([n, 1+T, hidden], valid mask [n, 1+T])."""
        device = self.proj.weight.device
        if next(self.encoder.parameters()).device != device:
            self.encoder.to(device)
        tok = self.tokenizer(captions, padding=True, truncation=True, max_length=MAX_TOKENS,
                             return_tensors="pt").to(device)
        hidden = self.encoder(input_ids=tok["input_ids"],
                              attention_mask=tok["attention_mask"]).last_hidden_state
        valid = tok["attention_mask"].bool()
        pooled = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        feats = torch.cat([pooled.unsqueeze(1), hidden], 1)
        valid = torch.cat([torch.ones_like(valid[:, :1]), valid], 1)
        return feats, valid

    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (memory [B, T, d_model], padding_mask [B, T]). Empty caption -> null embedding."""
        B, d = len(captions), self.proj.out_features
        device = self.proj.weight.device
        has_text = [i for i, c in enumerate(captions) if c.strip()]
        if not has_text:
            return self.null.expand(B, 1, d), torch.zeros(B, 1, dtype=torch.bool, device=device)

        feats, valid = self.embed([captions[i] for i in has_text])
        projected = self.proj(feats)                       # gradients flow from here on
        T = projected.size(1)
        memory = self.null.expand(B, T, d).clone()          # null in slot 0 for empty captions
        pad = torch.ones(B, T, dtype=torch.bool, device=device)
        pad[:, 0] = False
        idx = torch.tensor(has_text, device=device)
        memory = memory.index_copy(0, idx, projected)
        pad = pad.index_copy(0, idx, ~valid)
        return memory, pad

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # frozen encoder never uses dropout
        return self
