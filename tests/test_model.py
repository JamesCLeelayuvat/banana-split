"""SoloDecoder / SoloModel: shapes, causality, padding, conditioning, and overfitting one batch."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.data.dataset import make_batch
from src.model.solo_model import SoloModel
from src.tokens import chord_vocab, remi

ROOT = Path(__file__).resolve().parents[1]
TINY = {"d_model": 64, "chord_encoder": {"layers": 1, "heads": 2},
        "decoder": {"layers": 2, "heads": 2, "ff_dim": 128}, "max_len": 256, "dropout": 0.0,
        "text_encoder": {"name": "unused"}}

PHRASES = [
    {"notes": [{"pitch": 60 + i % 7, "onset": i * 6, "dur": 6} for i in range(24)], "bars": 4,
     "chords": ["Dm7"] * 2 + ["G7"] * 2 + ["Cmaj7"] * 4 + ["A7"] * 4 + ["Dm7"] * 4},
    {"notes": [{"pitch": 70, "onset": 12, "dur": 24}, {"pitch": 67, "onset": 60, "dur": 3}],
     "bars": 4, "chords": ["Bb7"] * 16},
]


@pytest.fixture
def batch():
    return make_batch(PHRASES)


@pytest.mark.parametrize("cond", ["none", "chords"])
def test_logits_shape(batch, cond):
    model = SoloModel(TINY, cond).eval()
    out = model(batch)
    assert out.shape == (*batch["ids"].shape, len(remi.vocab()))
    assert torch.isfinite(out).all()


def test_causal(batch):
    """Changing token k leaves the logits before k unchanged."""
    model = SoloModel(TINY, "chords").eval()
    k = 10
    other = {**batch, "ids": batch["ids"].clone()}
    other["ids"][:, k] = remi.vocab()["Pitch_80"]
    with torch.no_grad():
        a, b = model(batch), model(other)
    assert torch.allclose(a[:, :k], b[:, :k], atol=1e-5)
    assert not torch.allclose(a[:, k:], b[:, k:])


def test_right_padding_does_not_change_real_tokens(batch):
    model = SoloModel(TINY, "chords").eval()
    single = make_batch(PHRASES[1:])  # the short phrase on its own, no padding
    n = single["ids"].size(1)
    with torch.no_grad():
        alone, batched = model(single), model(batch)
    assert torch.allclose(alone[0], batched[1, :n], atol=1e-5)


def test_none_ignores_chords_and_chords_uses_them(batch):
    other = {**batch, "chord_ids": torch.full_like(batch["chord_ids"], chord_vocab.chord_to_id((6, "7alt")))}
    with torch.no_grad():
        none = SoloModel(TINY, "none").eval()
        assert torch.equal(none(batch), none(other))
        chords = SoloModel(TINY, "chords").eval()
        assert not torch.allclose(chords(batch), chords(other))


def test_output_tied_to_token_embedding():
    model = SoloModel(TINY, "none")
    assert model.decoder.out.weight is model.decoder.tok.weight


def test_too_long_raises(batch):
    model = SoloModel({**TINY, "max_len": 8}, "none")
    with pytest.raises(ValueError):
        model(batch)


def test_small_config_size():
    cfg = yaml.safe_load(open(ROOT / "configs" / "model_small.yaml"))
    n = sum(p.numel() for p in SoloModel(cfg, "chords").parameters())
    assert 2e6 < n < 8e6  # design doc: ~5M trainable for the small config


def _real_phrases(k=4):
    files = sorted((ROOT / "data" / "phrases").glob("*.json"))
    if not files:
        pytest.skip("needs data/phrases")
    return json.load(open(files[0]))[:k]


def test_overfit_one_batch():
    """Design doc run 1: the pipeline is wired if loss on one batch goes near zero."""
    torch.manual_seed(0)
    batch = make_batch(_real_phrases())
    model = SoloModel(TINY, "chords")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    inp = {**batch, "ids": batch["ids"][:, :-1], "bar_pos": batch["bar_pos"][:, :-1],
           "beat": batch["beat"][:, :-1]}
    target = batch["ids"][:, 1:]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=remi.pad_id())
    for _ in range(300):
        loss = loss_fn(model(inp).flatten(0, 1), target.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
