"""Frozen text encoder + projection, and the full model using it. Downloads MiniLM on first run."""
import pytest
import torch

from src.data.dataset import make_batch
from src.model.solo_model import SoloModel
from src.model.text_encoder import TextEncoder

NAME = "sentence-transformers/all-MiniLM-L6-v2"
D = 64


@pytest.fixture(scope="module")
def enc():
    try:
        return TextEncoder(NAME, D)
    except OSError as e:  # offline and not cached
        pytest.skip(f"can't load {NAME}: {e}")


def test_shapes_and_padding(enc):
    mem, pad = enc(["a dense descending bebop line", "sparse", ""])
    assert mem.shape[0] == 3 and mem.shape[2] == D and pad.shape == mem.shape[:2]
    assert not pad[:, 0].any()                    # slot 0 (pooled / null) is always real
    assert pad[1].sum() > pad[0].sum()            # shorter caption -> more padding
    assert pad[2, 1:].all()                       # empty caption: only the null vector
    assert torch.allclose(mem[2, 0], enc.null[0])


def test_all_empty_is_null_only(enc):
    mem, pad = enc(["", "  "])
    assert mem.shape == (2, 1, D) and not pad.any()


def test_wording_changes_memory(enc):
    a, _ = enc(["a line in the high register"])
    b, _ = enc(["a line in the low register"])
    assert not torch.allclose(a, b)


def test_frozen_and_not_saved(enc):
    keys = set(enc.state_dict())
    assert keys == {"proj.weight", "proj.bias", "null"}
    assert all(p.requires_grad for p in enc.parameters())          # only trainable ones listed
    assert not any(p.requires_grad for p in enc.encoder.parameters())
    enc.train()
    assert not enc.encoder.training


def test_gradients_reach_projection_only(enc):
    enc.zero_grad()
    mem, pad = enc(["moderate eighth-note lines", ""])
    mem[~pad].sum().backward()
    assert enc.proj.weight.grad is not None and enc.null.grad is not None
    assert all(p.grad is None for p in enc.encoder.parameters())


def test_full_model_forward_and_guidance_signal(enc):
    cfg = {"d_model": D, "chord_encoder": {"layers": 1, "heads": 2},
           "decoder": {"layers": 2, "heads": 2, "ff_dim": 128}, "max_len": 256, "dropout": 0.0,
           "text_encoder": {"name": NAME}}
    model = SoloModel(cfg, "full").eval()
    phrase = {"notes": [{"pitch": 64, "onset": i * 6, "dur": 6} for i in range(8)], "bars": 1,
              "chords": ["Dm7", "Dm7", "G7", "G7"]}
    with torch.no_grad():
        cond = model(make_batch([phrase], ["a high, dense bebop line"]))
        uncond = model(make_batch([phrase], [""]))
    assert cond.shape == uncond.shape and torch.isfinite(cond).all()
    assert not torch.allclose(cond, uncond)       # classifier-free guidance has something to push
    assert not any(k.startswith("text_encoder.encoder") for k in model.state_dict())
