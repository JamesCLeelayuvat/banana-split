"""Sampling: grammar mask, nucleus sampling, and end-to-end generation with a tiny model."""
import json
from pathlib import Path

import pytest
import torch

from src.data.dataset import make_batch
from src.generate import Grammar, _vocab, allowed_mask, load_model, nucleus, sample
from src.model.solo_model import SoloModel
from src.tokens import remi

ROOT = Path(__file__).resolve().parents[1]
TINY = {"d_model": 32, "chord_encoder": {"layers": 1, "heads": 2},
        "decoder": {"layers": 1, "heads": 2, "ff_dim": 64}, "max_len": 256, "dropout": 0.0,
        "text_encoder": {"name": "unused"}}
CHORDS = ["Dm7"] * 4 + ["G7"] * 4 + ["Cmaj7"] * 8


def check_valid(phrase, bars):
    tokens = phrase["tokens"]
    g = Grammar(bars, max_len=10_000)
    for t in tokens:
        g.step(t)  # raises if any token breaks the grammar
    assert g.done and phrase["bars"] == bars
    onsets = [n["onset"] for n in phrase["notes"]]
    assert onsets == sorted(set(onsets))  # monophonic, in order
    assert all(0 <= o < bars * remi.TICKS_PER_BAR for o in onsets)


def test_real_phrases_follow_the_grammar():
    files = sorted((ROOT / "data" / "phrases").glob("*.json"))[:20]
    if not files:
        pytest.skip("needs data/phrases")
    checked = 0
    for f in files:
        for p in json.loads(f.read_text()):
            onsets = [n["onset"] for n in p["notes"]]
            if len(onsets) != len(set(onsets)):
                continue  # the rare same-onset (grace note) pairs aren't monophonic
            g = Grammar(p["bars"], max_len=10_000)
            for t in remi.encode(p):
                g.step(t)
            assert g.done
            checked += 1
    assert checked > 100


def test_grammar_basics():
    V = _vocab()
    assert allowed_mask([], 4, 256).nonzero().flatten().tolist() == [V.bos]
    after_bar = allowed_mask([V.bos, V.bar], 4, 256)
    assert after_bar[V.bar] and not after_bar[V.eos] and after_bar[V.position[0]]
    after_pos = allowed_mask([V.bos, V.bar, V.position[12]], 4, 256)
    assert after_pos[V.pitch[60]] and not after_pos[V.bar]
    # after a note at position 12, earlier positions are gone
    after_note = allowed_mask([V.bos, V.bar, V.position[12], V.pitch[60], V.duration[5]], 4, 256)
    assert not after_note[V.position[12]] and after_note[V.position[13]]
    # all four bars seen: EOS, not another Bar
    last_bar = allowed_mask([V.bos] + [V.bar] * 4, 4, 256)
    assert last_bar[V.eos] and not last_bar[V.bar]


def test_grammar_respects_length_budget():
    V = _vocab()
    # 4 bars, max_len 8: BOS Bar(1) leaves 6; a note needs 3 + 3 more Bars + EOS = 7 -> no room
    assert not allowed_mask([V.bos, V.bar], 4, 8)[V.position[0]]
    assert allowed_mask([V.bos, V.bar], 4, 9)[V.position[0]]


def test_pitch_range():
    V = _vocab()
    m = allowed_mask([V.bos, V.bar, V.position[0]], 4, 256, pitch_range=(60, 72))
    assert m[V.pitch[60]] and m[V.pitch[72]] and not m[V.pitch[59]] and not m[V.pitch[73]]


def test_nucleus():
    logits = torch.tensor([[2.0, 1.0, 0.0, float("-inf")]] * 200)
    assert (nucleus(logits, temperature=0) == 0).all()
    assert (nucleus(logits, top_p=0.01) == 0).all()         # only the top token survives
    draws = nucleus(logits, temperature=1.0, top_p=1.0, generator=torch.Generator().manual_seed(0))
    assert set(draws.tolist()) <= {0, 1, 2}                  # never the masked token
    assert len(set(draws.tolist())) == 3


@pytest.mark.parametrize("cond", ["none", "chords"])
def test_sample_produces_valid_phrases(cond):
    torch.manual_seed(0)
    model = SoloModel(TINY, cond)
    phrases = sample(model, CHORDS, n=3, temperature=1.0, top_p=0.95, seed=0)
    assert len(phrases) == 3
    for p in phrases:
        check_valid(p, bars=4)
        assert p["chords"] == CHORDS


def test_sample_fits_short_max_len():
    torch.manual_seed(0)
    model = SoloModel({**TINY, "max_len": 24}, "chords")
    for p in sample(model, CHORDS, n=4, temperature=1.5, top_p=1.0, seed=1):
        check_valid(p, bars=4)
        assert len(p["tokens"]) <= 24


def test_sample_is_reproducible_with_seed():
    model = SoloModel(TINY, "chords")
    a = sample(model, CHORDS, n=2, seed=3)
    b = sample(model, CHORDS, n=2, seed=3)
    assert [p["tokens"] for p in a] == [p["tokens"] for p in b]


def test_wrong_chord_count_raises():
    with pytest.raises(ValueError):
        sample(SoloModel(TINY, "chords"), CHORDS[:-1])


def test_guidance_combines_cond_and_uncond(monkeypatch):
    """Guided sampling runs n conditional + n unconditional rows; guidance=1 or an empty text
    runs only the n conditional rows."""
    model = SoloModel(TINY, "chords")
    model.conditioning = "full"  # pretend; encode() is stubbed so no text encoder is needed
    seen_rows = []

    def fake_encode(batch):
        seen_rows.append(len(batch["captions"]))
        return None, None

    monkeypatch.setattr(model, "encode", fake_encode)
    monkeypatch.setattr(model, "current_chords", lambda b: torch.zeros_like(b["beat"]))
    sample(model, CHORDS, text="dense high line", n=2, guidance=2.0, seed=0)
    sample(model, CHORDS, text="dense high line", n=2, guidance=1.0, seed=0)
    sample(model, CHORDS, text="", n=2, guidance=2.0, seed=0)
    assert seen_rows == [4, 2, 2]


def test_greedy_reproduces_an_overfit_phrase():
    """Generation must feed the model the same context (bar_pos, beat, chords) as training:
    a model overfit on one phrase, sampled greedily over its chords, gives it back exactly."""
    torch.manual_seed(0)
    phrase = {"notes": [{"pitch": 60 + (i * 5) % 12, "onset": i * 8, "dur": 4} for i in range(20)],
              "bars": 4, "chords": CHORDS}
    model = SoloModel({**TINY, "d_model": 64, "decoder": {"layers": 2, "heads": 2, "ff_dim": 128}},
                      "chords")
    batch = make_batch([phrase])
    inp = {**batch, "ids": batch["ids"][:, :-1], "bar_pos": batch["bar_pos"][:, :-1],
           "beat": batch["beat"][:, :-1]}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        loss = loss_fn(model(inp).flatten(0, 1), batch["ids"][:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = sample(model, CHORDS, n=1, temperature=0)[0]
    assert out["tokens"] == remi.encode(phrase)
    assert out["notes"] == phrase["notes"]


def test_load_model_roundtrip(tmp_path):
    model = SoloModel(TINY, "chords")
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "model_cfg": TINY,
                "train_cfg": {"conditioning": "chords"}}, path)
    loaded = load_model(path)
    assert loaded.conditioning == "chords"
    a = sample(model, CHORDS, n=1, seed=0)[0]["tokens"]
    assert sample(loaded, CHORDS, n=1, seed=0)[0]["tokens"] == a
