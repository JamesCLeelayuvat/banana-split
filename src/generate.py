"""Sampling: nucleus (top-p) sampling with temperature, a grammar mask, and classifier-free
guidance on text.

Usage: python -m src.generate --ckpt outputs/checkpoints/<run>/best.pt --chords "Dm7 G7 | Cmaj7 | %" \
           --tempo 180 --text "a dense descending bebop line" --n 4 --out outputs/samples/

Generation runs n samples in parallel. Each step re-runs the decoder on the whole prefix (no KV
cache: phrases are ~150 tokens, so this is fast enough) and picks the next token:

1. Logits for the last position. With text guidance (conditioning='full', a non-empty --text,
   guidance != 1), the batch also holds n copies with an empty caption, and the logits become
   uncond + guidance * (cond - uncond). Training's caption dropout is what makes the empty
   caption a valid unconditional model.
2. The grammar mask (Grammar) removes every token that would make the sequence invalid.
3. Nucleus sampling: divide by temperature, keep the smallest set of tokens whose probability
   adds up to top_p, sample from it. temperature=0 means greedy.

The grammar (REMI without velocity): BOS Bar, then per note Position Pitch Duration, with
Positions strictly increasing within a bar (so one note per onset: lines are monophonic),
exactly `bars` Bar tokens, then EOS. Positions are refused once the remaining length budget
(max_len) is only enough to close the phrase, so every sample ends cleanly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.chords import parse
from src.model.solo_model import SoloModel
from src.tokens import chord_vocab, remi

ROOT = Path(__file__).resolve().parents[1]


def parse_changes(s: str, beats_per_bar: int = 4) -> list[str]:
    """'Dm7 G7 | Cmaj7' -> one chord symbol per beat.

    Bars are separated by '|'. The chords in a bar share its beats evenly, with any remainder
    going to the earlier chords (3 chords in 4/4 -> 2+1+1). A bar holding only '%' repeats the
    previous bar; an empty bar holds the previous chord.
    """
    beats: list[str] = []
    prev_bar = ["NC"]
    for bar in s.strip().strip("|").split("|"):
        syms = bar.split()
        if syms == ["%"]:
            syms = prev_bar
        elif not syms:
            syms = [beats[-1] if beats else "NC"]
        if len(syms) > beats_per_bar:
            raise ValueError(f"bar {bar!r} has more chords than beats")
        base, extra = divmod(beats_per_bar, len(syms))
        for i, sym in enumerate(syms):
            beats += [sym] * (base + (1 if i < extra else 0))
        prev_bar = syms
    return beats


# ---------------------------------------------------------------- grammar

class _Vocab:
    """Token ids grouped by kind, built once."""

    def __init__(self):
        v = remi.vocab()
        self.size = len(v)
        self.pad, self.bos, self.eos, self.bar = v[remi.PAD], v[remi.BOS], v[remi.EOS], v[remi.BAR]
        self.position = {int(n.split("_")[1]): i for n, i in v.items() if n.startswith("Position_")}
        self.pitch = {int(n.split("_")[1]): i for n, i in v.items() if n.startswith("Pitch_")}
        self.duration = [i for n, i in v.items() if n.startswith("Duration_")]
        self.kind = {}
        for n, i in v.items():
            self.kind[i] = n.split("_")[0]  # PAD BOS EOS Bar Position Pitch Duration
        self.pos_of = {i: p for p, i in self.position.items()}


_VOCAB: _Vocab | None = None


def _vocab() -> _Vocab:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _Vocab()
    return _VOCAB


class Grammar:
    """Which tokens may come next in one sequence. Feed it each token with step()."""

    def __init__(self, bars: int, max_len: int, pitch_range: tuple[int, int] | None = None):
        V = _vocab()
        self.bars, self.max_len = bars, max_len
        self.last, self.length = None, 0     # kind of the last token, tokens so far
        self.bars_seen, self.pos = 0, -1     # Bar tokens so far, last Position in this bar
        lo, hi = pitch_range or (remi.PITCH_MIN, remi.PITCH_MAX)
        self._pitch_mask = self._mask([i for p, i in V.pitch.items() if lo <= p <= hi])
        self._dur_mask = self._mask(V.duration)

    @staticmethod
    def _mask(ids) -> torch.Tensor:
        m = torch.zeros(_vocab().size, dtype=torch.bool)
        m[list(ids)] = True
        return m

    @property
    def done(self) -> bool:
        return self.last in ("EOS", "PAD")

    def allowed(self) -> torch.Tensor:
        """Bool mask [vocab] of tokens allowed next."""
        V = _vocab()
        if self.last is None:
            return self._mask([V.bos])
        if self.done:
            return self._mask([V.pad])
        if self.last == "BOS":
            return self._mask([V.bar])
        if self.last == "Position":
            return self._pitch_mask
        if self.last == "Pitch":
            return self._dur_mask
        # after Bar or Duration: a later Position in this bar, the next Bar, or EOS
        ids = [V.bar] if self.bars_seen < self.bars else [V.eos]
        # a new note costs Position Pitch Duration, then the remaining Bars and EOS must fit
        closing = 3 + (self.bars - self.bars_seen) + 1
        if self.bars_seen >= 1 and self.length + closing <= self.max_len:
            ids += [i for p, i in V.position.items() if p > self.pos]
        return self._mask(ids)

    def step(self, token: int) -> None:
        V = _vocab()
        if not self.allowed()[token]:
            raise ValueError(f"token {remi.ids_to_strings([token])[0]} not allowed here")
        kind = V.kind[token]
        if kind == "Bar":
            self.bars_seen, self.pos = self.bars_seen + 1, -1
        elif kind == "Position":
            self.pos = V.pos_of[token]
        self.last, self.length = kind, self.length + 1


def allowed_mask(prev_tokens: list[int], bars: int, max_len: int,
                 pitch_range: tuple[int, int] | None = None) -> torch.Tensor:
    """Bool mask [vocab] of tokens allowed after prev_tokens (which must start with BOS)."""
    g = Grammar(bars, max_len, pitch_range)
    for t in prev_tokens:
        g.step(t)
    return g.allowed()


# ---------------------------------------------------------------- sampling

def nucleus(logits: torch.Tensor, temperature: float = 0.9, top_p: float = 0.9,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one token per row of logits [B, vocab] (masked tokens at -inf). temperature=0
    is greedy."""
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, -1)
    sorted_p, idx = probs.sort(-1, descending=True)
    # drop a token if the tokens ranked above it already reach top_p (the top one always stays)
    sorted_p = sorted_p.masked_fill(sorted_p.cumsum(-1) - sorted_p >= top_p, 0.0)
    choice = torch.multinomial(sorted_p, 1, generator=generator)
    return idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def sample(model: SoloModel, chords: list[str], text: str = "", bars: int | None = None,
           n: int = 1, temperature: float = 0.9, top_p: float = 0.9, guidance: float = 2.0,
           seed: int | None = None, pitch_range: tuple[int, int] | None = None) -> list[dict]:
    """Generate n phrases over `chords` (one symbol per beat, bars * 4 of them).

    Returns phrase dicts {"notes", "bars", "chords", "text", "tokens"}, notes in ticks as in
    data/phrases. `text` is ignored unless the model was trained with conditioning='full'.
    """
    model.eval()
    device = next(model.parameters()).device
    beats_per_bar = remi.BEATS_PER_BAR
    bars = bars or len(chords) // beats_per_bar
    if len(chords) != bars * beats_per_bar:
        raise ValueError(f"{len(chords)} chords for {bars} bars: need one per beat")
    max_len = model.decoder.max_len
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    use_text = model.conditioning == "full" and bool(text.strip())
    guided = use_text and guidance != 1.0
    rows = 2 * n if guided else n
    chord_ids = torch.tensor([[chord_vocab.chord_to_id(parse(c)) for c in chords]] * rows,
                             device=device)
    captions = [text if use_text else ""] * n + ([""] * n if guided else [])
    static = {"chord_ids": chord_ids, "captions": captions}
    memory = model.encode(static)

    V = _vocab()
    grammars = [Grammar(bars, max_len, pitch_range) for _ in range(n)]
    seqs = [[V.bos] for _ in range(n)]
    for g in grammars:
        g.step(V.bos)
    while not all(g.done for g in grammars) and len(seqs[0]) < max_len:
        ids = torch.tensor(seqs + (seqs if guided else []), device=device)
        ctx = [remi.token_context(s) for s in seqs]
        bar_pos = torch.tensor([c[0] for c in ctx] * (2 if guided else 1), device=device)
        beat = torch.tensor([c[1] for c in ctx] * (2 if guided else 1), device=device)
        logits = model({**static, "ids": ids, "bar_pos": bar_pos, "beat": beat},
                       memory=memory)[:, -1].float()
        if guided:
            cond, uncond = logits[:n], logits[n:]
            logits = uncond + guidance * (cond - uncond)
        mask = torch.stack([g.allowed() for g in grammars]).to(device)
        logits = logits.masked_fill(~mask, float("-inf"))
        nxt = nucleus(logits, temperature, top_p, generator).tolist()
        for s, g, t in zip(seqs, grammars, nxt):
            g.step(t)
            s.append(t)

    out = []
    for s in seqs:
        phrase = remi.decode(s)
        out.append({**phrase, "chords": list(chords), "text": text if use_text else "",
                    "tokens": s[:s.index(V.eos) + 1] if V.eos in s else s})
    return out


# ---------------------------------------------------------------- checkpoints / CLI

def load_model(ckpt_path: str | Path, device: str | torch.device = "cpu") -> SoloModel:
    """Rebuild a SoloModel from a train.py checkpoint (best.pt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SoloModel(ckpt["model_cfg"], ckpt["train_cfg"]["conditioning"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chords", required=True, help='changes, e.g. "Dm7 G7 | Cmaj7 | %%"')
    ap.add_argument("--tempo", type=int, default=160, help="written into the MIDI files")
    ap.add_argument("--text", default="")
    ap.add_argument("--bars", type=int, help="repeat or cut the changes to this many bars")
    ap.add_argument("--n", type=int, default=4, help="samples to generate")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="outputs/samples")
    args = ap.parse_args()

    chords = parse_changes(args.chords)
    if len(chords) % remi.BEATS_PER_BAR:
        raise SystemExit("changes must fill whole bars")
    if args.bars:
        need = args.bars * remi.BEATS_PER_BAR
        chords = (chords * (need // len(chords) + 1))[:need]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    if args.text and model.conditioning != "full":
        print(f"note: this is a '{model.conditioning}' model; --text is ignored")
    phrases = sample(model, chords, args.text, n=args.n, temperature=args.temperature,
                     top_p=args.top_p, guidance=args.guidance, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(phrases):
        remi.to_midi(p, out / f"sample_{i:03d}.mid", tempo=args.tempo)
    # A list of phrase dicts (data/phrases format), so src.render can read it with --index and
    # play each sample over the exact per-beat chords it was generated on.
    settings = {k: v for k, v in vars(args).items() if k != "out"}
    records = [{"phrase_id": f"sample_{i:03d}",
                **{k: v for k, v in p.items() if k != "tokens"},
                "meta": {**settings, "tempo": args.tempo}}
               for i, p in enumerate(phrases)]
    (out / "samples.json").write_text(json.dumps(records, indent=1))
    print(f"{len(phrases)} samples -> {out} (notes per sample: "
          f"{[len(p['notes']) for p in phrases]})")
    print(f"listen: python -m src.render {out / 'samples.json'} --index 0 --out outputs/audio/")


if __name__ == "__main__":
    main()
