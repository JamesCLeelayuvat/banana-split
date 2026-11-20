"""Quantize onsets, cut 4-bar phrases (2-bar stride), split by solo, write data/phrases/*.json (H1).

Phrase JSON:
{
  "phrase_id": str, "solo_id": str,
  "start_bar": int,                                        # bar in the solo where it starts
  "bars": int,                                             # phrase length in bars (4)
  "notes": [{"pitch": int, "onset": int, "dur": int}],   # ticks, 12 per beat, from phrase start
  "chords": [str, ...],                                    # one per beat (bars * 4 entries)
  "meta": {"performer", "instrument", "style", "tempo", "key", ...}
}

Files: one JSON per solo, data/phrases/<solo_id>.json, holding the list of that solo's phrases.
Splits: data/splits/{train,val,test}.txt, one solo_id per line. Use load_phrases(split) to read.

Run:  python -m src.data.phrases
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT / "configs" / "data.yaml"
SPLITS = ("train", "val", "test")


def _config() -> dict:
    with open(DATA_CONFIG) as f:
        return yaml.safe_load(f)


def quantize(solo: dict, ticks_per_beat: int = 12) -> dict:
    """Solo with onsets/durations in beats (floats) -> same solo with integer ticks.

    Onset and note end are each rounded to the nearest tick, and dur = end - onset (at least 1
    tick). If rounding puts two notes of the same pitch on the same tick, the longer one is kept.
    """
    best: dict[tuple[int, int], dict] = {}
    for n in solo["notes"]:
        on = round(n["onset"] * ticks_per_beat)
        end = round((n["onset"] + n["dur"]) * ticks_per_beat)
        q = {"pitch": int(n["pitch"]), "onset": on, "dur": max(1, end - on)}
        key = (q["onset"], q["pitch"])
        if key not in best or q["dur"] > best[key]["dur"]:
            best[key] = q
    notes = sorted(best.values(), key=lambda n: (n["onset"], n["pitch"]))
    return {**solo, "notes": notes, "ticks_per_beat": ticks_per_beat}


def _rest_fraction(notes: list[dict], length: int) -> float:
    """Share of the phrase's ticks not covered by any sounding note."""
    covered, cur_end = 0, 0
    for n in sorted(notes, key=lambda n: n["onset"]):
        start, end = max(n["onset"], cur_end), min(n["onset"] + n["dur"], length)
        if end > start:
            covered += end - start
        cur_end = max(cur_end, end)
    return 1 - covered / length


def cut_phrases(
    solo: dict,
    bars: int = 4,
    stride: int = 2,
    max_rest: float = 0.6,
    beats_per_bar: int = 4,
    max_dur_bars: int = 2,
) -> list[dict]:
    """Quantized solo -> overlapping phrases of `bars` bars, starting every `stride` bars.

    A phrase holds every note whose onset falls inside its window, with onsets shifted to the
    phrase start. Durations are clipped at the phrase end and at `max_dur_bars` (the tokenizer's
    longest duration). Only full windows are kept; phrases that are empty or more than
    `max_rest` rest are dropped.
    """
    tpb = solo["ticks_per_beat"]
    bar_ticks = tpb * beats_per_bar
    length = bars * bar_ticks
    max_dur = max_dur_bars * bar_ticks
    n_beats = len(solo["chords"])
    n_bars = n_beats // beats_per_bar
    notes = solo["notes"]

    phrases = []
    for start_bar in range(0, n_bars - bars + 1, stride):
        t0 = start_bar * bar_ticks
        window = []
        for n in notes:
            on = n["onset"] - t0
            if on < 0:
                continue
            if on >= length:
                break  # notes are sorted by onset
            dur = min(n["dur"], length - on, max_dur)
            window.append({"pitch": n["pitch"], "onset": on, "dur": dur})
        if not window or _rest_fraction(window, length) > max_rest:
            continue
        b0 = start_bar * beats_per_bar
        phrases.append({
            "phrase_id": f"{solo['solo_id']}_b{start_bar:03d}",
            "solo_id": solo["solo_id"],
            "start_bar": start_bar,
            "bars": bars,
            "notes": window,
            "chords": solo["chords"][b0 : b0 + bars * beats_per_bar],
            "meta": solo["meta"],
        })
    return phrases


def split_by_solo(
    solo_ids: list[str],
    ratios=(0.8, 0.1, 0.1),
    seed: int = 0,
    heldout: set[str] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """Write data/splits/{train,val,test}.txt.

    Solos are shuffled with `seed` and divided by `ratios`. Solo ids in `heldout` (whole tunes
    kept out as an unseen-changes test) always go to test, on top of its share.
    """
    heldout = set(heldout or ())
    rest = sorted(s for s in solo_ids if s not in heldout)
    random.Random(seed).shuffle(rest)
    n_train = round(ratios[0] * len(rest))
    n_val = round(ratios[1] * len(rest))
    splits = {
        "train": sorted(rest[:n_train]),
        "val": sorted(rest[n_train : n_train + n_val]),
        "test": sorted(rest[n_train + n_val :] + list(heldout & set(solo_ids))),
    }
    out_dir = Path(out_dir) if out_dir else ROOT / _config()["paths"]["splits"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out_dir / f"{name}.txt").write_text("".join(f"{s}\n" for s in ids))
    return splits


def load_phrases(split: str | None = None) -> list[dict]:
    """Read phrases back from data/phrases/, for one split or (split=None) all of them."""
    paths = _config()["paths"]
    phrase_dir = ROOT / paths["phrases"]
    if split is None:
        files = sorted(phrase_dir.glob("*.json"))
    else:
        ids = (ROOT / paths["splits"] / f"{split}.txt").read_text().split()
        files = [phrase_dir / f"{s}.json" for s in ids]
    out = []
    for f in files:
        out.extend(json.loads(f.read_text()))
    return out


def build(verbose: bool = True) -> dict[str, list[str]]:
    """Load every solo, cut phrases, write data/phrases/*.json and the split files."""
    from src.data import load_weimar

    cfg = _config()
    solos = load_weimar.load_solos(verbose=verbose)
    try:
        from src.data import load_filosax
        solos += load_filosax.load_solos()
    except (NotImplementedError, FileNotFoundError):
        if verbose:
            print("Filosax not available, using Weimar only")

    phrase_dir = ROOT / cfg["paths"]["phrases"]
    phrase_dir.mkdir(parents=True, exist_ok=True)
    kept, n_phrases = [], 0
    for solo in solos:
        q = quantize(solo, cfg["ticks_per_beat"])
        phrases = cut_phrases(
            q,
            bars=cfg["phrase_bars"],
            stride=cfg["stride_bars"],
            max_rest=cfg["max_rest_fraction"],
            beats_per_bar=cfg["beats_per_bar"],
        )
        if not phrases:
            continue
        (phrase_dir / f"{solo['solo_id']}.json").write_text(json.dumps(phrases))
        kept.append(solo)
        n_phrases += len(phrases)

    sp = cfg["splits"]
    heldout_titles = set(sp.get("heldout_tunes") or [])
    heldout = {s["solo_id"] for s in kept if s["meta"].get("title") in heldout_titles}
    splits = split_by_solo(
        [s["solo_id"] for s in kept],
        ratios=(sp["train"], sp["val"], sp["test"]),
        seed=sp["seed"],
        heldout=heldout,
    )
    if verbose:
        print(f"{n_phrases} phrases from {len(kept)} solos -> {phrase_dir}")
        for name in SPLITS:
            print(f"  {name}: {len(splits[name])} solos")
    return splits


if __name__ == "__main__":
    build()
