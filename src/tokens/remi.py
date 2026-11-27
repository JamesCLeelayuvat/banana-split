"""miditok REMI config and encode/decode for solo phrases.

Tokens (miditok names): Bar_None, Position_k (12 ticks/beat, 48/bar), Pitch_p (0-127; phrases use 0-126,
PITCH_MAX, and Pitch_127 is an unused extra that miditok 3.1 adds),
Duration_b.t.12 (1 tick up to 2 bars), specials PAD_None / BOS_None / EOS_None. No velocity,
tempo, chord, or rest tokens: rests are implied by gaps between positions.

Phrase dicts use the H1 format from src/data/phrases.py:
    {"notes": [{"pitch": int, "onset": int, "dur": int}, ...]}   # ticks, 12 per beat

Every encoded sequence is  BOS  Bar ...  EOS,  with one Bar token per bar of the phrase. If the
phrase dict has "bars", trailing empty bars are padded with Bar tokens so the length is explicit;
decode() returns the bar count under "bars" so encode(decode(x)) == x.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from miditok import REMI, TokenizerConfig
from symusic import Note, Score, Tempo, TimeSignature, Track

ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT / "configs" / "data.yaml"

SPECIAL_TOKENS = ["PAD", "BOS", "EOS"]
PAD, BOS, EOS, BAR = "PAD_None", "BOS_None", "EOS_None", "Bar_None"


def _data_config() -> dict:
    with open(DATA_CONFIG) as f:
        return yaml.safe_load(f)


_cfg = _data_config()
TICKS_PER_BEAT: int = _cfg["ticks_per_beat"]
BEATS_PER_BAR: int = _cfg["beats_per_bar"]
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR
PITCH_MIN, PITCH_MAX = _cfg["transpose"]["pitch_range"]  # inclusive
MAX_DUR = 2 * TICKS_PER_BAR  # duration cap: 2 bars


class TokenizeError(ValueError):
    """A phrase contains a note the vocabulary can't represent."""


@lru_cache(maxsize=1)
def build_tokenizer() -> REMI:
    config = TokenizerConfig(
        # miditok 3.1 treats this upper bound as inclusive, so the vocab also gets an unused
        # Pitch_{PITCH_MAX + 1}. Keep it: checkpoints were trained with that vocab (276 tokens),
        # and removing it would shift every Position/Duration id. Needs miditok==3.1.1.
        pitch_range=(PITCH_MIN, PITCH_MAX + 1),
        # one resolution for the whole duration range: every tick from 1 to 2 bars is a token
        beat_res={(0, 2 * BEATS_PER_BAR): TICKS_PER_BEAT},
        special_tokens=SPECIAL_TOKENS,
        use_velocities=False,
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=False,
        use_programs=False,
        use_sustain_pedals=False,
        use_pitch_bends=False,
        use_pitchdrum_tokens=False,  # otherwise miditok adds 61 unused drum-pitch tokens
    )
    return REMI(config)


def vocab() -> dict[str, int]:
    return build_tokenizer().vocab


def pad_id() -> int:
    return vocab()[PAD]


# ---------------------------------------------------------------- phrase <-> symusic Score

def _validate(notes: list[dict]) -> None:
    for i, n in enumerate(notes):
        p, on, d = n["pitch"], n["onset"], n["dur"]
        if not all(isinstance(v, int) for v in (p, on, d)):
            raise TokenizeError(f"note {i}: pitch/onset/dur must be ints, got {n}")
        if not PITCH_MIN <= p <= PITCH_MAX:
            raise TokenizeError(f"note {i}: pitch {p} outside [{PITCH_MIN}, {PITCH_MAX}]")
        if not 1 <= d <= MAX_DUR:
            raise TokenizeError(f"note {i}: duration {d} outside [1, {MAX_DUR}] ticks")
        if on < 0:
            raise TokenizeError(f"note {i}: negative onset {on}")


def phrase_to_score(phrase: dict, tempo: float = 120.0) -> Score:
    """Phrase dict -> one-track symusic Score in 4/4 at TICKS_PER_BEAT ticks per quarter."""
    notes = phrase["notes"]
    _validate(notes)
    score = Score(TICKS_PER_BEAT)
    score.time_signatures.append(TimeSignature(0, BEATS_PER_BAR, 4))
    score.tempos.append(Tempo(0, tempo))
    track = Track(program=66)  # tenor sax
    for n in notes:
        track.notes.append(Note(n["onset"], n["dur"], n["pitch"], 100))
    score.tracks.append(track)
    return score


def score_to_phrase(score: Score) -> dict:
    """symusic Score -> phrase dict, resampled to TICKS_PER_BEAT. Uses the first track."""
    if score.ticks_per_quarter != TICKS_PER_BEAT:
        score = score.resample(TICKS_PER_BEAT, min_dur=1)
    if not score.tracks:
        return {"notes": []}
    notes = [{"pitch": n.pitch, "onset": n.time, "dur": n.duration} for n in score.tracks[0].notes]
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return {"notes": notes}


def to_midi(phrase: dict, path: str | Path, tempo: float = 120.0) -> None:
    phrase_to_score(phrase, tempo).dump_midi(str(path))


def from_midi(path: str | Path) -> dict:
    return score_to_phrase(Score(str(path)))


# ---------------------------------------------------------------- encode / decode

def _ids_from(tokseq) -> list[int]:
    # miditok returns a TokSequence, or a list of them (one per track)
    if isinstance(tokseq, list):
        tokseq = tokseq[0] if tokseq else None
    return [] if tokseq is None else list(tokseq.ids)


def encode(phrase: dict) -> list[int]:
    """Phrase dict -> token ids: BOS, REMI tokens, EOS. Raises TokenizeError on bad notes."""
    tok = build_tokenizer()
    v = tok.vocab
    ids = _ids_from(tok.encode(phrase_to_score(phrase))) if phrase["notes"] else []
    if not ids or ids[0] != v[BAR]:  # empty phrase: still starts on a bar
        ids = [v[BAR]] + ids
    want_bars = phrase.get("bars")
    if want_bars is not None:
        have = ids.count(v[BAR])
        if have > want_bars:
            raise TokenizeError(f"phrase spans {have} bars but says bars={want_bars}")
        ids += [v[BAR]] * (want_bars - have)
    return [v[BOS]] + ids + [v[EOS]]


def decode(tokens: list[int]) -> dict:
    """Token ids -> phrase dict {"notes": [...], "bars": n}. BOS/EOS/PAD are ignored;
    anything after the first EOS is dropped."""
    tok = build_tokenizer()
    v = tok.vocab
    specials = {v[PAD], v[BOS]}
    body = []
    for t in tokens:
        if t == v[EOS]:
            break
        if t not in specials:
            body.append(t)
    n_bars = body.count(v[BAR])
    if not any(tok[t].startswith("Pitch_") for t in body):
        return {"notes": [], "bars": n_bars}
    phrase = score_to_phrase(tok.decode([body]))
    phrase["bars"] = n_bars
    return phrase


def ids_to_strings(tokens: list[int]) -> list[str]:
    tok = build_tokenizer()
    return [tok[t] for t in tokens]


# ---------------------------------------------------------------- per-token context for the decoder

def token_context(tokens: list[int]) -> tuple[list[int], list[int]]:
    """For each token, the bar position (0..47) and beat index (bar*4 + pos//12) it sits at.

    BOS and Bar are at position 0 of their bar; Pitch and Duration inherit the latest Position;
    EOS keeps the last state. The beat index maps straight onto the phrase's per-beat chord list
    (clamp it at len(chords)-1 when looking up). Works on any prefix, so generation can reuse it.
    """
    tok = build_tokenizer()
    bar, pos = -1, 0
    bar_pos, beat = [], []
    for t in tokens:
        name = tok[t]
        if name == BAR:
            bar, pos = bar + 1, 0
        elif name.startswith("Position_"):
            pos = int(name.split("_")[1])
        b = max(bar, 0)
        bar_pos.append(pos)
        beat.append(b * BEATS_PER_BAR + pos // TICKS_PER_BEAT)
    return bar_pos, beat


def encode_with_context(phrase: dict) -> dict[str, list[int]]:
    """Token ids plus aligned bar_pos and beat sequences (same length)."""
    ids = encode(phrase)
    bar_pos, beat = token_context(ids)
    return {"ids": ids, "bar_pos": bar_pos, "beat": beat}
