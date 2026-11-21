"""MIDI -> audio: swing on eighth notes at playback, FluidSynth + soundfont, simple bass/drums/piano backing.

The solo comes in straight (on the model's 12-ticks-per-beat grid). add_backing() writes a
walking bass, comping piano and a ride/hi-hat swing pattern under it, also straight;
apply_swing() then swings every track at once, so solo and band agree on where the off-beats
are. FluidSynth renders the result to a WAV.

Usage:
    python -m src.render outputs/samples/x.mid --chords "Dm7 G7 | Cmaj7 | %" --tempo 160 \
        --out outputs/audio/ --soundfont path/to.sf2
    python -m src.render data/phrases/weimar_1.json --index 3 --out outputs/audio/   # a real phrase

The soundfont defaults to $SOUNDFONT and the synth to `fluidsynth` on PATH (or $FLUIDSYNTH).
Writes <out>/<name>.wav plus <out>/<name>_mix.mid, the full swung arrangement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

from symusic import Note, Score, Tempo, TimeSignature, Track

from src.data.chords import INTERVALS, parse
from src.generate import parse_changes
from src.tokens import remi

TPQ = 480                                # render resolution, ticks per beat
STEP = TPQ // remi.TICKS_PER_BEAT        # one model tick (1/12 beat) in render ticks
BEATS = remi.BEATS_PER_BAR
BAR = TPQ * BEATS

Chord = tuple[int, str] | None

# General MIDI programs / drum keys
BASS_PROGRAM, PIANO_PROGRAM = 32, 0      # acoustic bass, acoustic grand
STICKS, KICK, HAT_PEDAL, CRASH, RIDE = 31, 36, 44, 49, 51

BASS_RANGE = (28, 55)                    # E1..G3
PIANO_LOW = (48, 62)                     # lowest voiced note sits in C3..D4

# Rootless left-hand voicings, semitones above the root (3-7-9 style shells).
VOICINGS = {
    "maj7": (4, 7, 11, 14),
    "6": (4, 7, 9, 14),
    "7": (4, 9, 10, 14),                 # 3 13 b7 9
    "m7": (3, 7, 10, 14),
    "m6": (3, 7, 9, 14),
    "m7b5": (3, 6, 10, 12),
    "dim7": (3, 6, 9, 12),
    "7alt": (4, 10, 15, 20),             # 3 b7 #9 b13
    "sus": (5, 7, 10, 14),
}

# Piano comp rhythms, one bar each: (onset, duration) in model ticks (12 per beat).
COMP_PATTERNS = [
    [(0, 8), (18, 6)],                   # Charleston: 1, and-of-2
    [(6, 6), (30, 6)],                   # and-of-1, and-of-3
    [(12, 10), (36, 10)],                # 2 and 4
    [(18, 6), (42, 6)],                  # anticipations into beats 3 and next 1
    [(0, 22)],                           # long hit on 1
    [(30, 8)],                           # one stab on and-of-3
]


# ---------------------------------------------------------------- swing

def swing_for_tempo(tempo: float) -> float:
    """Swing ratio (long eighth / beat): 2/3 up to 160 bpm, flattening to 0.56 by 320 bpm."""
    t = min(max((tempo - 160) / 160, 0.0), 1.0)
    return 2 / 3 + t * (0.56 - 2 / 3)


def _swing_tick(t: int, ratio: float) -> int:
    """Move a time on the straight 8th/16th grid to its swung place; leave triplets alone."""
    beat, p = divmod(t, TPQ)
    if p % (TPQ // 4):
        return t
    half = TPQ / 2
    q = p * 2 * ratio if p <= half else TPQ * ratio + (p - half) * 2 * (1 - ratio)
    return beat * TPQ + round(q)


def apply_swing(midi: Score, ratio: float = 2 / 3) -> Score:
    """Swung copy of `midi` at TPQ resolution. Note starts and ends on straight 8ths/16ths are
    pushed late (ratio 2/3 = triplet swing, 0.5 = straight); triplet positions are untouched."""
    out = midi.resample(TPQ, min_dur=1) if midi.ticks_per_quarter != TPQ else midi.copy()
    for track in out.tracks:
        for n in track.notes:
            start = _swing_tick(n.time, ratio)
            end = _swing_tick(n.time + n.duration, ratio)
            n.time, n.duration = start, max(1, end - start)
    return out


# ---------------------------------------------------------------- backing parts

def _nearest(pc: int, ref: int, lo: int, hi: int) -> int:
    """The pitch with pitch class `pc` in [lo, hi] closest to `ref`."""
    cands = [p for p in range(lo, hi + 1) if p % 12 == pc]
    return min(cands, key=lambda p: (abs(p - ref), p))


def _bass(chords: list[Chord], rng: random.Random) -> list[Note]:
    """Walking bass, one quarter per beat: root on chord changes and downbeats, chord tones in
    between, and a chromatic approach into the next chord's root."""
    lo, hi = BASS_RANGE
    notes, prev = [], 40
    for i, ch in enumerate(chords):
        if ch is None:
            continue
        root, quality = ch
        nxt = chords[i + 1] if i + 1 < len(chords) else None
        if i == 0 or chords[i - 1] != ch or i % BEATS == 0:
            pitch = _nearest(root, prev, lo, hi)
        elif nxt is not None and nxt != ch:
            target = _nearest(nxt[0], prev, lo + 1, hi - 1)
            pitch = target - 1 if prev < target else target + 1
        else:
            tones = {(root + iv) % 12 for iv in INTERVALS[quality]}
            cands = sorted((p for p in range(lo, hi + 1) if p % 12 in tones and p != prev),
                           key=lambda p: abs(p - prev))
            pitch = rng.choice(cands[:3])
        vel = 92 if i % BEATS == 0 else 84
        notes.append(Note(i * TPQ, TPQ - STEP, pitch, vel))
        prev = pitch
    return notes


def _voicing(ch: tuple[int, str], center: float) -> list[int]:
    root, quality = ch
    shape = VOICINGS[quality]
    lo, hi = PIANO_LOW
    options = []
    for base in range(lo - 24, hi + 1):
        if base % 12 == root and lo <= base + shape[0] <= hi:
            options.append([base + iv for iv in shape])
    return min(options, key=lambda v: abs(sum(v) / len(v) - center))


def _piano(chords: list[Chord], n_bars: int, rng: random.Random) -> list[Note]:
    """Comping: one rhythm pattern per bar, rootless voicings with simple voice leading.
    Off-beat hits take the chord of the beat they anticipate; any chord the pattern misses
    gets its own hit where it starts."""
    notes, center = [], 64.0
    beats_tick = remi.TICKS_PER_BEAT
    for bar in range(n_bars):
        b0 = bar * BEATS
        hits = []
        for onset, dur in rng.choice(COMP_PATTERNS):
            beat = min(b0 + (onset + beats_tick // 2) // beats_tick, len(chords) - 1)
            hits.append((onset, dur, chords[beat]))
        for b in range(b0, b0 + BEATS):  # chord changes the pattern didn't hit
            ch = chords[b]
            if ch is not None and (b == 0 or chords[b - 1] != ch) and ch not in {h[2] for h in hits}:
                hits.append(((b - b0) * beats_tick, 8, ch))
        for onset, dur, ch in sorted(hits, key=lambda h: h[0]):
            if ch is None:
                continue
            voicing = _voicing(ch, center)
            center = sum(voicing) / len(voicing)
            vel = rng.randint(52, 66)
            for p in voicing:
                notes.append(Note(bar * BAR + onset * STEP, dur * STEP, p, vel))
    return notes


def _drums(n_bars: int, count_in: int) -> list[Note]:
    """Count-in on sticks, then ride 'spang-a-lang', hi-hat on 2 and 4, feathered kick."""
    notes = []
    for bar in range(n_bars):
        t0 = bar * BAR
        if bar < count_in:
            notes += [Note(t0 + b * TPQ, STEP * 2, STICKS, 80) for b in range(BEATS)]
            continue
        for b in range(BEATS):
            t = t0 + b * TPQ
            notes.append(Note(t, TPQ // 2, RIDE, 78 if b % 2 else 70))
            notes.append(Note(t, TPQ // 2, KICK, 30))
            if b % 2:
                notes.append(Note(t, TPQ // 2, HAT_PEDAL, 72))
                notes.append(Note(t + TPQ // 2, TPQ // 4, RIDE, 56))  # skip note, swung later
    return notes


# ---------------------------------------------------------------- arrangement

def _solo_end(midi: Score) -> int:
    return max((n.time + n.duration for t in midi.tracks for n in t.notes), default=0)


def add_backing(midi: Score, chords: list[str], tempo: float, count_in: int = 1,
                bars: int | None = None, seed: int = 0) -> Score:
    """Straight (unswung) arrangement at TPQ: the solo tracks of `midi`, shifted by `count_in`
    bars, over bass, piano and drums following `chords` (one symbol per beat, looped if shorter
    than the solo). Ends with one bar ringing the first chord, like a tag back to the top."""
    rng = random.Random(seed)
    solo = midi.resample(TPQ, min_dur=1) if midi.ticks_per_quarter != TPQ else midi.copy()
    if bars is None:
        bars = max(math.ceil(_solo_end(solo) / BAR), math.ceil(len(chords) / BEATS), 1)
    parsed = [parse(c) for c in chords] or [None]
    n_beats = bars * BEATS
    per_beat = [parsed[i % len(parsed)] for i in range(n_beats)]
    final = next((c for c in parsed if c is not None), None)

    score = Score(TPQ)
    score.time_signatures.append(TimeSignature(0, BEATS, 4))
    score.tempos.append(Tempo(0, tempo))
    shift = count_in * BAR
    for t in solo.tracks:
        track = Track(name=t.name or "Solo", program=t.program, is_drum=t.is_drum)
        for n in t.notes:
            track.notes.append(Note(n.time + shift, n.duration, n.pitch, n.velocity))
        score.tracks.append(track)

    bass = Track(name="Bass", program=BASS_PROGRAM)
    piano = Track(name="Piano", program=PIANO_PROGRAM)
    drums = Track(name="Drums", program=0, is_drum=True)
    for n in _bass(per_beat, rng):
        bass.notes.append(Note(n.time + shift, n.duration, n.pitch, n.velocity))
    for n in _piano(per_beat, bars, rng):
        piano.notes.append(Note(n.time + shift, n.duration, n.pitch, n.velocity))
    for n in _drums(count_in + bars, count_in):
        drums.notes.append(n)

    end = shift + bars * BAR  # tag bar: final hit, let it ring
    drums.notes.append(Note(end, BAR, CRASH, 80))
    drums.notes.append(Note(end, TPQ, KICK, 60))
    if final is not None:
        bass.notes.append(Note(end, BAR - STEP, _nearest(final[0], 36, *BASS_RANGE), 90))
        for p in _voicing(final, 64.0):
            piano.notes.append(Note(end, BAR - STEP, p, 62))

    score.tracks.extend([bass, piano, drums])
    return score


# ---------------------------------------------------------------- FluidSynth

def _fluidsynth_exe() -> str:
    exe = os.environ.get("FLUIDSYNTH") or shutil.which("fluidsynth")
    if not exe:
        raise FileNotFoundError(
            "FluidSynth not found: put `fluidsynth` on PATH or set $FLUIDSYNTH to the executable "
            "(Windows builds: https://github.com/FluidSynth/fluidsynth/releases)")
    return exe


def synthesize(midi_path: str | Path, wav_path: str | Path, soundfont: str | Path,
               gain: float = 1.0, sample_rate: int = 44100) -> Path:
    """Render a MIDI file to WAV with the FluidSynth command-line synth."""
    if not Path(soundfont).is_file():
        raise FileNotFoundError(f"soundfont not found: {soundfont}")
    cmd = [_fluidsynth_exe(), "-ni", "-q", "-g", str(gain), "-r", str(sample_rate),
           "-F", str(wav_path), "-T", "wav", str(soundfont), str(midi_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(wav_path)


def render_score(solo: Score, out_dir: str | Path, soundfont: str | Path, name: str,
                 chords: list[str] | None = None, tempo: float | None = None,
                 swing: float | None = None, backing: bool = True, count_in: int = 1,
                 seed: int = 0) -> Path:
    """Solo score -> <out_dir>/<name>.wav (+ <name>_mix.mid). Returns the WAV path."""
    if tempo is None:
        tempo = solo.tempos[0].qpm if len(solo.tempos) else 120.0
    if backing:
        score = add_backing(solo, chords or [], tempo, count_in=count_in, seed=seed)
    else:
        score = solo.resample(TPQ, min_dur=1)
        score.tempos.clear()
        score.tempos.append(Tempo(0, tempo))
    score = apply_swing(score, swing_for_tempo(tempo) if swing is None else swing)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_path, wav_path = out_dir / f"{name}_mix.mid", out_dir / f"{name}.wav"
    score.dump_midi(str(mid_path))
    return synthesize(mid_path, wav_path, soundfont)


def render(midi_path: str, out_dir: str, soundfont: str, chords: list[str] | None = None,
           tempo: float | None = None, **kwargs) -> Path:
    """Render a solo MIDI file (e.g. from src.generate) over a backing for `chords`."""
    return render_score(Score(str(midi_path)), out_dir, soundfont, Path(midi_path).stem,
                        chords=chords, tempo=tempo, **kwargs)


def render_phrase(phrase: dict, out_dir: str, soundfont: str, tempo: float | None = None,
                  name: str | None = None, **kwargs) -> Path:
    """Render a phrase dict (H1 format) over its own per-beat chords; tempo defaults to the
    phrase's meta tempo."""
    if tempo is None:
        tempo = float(phrase.get("meta", {}).get("tempo") or 160)
    solo = remi.phrase_to_score(phrase, tempo)
    bars = phrase.get("bars")
    chords = phrase.get("chords") or []
    if bars and len(chords) < bars * BEATS:
        chords = chords + chords[-1:] * (bars * BEATS - len(chords))
    return render_score(solo, out_dir, soundfont, name or phrase.get("phrase_id", "phrase"),
                        chords=chords, tempo=tempo, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="solo .mid, or a phrase .json (one phrase or a solo's list)")
    ap.add_argument("--chords", help='changes, e.g. "Dm7 G7 | Cmaj7 | %%" (.mid input)')
    ap.add_argument("--index", type=int, default=0, help="phrase index in a .json list")
    ap.add_argument("--tempo", type=float, help="bpm (default: from the MIDI / phrase meta)")
    ap.add_argument("--swing", type=float, help="swing ratio, 0.5 straight .. 0.75 hard "
                                                 "(default: by tempo, 2/3 at medium tempos)")
    ap.add_argument("--count-in", type=int, default=1, help="bars of sticks before the solo")
    ap.add_argument("--no-backing", action="store_true", help="render the solo alone")
    ap.add_argument("--seed", type=int, default=0, help="seed for bass/comping choices")
    ap.add_argument("--out", default="outputs/audio")
    ap.add_argument("--soundfont", default=os.environ.get("SOUNDFONT"))
    args = ap.parse_args()
    if not args.soundfont:
        ap.error("no soundfont: pass --soundfont or set $SOUNDFONT")

    opts = dict(swing=args.swing, backing=not args.no_backing, count_in=args.count_in,
                seed=args.seed)
    if args.input.endswith(".json"):
        with open(args.input) as f:
            data = json.load(f)
        phrase = data[args.index] if isinstance(data, list) else data
        wav = render_phrase(phrase, args.out, args.soundfont, tempo=args.tempo, **opts)
    else:
        chords = parse_changes(args.chords) if args.chords else None
        if chords is None and not args.no_backing:
            print("no --chords given: backing will be drums only")
        wav = render(args.input, args.out, args.soundfont, chords=chords, tempo=args.tempo, **opts)
    print(f"wrote {wav} and {wav.with_name(wav.stem + '_mix.mid')}")


if __name__ == "__main__":
    main()
