"""MIDI -> audio: swing on eighth notes at playback, FluidSynth + soundfont, simple bass/drums/piano backing.

Usage: python -m src.render outputs/samples/x.mid --out outputs/audio/ --soundfont path/to.sf2
"""


def apply_swing(midi, ratio: float = 2 / 3):
    raise NotImplementedError


def add_backing(midi, chords: list[str], tempo: int):
    raise NotImplementedError


def render(midi_path: str, out_dir: str, soundfont: str):
    raise NotImplementedError
