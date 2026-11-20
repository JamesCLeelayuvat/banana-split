"""Chord fit: share of strong-beat notes that are chord tones; share of approach notes resolving to one."""


def chord_tone_rate(phrase: dict, strong_beats=(0, 2)) -> float:
    raise NotImplementedError


def approach_rate(phrase: dict) -> float:
    raise NotImplementedError
