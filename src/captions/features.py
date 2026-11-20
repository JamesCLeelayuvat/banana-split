"""Compute caption features per phrase -> data/features/<phrase_id>.json.

Register, range, density, rhythmic grid, contour, interval size, harmonic fit, space, context.
Word cut-offs are set at the training set's thirds.
"""


def compute_features(phrase: dict) -> dict:
    raise NotImplementedError


def fit_bins(train_features: list[dict]) -> dict:
    """Tercile cut-offs per numeric feature, from the training split only."""
    raise NotImplementedError


def to_words(features: dict, bins: dict) -> dict:
    """e.g. {"register": "high", "density": "dense", "contour": "descending", ...}"""
    raise NotImplementedError
