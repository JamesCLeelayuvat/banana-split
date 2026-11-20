"""Description control: generate N phrases per wording over the same changes, measure the matching feature.

Report the difference between wordings and the share landing in the requested bin.
Run for register, density, contour, harmonic fit; repeat with data/human_eval descriptions.
"""

PAIRS = {
    "register": ("low register", "high register"),
    "density": ("sparse", "dense"),
    "contour": ("ascending", "descending"),
    "harmonic_fit": ("inside", "outside"),
}


def run(samples_dir: str, feature: str) -> dict:
    raise NotImplementedError
