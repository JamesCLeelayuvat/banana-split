"""Diversity and copying: pitch-class / rhythm entropy, longest note n-gram shared with any training phrase."""


def pitch_class_entropy(phrases: list[dict]) -> float:
    raise NotImplementedError


def rhythm_entropy(phrases: list[dict]) -> float:
    raise NotImplementedError


def longest_copied_run(phrase: dict, train_index) -> int:
    raise NotImplementedError
