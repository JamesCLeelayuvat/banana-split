"""Features + hand-written examples -> 3 differently worded captions per phrase (H2).

Writes data/captions/captions.jsonl: {"phrase_id": str, "captions": [str, str, str]}
"""
import json
from pathlib import Path


def build_prompt(feature_words: dict, meta: dict, examples: list[dict]) -> str:
    raise NotImplementedError


def caption_phrase(feature_words: dict, meta: dict, examples: list[dict], n: int = 3) -> list[str]:
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
