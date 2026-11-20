"""Quantize onsets, cut 4-bar phrases (2-bar stride), split by solo, write data/phrases/*.json (H1).

Phrase JSON:
{
  "phrase_id": str, "solo_id": str,
  "notes": [{"pitch": int, "onset": int, "dur": int}],   # ticks, 12 per beat
  "chords": [str, ...],                                    # one per beat
  "meta": {"performer", "instrument", "style", "tempo", "key", ...}
}
"""


def quantize(solo: dict, ticks_per_beat: int = 12) -> dict:
    raise NotImplementedError


def cut_phrases(solo: dict, bars: int = 4, stride: int = 2, max_rest: float = 0.6) -> list[dict]:
    raise NotImplementedError


def split_by_solo(solo_ids: list[str], ratios=(0.8, 0.1, 0.1), seed: int = 0) -> dict[str, list[str]]:
    """Write data/splits/{train,val,test}.txt."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
