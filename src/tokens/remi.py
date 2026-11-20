"""miditok REMI config and encode/decode.

Tokens: BAR, POS_k (12 ticks/beat, 48/bar), PITCH_p (instrument range), DUR_d (cap 2 bars),
specials BOS / EOS / PAD. Rests are implied by gaps.
"""


def build_tokenizer():
    raise NotImplementedError


def encode(phrase: dict) -> list[int]:
    raise NotImplementedError


def decode(tokens: list[int]) -> dict:
    """Return a phrase dict ({"notes": [...]}) from token ids."""
    raise NotImplementedError
