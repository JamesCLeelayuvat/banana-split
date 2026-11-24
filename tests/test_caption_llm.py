"""LLM captioning, offline: prompt building, request shape, response validation."""
import json
from types import SimpleNamespace

import pytest

from src.captions import caption_llm as C

WORDS = {"register": "high", "density": "dense", "contour": "descending", "rhythm": "eighth notes",
         "harmony": "chromatic", "range": None, "intervals": "stepwise", "space": None,
         "style": "bebop", "tempo": "up-tempo", "instrument": "alto sax"}


def message(captions, stop_reason="end_turn"):
    text = json.dumps({"captions": captions})
    return SimpleNamespace(stop_reason=stop_reason,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)])


def test_describe_skips_unmeasured_and_keeps_order():
    lines = C.describe(WORDS).splitlines()
    assert lines[0] == "register: high" and lines[-1] == "instrument: alto sax"
    assert not any(line.startswith(("range", "space")) for line in lines)


def test_system_prompt_includes_examples_and_is_deterministic():
    ex = [{"feature_words": {"register": "low"}, "caption": "a low, lazy line"}]
    a, b = C.build_system(ex), C.build_system(ex)
    assert a == b  # identical across requests, so the prompt cache can hit
    assert "a low, lazy line" in a[0]["text"] and a[0]["cache_control"] == {"type": "ephemeral"}
    assert "a low, lazy line" not in C.build_system([])[0]["text"]


def test_load_examples_skips_unwritten(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(json.dumps([{"caption": ""}, {"caption": "TODO: describe"},
                                {"caption": "dense line", "feature_words": {}}]))
    assert [e["caption"] for e in C.load_examples(path)] == ["dense line"]


def test_request_params_are_valid_batch_params():
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    params = C.request_params(WORDS, C.build_system([]))
    assert params["output_config"]["format"]["schema"] == C.SCHEMA
    assert "register: high" in params["messages"][0]["content"]
    Request(custom_id="p", params=MessageCreateParamsNonStreaming(**params))


def test_parse_captions_accepts_three_distinct():
    caps = ["a dense high line", "dense and high", "dense high bebop"]
    assert C.parse_captions(message([c + " " for c in caps])) == caps


@pytest.mark.parametrize("caps,stop", [
    (["one", "two"], "end_turn"),                 # wrong count
    (["same", "same", "other"], "end_turn"),       # duplicates
    (["one", "", "three"], "end_turn"),            # empty
    (["a", "b", "c"], "refusal"),
    (["a", "b", "c"], "max_tokens"),
])
def test_parse_captions_rejects(caps, stop):
    with pytest.raises(ValueError):
        C.parse_captions(message(caps, stop))
