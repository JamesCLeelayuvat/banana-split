"""LLM captioning, offline: prompt building, request shape, response validation."""
import json

import pytest

from src.captions import caption_llm as C

WORDS = {"register": "high", "density": "dense", "contour": "descending", "rhythm": "eighth notes",
         "harmony": "chromatic", "range": None, "intervals": "stepwise", "space": None,
         "style": "bebop", "tempo": "up-tempo", "instrument": "alto sax"}


def response(captions, finish="STOP"):
    """A GenerateContentResponse in REST dict form, with a thought part before the answer."""
    return {"candidates": [{"finishReason": finish, "content": {"role": "model", "parts": [
        {"text": "thinking...", "thought": True},
        {"text": json.dumps({"captions": captions})},
    ]}}]}


def test_describe_skips_unmeasured_and_keeps_order():
    lines = C.describe(WORDS).splitlines()
    assert lines[0] == "register: high" and lines[-1] == "instrument: alto sax"
    assert not any(line.startswith(("range", "space")) for line in lines)


def test_system_prompt_includes_examples_and_is_deterministic():
    ex = [{"feature_words": {"register": "low"}, "caption": "a low, lazy line"}]
    a, b = C.build_system(ex), C.build_system(ex)
    assert a == b  # identical across requests, so implicit caching can hit
    assert "a low, lazy line" in a
    assert "a low, lazy line" not in C.build_system([])


def test_load_examples_skips_unwritten(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(json.dumps([{"caption": ""}, {"caption": "TODO: describe"},
                                {"caption": "dense line", "feature_words": {}}]))
    assert [e["caption"] for e in C.load_examples(path)] == ["dense line"]


def test_request_body_is_valid_for_sdk_and_batch_file():
    from google.genai import types

    body = C.request_body(WORDS, C.build_system([]))
    assert "register: high" in body["contents"][0]["parts"][0]["text"]
    json.dumps({"key": "p", "request": body})  # batch input line must serialize
    config = types.GenerateContentConfig.model_validate(
        {**body["generationConfig"], "systemInstruction": body["systemInstruction"]})
    assert config.response_json_schema == C.SCHEMA
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.system_instruction.parts[0].text == C.SYSTEM
    types.Content.model_validate(body["contents"][0])


def test_parse_captions_accepts_three_distinct_and_skips_thoughts():
    caps = ["a dense high line", "dense and high", "dense high bebop"]
    assert C.parse_captions(response([c + " " for c in caps])) == caps


def test_parse_captions_reads_sdk_response():
    from google.genai import types

    caps = ["a dense high line", "dense and high", "dense high bebop"]
    sdk = types.GenerateContentResponse.model_validate(response(caps))
    assert C.parse_captions(sdk.model_dump(mode="json", by_alias=True, exclude_none=True)) == caps


@pytest.mark.parametrize("resp", [
    response(["one", "two"]),                         # wrong count
    response(["same", "same", "other"]),              # duplicates
    response(["one", "", "three"]),                   # empty
    response(["a", "b", "c"], "SAFETY"),
    response(["a", "b", "c"], "MAX_TOKENS"),
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": []},
])
def test_parse_captions_rejects(resp):
    with pytest.raises(ValueError):
        C.parse_captions(resp)


def test_write_results_appends_good_rows_and_counts_failures(tmp_path):
    out = tmp_path / "captions.jsonl"
    out.write_text(json.dumps({"phrase_id": "old", "captions": ["x", "y", "z"]}) + "\n")
    good = ["a dense high line", "dense and high", "dense high bebop"]
    raw = "\n".join(json.dumps(r) for r in [
        {"key": "p1", "response": response(good)},
        {"key": "old", "response": response(good)},          # already captioned: skipped
        {"key": "p2", "error": {"code": 500, "message": "boom"}},
        {"key": "p3", "response": response(["a", "a", "b"])},
    ]).encode()
    ok, failed = C.write_results(raw, out)
    assert ok == 1 and sum(failed.values()) == 2
    assert C.load_done(out) == {"old", "p1"}


def test_create_when_room_waits_out_quota_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from google.genai import errors

    quota = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": "active batch tokens"}})
    attempts, polls = [], []

    class Batches:
        def create(self, model, src, config):
            attempts.append(src)
            if len(attempts) < 3:
                raise quota
            return SimpleNamespace(name="batches/new")

        def get(self, name):
            polls.append(name)
            return SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_RUNNING"))

    client = SimpleNamespace(batches=Batches())
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    state = [{"id": "batches/old", "n": 1, "model": "m", "collected": False}]
    batch = C.create_when_room(client, "m", "files/x", "name", state, tmp_path / "b.json",
                               tmp_path / "c.jsonl")
    assert batch.name == "batches/new" and len(attempts) == 3 and polls == ["batches/old"] * 2


def test_create_when_room_gives_up_when_nothing_is_running(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest
    from google.genai import errors

    class Batches:
        def create(self, model, src, config):
            raise errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                     "message": "too big"}})

    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="max-active-tokens"):
        C.create_when_room(SimpleNamespace(batches=Batches()), "m", "f", "n", [],
                           tmp_path / "b.json", tmp_path / "c.jsonl")
