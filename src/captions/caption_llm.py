"""Feature words + hand-written examples -> 3 differently worded captions per phrase (H2).

Reads data/features/features.jsonl (python -m src.captions.features) and
src/captions/examples.json. Writes data/captions/captions.jsonl:
    {"phrase_id": str, "captions": [str, str, str]}

Steps (python -m src.captions.caption_llm <command>):
    pick-examples  choose ~30 varied training phrases and write them to examples.json with an
                   empty "caption" for you to fill in (listen with the render command shown)
    try            caption a few phrases right away and print them, to check the prompt
    submit         send every phrase that has no caption yet to the Gemini Batch API
                   (50% price, usually done within a few hours); batch ids go to batches.json.
                   Jobs are sized to fit the account's limit on tokens in active batches; when
                   it is full, submit waits, collects whatever has finished, and continues
    collect        wait for the submitted batches and append their captions to captions.jsonl

Re-running submit only sends phrases still missing from captions.jsonl, so a phrase whose
captions failed validation is simply picked up again.

Uses Gemini through the google-genai SDK. Needs GEMINI_API_KEY, set in the environment or as a
line in a .env file at the repo root (gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = Path(__file__).with_name("examples.json")
MODEL = "gemini-3.8-flash"
THINKING = "LOW"          # paraphrasing a handful of given words; little reasoning needed
MAX_TOKENS = 4096         # captions themselves are ~100 tokens
N_CAPTIONS = 3
MAX_ACTIVE_TOKENS = 2_500_000  # input tokens per batch job: under the 3M limit on tokens in
                               # active batch jobs (Tier 1); raise with --max-active-tokens
POLL_SECONDS = 60

# Order in which features are shown to the model (fixed, so prompts are deterministic).
FEATURE_ORDER = ("register", "density", "contour", "rhythm", "harmony", "range", "intervals",
                 "space")
CONTEXT_ORDER = ("style", "tempo", "instrument")

SCHEMA = {
    "type": "object",
    "properties": {"captions": {"type": "array", "items": {"type": "string"},
                                "minItems": N_CAPTIONS, "maxItems": N_CAPTIONS}},
    "required": ["captions"],
}
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
FAILED_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

SYSTEM = """\
You write short descriptions of jazz solo phrases, the way one musician would describe a line \
to another. They will be used as text prompts for a model that generates solos, so they must \
be accurate and should sound like something a player would actually type or say.

Each phrase is 4 bars of a monophonic improvised line. You never hear it: you get words that \
were measured from the notes, plus context from the recording. Describe only what those words \
say. Don't add details that weren't measured (specific notes, licks, quotes, emotions, dynamics, \
tone), and never contradict a word. Don't name performers or tunes.

What the words mean, so you can paraphrase them freely:
- register: low / middle / high: where the line sits in pitch ("upper register", "down low")
- density: sparse / moderate / dense: how many notes ("sparse, lots of air", "busy, packed")
- contour: descending / level / ascending / arching (rises then falls back)
- rhythm: eighth notes / sixteenth runs / triplets: the main subdivision
- harmony: inside (sticks to chord tones and the scale) / chromatic (half-step approach notes, \
enclosures) / outside (notes that clash with the changes, playing outside)
- range: narrow / wide: span from lowest to highest note
- intervals: stepwise (scalar, small steps) / intervallic (lots of leaps)
- space: busy (almost no rests) / spacious (lots of rests between ideas)
- style, tempo, instrument: from the recording, e.g. "hard bop", "up-tempo", "tenor sax"
Middle values ("middle", "moderate", "level") are unremarkable; mention them only lightly or \
leave them out. Features that aren't listed weren't measured or were unremarkable; don't \
mention them.

Write {n} captions for the phrase, each worded differently:
1. One natural sentence covering most of the features.
2. A shorter description picking out the two or three most distinctive features.
3. A terse prompt-style phrase, the way someone would type a request \
("dense high bebop line, mostly descending").
Vary the vocabulary across the three; don't reuse the feature words verbatim every time. Each \
caption should mention at least one of register, density, contour or harmony when they are \
listed. Keep each under 35 words. No numbers.

Return JSON: {{"captions": [...]}} with exactly {n} strings.\
""".format(n=N_CAPTIONS)


# ---------------------------------------------------------------- client

def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into os.environ (real environment variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_client():
    from google import genai

    load_env()
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("set GEMINI_API_KEY (environment variable, or a line in .env at the repo root)")
    return genai.Client()


# ---------------------------------------------------------------- data

def _paths() -> dict:
    with open(ROOT / "configs" / "data.yaml") as f:
        return {k: ROOT / v for k, v in yaml.safe_load(f)["paths"].items()}


def load_features() -> dict[str, dict]:
    """phrase_id -> features.jsonl row ({"split", "features", "words"})."""
    path = _paths()["features"] / "features.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found: run `python -m src.captions.features` first")
    with open(path, encoding="utf-8") as f:
        return {r["phrase_id"]: r for r in map(json.loads, f) if r}


def load_examples(path: Path = EXAMPLES) -> list[dict]:
    """Hand-written examples that have a caption (empty or TODO ones are skipped)."""
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("caption", "").strip() and "TODO" not in r["caption"]]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["phrase_id"] for line in f if line.strip()}


# ---------------------------------------------------------------- prompt

def describe(feature_words: dict) -> str:
    """Feature words -> the 'key: value' block the model sees (unmeasured ones left out)."""
    lines = [f"{k}: {feature_words[k]}" for k in FEATURE_ORDER + CONTEXT_ORDER
             if feature_words.get(k)]
    return "\n".join(lines)


def build_system(examples: list[dict]) -> str:
    """System instruction: instructions + examples. Identical for every phrase, so Gemini's
    implicit prompt caching can reuse it across requests."""
    text = SYSTEM
    if examples:
        shots = "\n\n".join(f"<example>\n{describe(e['feature_words'])}\n"
                            f"caption: {e['caption'].strip()}\n</example>" for e in examples)
        text += ("\n\nHand-written captions for other phrases, showing the vocabulary and tone "
                 "we want (each shows one caption; you write " + str(N_CAPTIONS) + "):\n\n" + shots)
    return text


def build_prompt(feature_words: dict) -> str:
    """User message for one phrase. Context (style, tempo, instrument) is already among the
    feature words, and the examples live in the shared system instruction."""
    return f"<phrase>\n{describe(feature_words)}\n</phrase>"


def request_body(feature_words: dict, system: str) -> dict:
    """One GenerateContentRequest in REST (camelCase) form: a line of the batch input file,
    and also what the synchronous call sends."""
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": build_prompt(feature_words)}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_TOKENS,
            "thinkingConfig": {"thinkingLevel": THINKING},
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
        },
    }


def parse_captions(response: dict) -> list[str]:
    """Validated captions from a GenerateContentResponse (REST dict form), or raise ValueError
    with the reason."""
    if reason := (response.get("promptFeedback") or {}).get("blockReason"):
        raise ValueError(f"blocked: {reason}")
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("no candidates")
    cand = candidates[0]
    finish = cand.get("finishReason", "STOP")
    if finish == "MAX_TOKENS":
        raise ValueError("hit max_tokens")
    if finish != "STOP":
        raise ValueError(f"finish reason {finish}")
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", [])
                   if not p.get("thought"))
    if not text:
        raise ValueError("no text")
    caps = [c.strip() for c in json.loads(text)["captions"]]
    if len(caps) != N_CAPTIONS or not all(caps) or len(set(caps)) != N_CAPTIONS:
        raise ValueError(f"expected {N_CAPTIONS} distinct non-empty captions, got {caps!r}")
    return caps


# ---------------------------------------------------------------- one phrase (synchronous)

def caption_phrase(feature_words: dict, examples: list[dict] | None = None, client=None,
                   model: str = MODEL) -> list[str]:
    """Caption one phrase with a direct API call (for `try` and spot checks)."""
    from google.genai import types

    client = client or make_client()
    body = request_body(feature_words, build_system(examples or []))
    config = types.GenerateContentConfig.model_validate(
        {**body["generationConfig"], "systemInstruction": body["systemInstruction"]})
    response = client.models.generate_content(model=model, contents=body["contents"],
                                              config=config)
    return parse_captions(response.model_dump(mode="json", by_alias=True, exclude_none=True))


# ---------------------------------------------------------------- commands

def cmd_pick_examples(args) -> None:
    """Greedy pick of training phrases covering every (feature, word) pair, then random fill."""
    if load_examples() and not args.force:
        sys.exit(f"{EXAMPLES} already has written captions; pass --force to overwrite")
    rows = [r for r in load_features().values() if r["split"] == "train"]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    keys = ("register", "density", "contour", "rhythm", "harmony", "range", "intervals", "space",
            "style", "tempo")
    covered, picked = set(), []
    while len(picked) < args.n and rows:
        pairs = lambda r: {(k, r["words"][k]) for k in keys if r["words"].get(k)}  # noqa: E731
        best = max(rows, key=lambda r: len(pairs(r) - covered))
        rows.remove(best)
        picked.append(best)
        covered |= pairs(best)

    out = []
    for r in picked:
        pid = r["phrase_id"]
        solo_id = pid.rsplit("_b", 1)[0]
        solo = json.loads((_paths()["phrases"] / f"{solo_id}.json").read_text())
        index = next(i for i, p in enumerate(solo) if p["phrase_id"] == pid)
        meta = solo[index]["meta"]
        out.append({
            "phrase_id": pid,
            "feature_words": r["words"],
            "caption": "",
            "listen": f"python -m src.render data/phrases/{solo_id}.json --index {index} "
                      f"--out outputs/audio/examples/",
            "source": f"{meta.get('performer')} - {meta.get('title')}, bar {solo[index]['start_bar']}",
        })
    EXAMPLES.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} phrases to {EXAMPLES}; fill in each \"caption\".")


def cmd_try(args) -> None:
    feats = load_features()
    examples = load_examples()
    if not examples:
        print("note: no hand-written examples yet; captions will use the instructions alone\n")
    ids = args.ids or random.Random(args.seed).sample(
        [pid for pid, r in feats.items() if r["split"] == "train"], args.n)
    client = make_client()
    for pid in ids:
        words = feats[pid]["words"]
        print(f"== {pid}\n{describe(words)}")
        try:
            for c in caption_phrase(words, examples=examples, client=client, model=args.model):
                print(f"  - {c}")
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            print(f"  ! {e}")
        print()


def cmd_submit(args) -> None:
    from google.genai import types

    paths = _paths()
    out_path = paths["captions"]
    feats = load_features()
    examples = load_examples()
    if not examples and not args.no_examples:
        sys.exit(f"no hand-written captions in {EXAMPLES}: run pick-examples and write them "
                 "(or pass --no-examples)")
    done = load_done(out_path)
    todo = [pid for pid, r in feats.items()
            if pid not in done and (not args.splits or r["split"] in args.splits)]
    if not todo:
        print("every phrase already has captions")
        return

    system = build_system(examples)
    client = make_client()
    n_in = client.models.count_tokens(
        model=args.model, contents=system + "\n\n" + build_prompt(feats[todo[0]]["words"])
    ).total_tokens
    print(f"{len(todo)} phrases to caption with {args.model}; ~{n_in} input tokens each "
          f"(~{n_in * len(todo) / 1e6:.1f}M total), plus thinking and output.")
    if not args.yes:
        sys.exit("re-run with --yes to submit (batch jobs bill at 50% of standard prices)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.parent / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else []
    chunk_size = max(1, args.max_active_tokens // n_in)
    n_chunks = -(-len(todo) // chunk_size)
    print(f"{n_chunks} batch job(s) of up to {chunk_size} phrases "
          f"(~{chunk_size * n_in / 1e6:.1f}M tokens each)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, start in enumerate(range(0, len(todo), chunk_size)):
        chunk = todo[start:start + chunk_size]
        src_path = out_path.parent / f"batch_input_{stamp}_{i}.jsonl"
        with open(src_path, "w", encoding="utf-8") as f:
            for pid in chunk:
                f.write(json.dumps({"key": pid,
                                    "request": request_body(feats[pid]["words"], system)}) + "\n")
        uploaded = client.files.upload(file=src_path, config=types.UploadFileConfig(
            display_name=src_path.stem, mime_type="jsonl"))
        batch = create_when_room(client, args.model, uploaded.name, f"captions-{stamp}-{i}",
                                 state, state_path, out_path)
        state.append({"id": batch.name, "n": len(chunk), "model": args.model,
                      "collected": False})
        state_path.write_text(json.dumps(state, indent=2))
        src_path.unlink()
        print(f"submitted {batch.name} ({len(chunk)} phrases, job {i + 1}/{n_chunks})")
    print(f"batch ids in {state_path}; run `collect --wait` to fetch the rest")


def _is_quota_error(e: Exception) -> bool:
    return getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)


def create_when_room(client, model: str, src: str, name: str, state: list, state_path: Path,
                     out_path: Path):
    """Create a batch job, waiting (and collecting finished jobs) while the account's
    active-batch token limit is full."""
    from google.genai import errors

    while True:
        try:
            return client.batches.create(model=model, src=src, config={"display_name": name})
        except errors.ClientError as e:
            if not _is_quota_error(e):
                raise
            if not collect_finished(client, state, state_path, out_path):
                raise SystemExit(f"quota error with no batch jobs running, so one job is too big "
                                 f"for the limit: lower --max-active-tokens\n{e}")
            print(f"  active-batch limit full; waiting for a running job "
                  f"(checking every {POLL_SECONDS}s)")
            time.sleep(POLL_SECONDS)


def write_results(raw: bytes, out_path: Path) -> tuple[int, dict]:
    """Append validated captions from a batch output file; return (n written, failure counts)."""
    done = load_done(out_path)
    ok, failed = 0, {}
    with open(out_path, "a", encoding="utf-8") as f:
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row.get("key")
            try:
                if "error" in row:
                    raise ValueError(f"error {row['error'].get('code', '')}".strip())
                caps = parse_captions(row["response"])
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                failed[str(e)[:60]] = failed.get(str(e)[:60], 0) + 1
                continue
            if pid not in done:
                f.write(json.dumps({"phrase_id": pid, "captions": caps}) + "\n")
                done.add(pid)
                ok += 1
    return ok, failed


def collect_finished(client, state: list, state_path: Path, out_path: Path) -> list[str]:
    """Write the results of every finished, uncollected batch job; return the ids of jobs still
    running."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    running = []
    for entry in [e for e in state if not e["collected"]]:
        if not entry["id"].startswith("batches/"):
            continue  # left over from the old setup
        batch = client.batches.get(name=entry["id"])
        if batch.state.name not in DONE_STATES | FAILED_STATES:
            running.append(entry["id"])
            continue
        if batch.state.name in FAILED_STATES:
            print(f"{entry['id']}: {batch.state.name}"
                  + (f" ({batch.error.message})" if batch.error else "")
                  + "; its phrases will be resent by the next submit")
        else:
            ok, failed = write_results(client.files.download(file=batch.dest.file_name), out_path)
            print(f"{entry['id']}: {ok} captioned" + (f", failed: {failed}" if failed else ""))
        entry["collected"] = True
        state_path.write_text(json.dumps(state, indent=2))
    return running


def cmd_collect(args) -> None:
    out_path = _paths()["captions"]
    state_path = out_path.parent / "batches.json"
    if not state_path.exists():
        sys.exit(f"no {state_path}: nothing submitted yet")
    state = json.loads(state_path.read_text())
    client = make_client()
    while (running := collect_finished(client, state, state_path, out_path)) and args.wait:
        time.sleep(POLL_SECONDS)
    if running:
        for job in running:
            print(f"{job}: still running")
        print(f"{len(running)} batch(es) not finished; run collect again (or with --wait)")
    else:
        print(f"all batches collected -> {out_path}; re-run submit to retry any failures")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pick-examples", help="write varied phrases to examples.json to caption by hand")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite written examples")
    p.set_defaults(func=cmd_pick_examples)

    p = sub.add_parser("try", help="caption a few phrases now and print them")
    p.add_argument("ids", nargs="*", help="phrase ids (default: random training phrases)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=MODEL)
    p.set_defaults(func=cmd_try)

    p = sub.add_parser("submit", help="send uncaptioned phrases to the Gemini Batch API")
    p.add_argument("--splits", nargs="*", choices=("train", "val", "test"),
                   help="only these splits (default: all)")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--yes", action="store_true", help="actually submit (otherwise just estimate)")
    p.add_argument("--no-examples", action="store_true", help="allow submitting without examples")
    p.add_argument("--max-active-tokens", type=int, default=MAX_ACTIVE_TOKENS,
                   help="input tokens per batch job (keep under your tier's active-batch limit)")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("collect", help="fetch finished batches into captions.jsonl")
    p.add_argument("--wait", action="store_true", help="poll until every batch has ended")
    p.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
