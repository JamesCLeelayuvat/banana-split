# Jazz solo generator: text and chord conditioning design doc

Nov 18, 2026 · @James

## Overview

We will build a Transformer that writes a jazz solo line (symbolic, note by note) over a given set of chord changes, steered by a plain-English description such as "a dense, mostly descending bebop line in the upper register." It is an encoder-decoder model: a frozen pretrained text encoder and a small chord encoder feed a decoder that generates note tokens and reads both through cross-attention.

This answers the feedback that chords alone are not interesting enough: the description adds a second, controllable input, and we can measure whether the model actually listens to it.

Success looks like three things:

1. **Follows the changes.** Generated solos land on chord tones on strong beats more often than an unconditioned baseline.
2. **Follows the description.** Changing one word ("low register" to "high register", "sparse" to "dense") moves the matching measured feature in the right direction.
3. **Sounds like jazz.** In a blind listening test, musicians rate our solos' fit to the chords and to the description above the baselines.

## Scope

The core deliverable is symbolic: the model outputs MIDI, and audio is only a rendering step for demos.

**In scope**

- Monophonic solo lines in 4/4, generated phrase by phrase (4 or 8 bars)
- Inputs: chord changes, tempo, a text description
- Training data from the Weimar Jazz Database and Filosax, with synthetic captions
- MIDI output rendered to audio with a soundfont, over a simple backing, for the demo and listening test

**Out of scope**

- Generating audio directly, and the earlier Demucs stem-splitting pipeline
- Expressive timing and dynamics (swing is applied at playback, not learned)
- Chords, comping, or more than one instrument at once
- Full-length solos over a whole tune form (we stitch phrases if time allows)

**Stretch goals**

- Pretraining on MidiCaps before fine-tuning on jazz
- A contrastive (CLIP-style) text–solo embedding for retrieval
- Multi-chorus generation that conditions on the previous phrase

## Data

We train on roughly 700 real jazz solos cut into phrases, which gives several thousand training examples after augmentation. No dataset of solos paired with descriptions exists, so the text half is built in the next section.

| Source | What it gives us | Size | Access |
| --- | --- | --- | --- |
| Weimar Jazz Database | Transcribed solos by famous players, with chord changes, beat grid, and metadata (performer, instrument, style, tempo class, key) | \~450 solos (the Park paper uses 453) | Free download |
| [Filosax](https://dave-foster.github.io/filosax) | Saxophone improvisations over 48 standard backing tracks, with note-level annotations | 5 players × 48 tunes = 240 improvisations | Request access on Zenodo; backing audio bought separately (we only need the notes) |
| [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) (stretch) | General MIDI files with captions, for pretraining | 168,407 files | Free, Creative Commons |

**Preprocessing steps**

1. Load Weimar from its SQLite database and export each solo as notes (pitch, onset in beats, duration) plus per-beat chords.
2. Load Filosax through the `mirdata` loader. Check whether chord symbols ship with it; if not, type in the 48 lead-sheet progressions by hand (about 2 hours of work).
3. Normalize chord symbols to one vocabulary: 12 roots × about 9 qualities (maj7, 6, 7, m7, m6, m7b5, dim7, 7alt, sus).
4. Quantize onsets to 12 ticks per beat, so both straight 16ths and eighth-note triplets fit.
5. Cut each solo into 4-bar phrases on bar lines, sliding 2 bars at a time; drop phrases that are mostly rest.
6. Augment by transposing each phrase and its chords by −5 to +6 semitones, rejecting notes outside the instrument's range.

**Splits.** 80 / 10 / 10 split **by solo**, not by phrase, so overlapping phrases from one solo never land on both sides. Hold out a few whole tunes as an extra test of unseen changes.

## Captions

Every phrase gets a description written by an LLM from features we compute, following the recipe MidiCaps used. We also write a small set of real descriptions ourselves for testing only.

**Step 1: compute features for every phrase.** Store them as JSON next to the phrase; the evaluation reuses them.

| Feature | How to compute | Words it becomes |
| --- | --- | --- |
| Register | Mean MIDI pitch, relative to the instrument's range | low / middle / high |
| Range | Highest minus lowest pitch | narrow / wide |
| Density | Notes per beat | sparse / moderate / dense |
| Rhythmic grid | Share of 16ths and of triplets | eighth-note lines / sixteenth runs / triplet-heavy |
| Contour | Share of descending steps; overall start-to-end direction | descending / ascending / arching |
| Interval size | Mean absolute interval; share of leaps over a 4th | stepwise / intervallic |
| Harmonic fit | Share of notes that are chord tones; chromatic approach notes | inside / chromatic / outside |
| Space | Share of the phrase that is rest | busy / spacious |
| Context | From metadata: style, tempo class, instrument | "bebop", "up-tempo", "tenor" |

Set the cut-offs for each word at the training set's thirds, so every word covers a real share of the data.

**Step 2: write 20–30 example captions by hand.** Pick varied phrases and describe them the way a player would. These teach the LLM our vocabulary.

**Step 3: caption the rest with an LLM.** Give it the feature words, the metadata, and the examples, and ask for 3 differently worded captions per phrase. Spot-check 50 against the audio before using them.

**Step 4: build a human test set.** Two of us each describe the same 50 test phrases independently, without seeing the features. These are never used for training.

**Caption dropout.** During training, replace the caption with an empty one 15% of the time. This lets the model work without text and enables classifier-free guidance at inference.

## Representation

Notes use REMI-style event tokens with absolute pitch plus transposition augmentation. Chords get their own token sequence, one per beat. Text uses the pretrained encoder's own tokenizer.

**Solo tokens (decoder output).** Each note is three tokens, with bar markers between bars:

```
BAR  POS_0  PITCH_67  DUR_6   POS_6  PITCH_65  DUR_6   POS_12  PITCH_64  DUR_3 ...
```

- `POS_k`: where the note starts in the bar, in ticks (12 per beat, 48 per bar)
- `PITCH_p`: MIDI pitch, limited to the instruments' range (about 50 values)
- `DUR_d`: length in ticks, capped at 2 bars
- Rests are implied by gaps between positions
- Special tokens: `BOS`, `EOS`, `PAD`

The vocabulary is about 200 tokens, and a 4-bar phrase is usually 60–150 tokens. The `miditok` library implements REMI, so we don't write the tokenizer from scratch.

**Why absolute pitch, given the Park paper's problem?** Pitch here is always read alongside the chord, so the model can learn "the 3rd of the current chord" from the pair. Transposition augmentation means every lick is seen in many keys. Interval tokens avoid the problem too, but one wrong interval shifts every note after it. BPE on top of REMI is an ablation, not the default.

**Chord tokens (chord encoder input).** One token per beat, combining root and quality (12 × 9 = 108 tokens, plus `NO_CHORD`). A 4-bar phrase is 16 chord tokens. The decoder also gets the chord token for the beat it is currently on, added to its input (see Model architecture).

**Text (text encoder input).** The caption, tokenized by whichever pretrained text model we pick.

## Model architecture

This is the encoder-decoder Transformer from "Attention Is All You Need," with two encoders instead of one. The decoder writes note tokens and, in every layer, reads the chords and the description through one shared cross-attention.

&#91;embedded content: model architecture · 2 encoders, 1 decoder\]

The text and chord encodings are joined into one memory sequence, with a learned segment embedding marking which positions are chords and which are text. Each decoder layer's cross-attention takes its queries from the notes written so far and its keys and values from that memory.

**Two signals that help the model follow the changes.** Each decoder input token gets three embeddings added together: the token itself, its position in the bar, and the chord on the current beat. Cross-attention still sees the whole progression, so the model can anticipate the next chord, but it never has to search for the current one.

| Part | Full configuration | Small configuration (CPU) |
| --- | --- | --- |
| Text encoder | Pretrained T5-base or CLAP text branch, frozen | Small sentence-embedding model, frozen |
| Text projection | Linear to d\_model | Linear to d\_model |
| Chord encoder | 2 layers, d\_model 512, 8 heads | 2 layers, d\_model 256, 4 heads |
| Decoder | 6 layers, d\_model 512, 8 heads, feed-forward 2048 | 4 layers, d\_model 256, 4 heads, feed-forward 1024 |
| Max length | 512 note tokens | 256 note tokens |
| Trainable parameters | about 30M | about 5M |

Build it from PyTorch's `nn.TransformerDecoder` rather than from scratch: it already has causal masking and cross-attention.

## Training

The model trains with plain next-token cross-entropy and teacher forcing. The text encoder stays frozen; everything else learns from scratch.

| Setting | Starting value | Notes |
| --- | --- | --- |
| Optimizer | AdamW, weight decay 0.01 |  |
| Learning rate | 3e-4, 1,000 warmup steps, cosine decay | Lower to 1e-4 if loss spikes |
| Batch | 32 phrases | Pad to the longest in the batch |
| Dropout | 0.1, raise to 0.2 if overfitting | The dataset is small |
| Label smoothing | 0.1 |  |
| Caption dropout | 15% empty caption | Enables classifier-free guidance |
| Stopping | Early stop on validation loss, patience 5 epochs | Keep the best checkpoint |

**Order of runs**

1. Overfit a single batch to confirm the pipeline is wired correctly (loss should go near zero).
2. Train the unconditioned baseline (decoder only).
3. Train the chords-only model.
4. Train the full model (chords + text).
5. Stretch: pretrain the decoder on MidiCaps melody tracks, then fine-tune steps 3–4.

**Compute.** The full model is small enough for one cloud GPU (a Colab or class-provided T4 is fine). If we only have CPUs, use the small configuration in Model architecture and fewer augmented keys. Log every run (config, data version, validation loss) in one shared sheet or Weights & Biases project.

## Inference

At demo time, a user types chord changes ("Dm7 G7 | Cmaj7 | ..."), a tempo, and a description, and gets back a MIDI solo plus an audio rendering.

1. **Parse the changes** into per-beat chord tokens with the same vocabulary as training.
2. **Encode** the description (text encoder) and the chords (chord encoder) once.
3. **Sample** note tokens one at a time: temperature about 0.9, nucleus (top-p) 0.9.
4. **Apply classifier-free guidance** on the text: run the decoder with and without the caption and push the output toward the captioned one. Start with a guidance scale of 2 and tune it by ear and by the control metrics.
5. **Mask impossible tokens** while sampling: a `POS` must come after the previous one in the bar, a `PITCH` must follow a `POS`, pitches stay in the instrument's range. This removes a whole class of broken output for free.
6. **Stop** at the requested number of bars. For longer solos, generate the next phrase with the previous bars as a prompt.
7. **Render**: tokens to MIDI with `miditok`/`pretty_midi`, swing applied to eighth notes at playback, then audio with FluidSynth and a tenor sax soundfont over a simple bass, drums and piano backing.

## Evaluation

The headline result is a control test: change one word in the description and show the matching measured feature moves, while chord fit stays above the baselines.

**Models compared**

| Model | Inputs | What it shows |
| --- | --- | --- |
| Unconditioned | none | Floor: generic jazz-like lines |
| Chords only | chords | What chords alone buy |
| Full model | chords + text | Our system |
| Real phrases | — | Ceiling, from the test set |

**Metrics**

- **Validation loss / perplexity** for each model on the test split.
- **Chord fit:** share of notes on strong beats that are chord tones, and share that are approach notes resolving to a chord tone.
- **Description control:** generate 50 phrases per wording (e.g. "low register" vs "high register") over the same changes, compute the feature, and report the difference and how often the output lands in the requested bin. Repeat for register, density, contour, and harmonic fit.
- **Human-caption test:** the same control test using the 50 descriptions we wrote ourselves, to check the model isn't only reacting to template wording.
- **Diversity and copying:** pitch-class and rhythm entropy across samples, and the longest note sequence shared with any training phrase (flags memorization).

**Listening test.** 8–10 musicians from school, blind. Each hears pairs and rates 1–5 for fit to the chords and fit to the description; real phrases are mixed in. Render everything with the same soundfont so sound quality doesn't give it away.

**Ablations** (only if time allows): BPE on solo tokens vs none; one shared cross-attention vs separate ones for chords and text; one caption per phrase vs three paraphrases; guidance scale 1 vs 2 vs 3.

## Two-week plan for two people

The work splits into two tracks that run side by side for 14 days. Person A owns data, captions, and evaluation. Person B owns the model and generation. They meet at three handoffs, and B is never blocked: B builds and tests on fake data until A's real data lands on day 3.

**Cuts to fit two weeks**

- Weimar only. Request Filosax access on day 1 anyway; if it arrives by day 3, A adds it on day 4, otherwise skip it.
- No MidiCaps pretraining, and no contrastive loss.
- One ablation only: guidance scale. Everything else in Evaluation's ablation list is cut.
- Listening test with 6–8 musicians instead of 8–10.
- Use the small model configuration unless a GPU is confirmed on day 1.

**How you stay in sync**

- Day 1 together: agree the file formats before splitting up (one JSON file per phrase with notes, per-beat chords, and metadata; the token vocabulary; the chord vocabulary). This is what lets both tracks start at once.
- A 15-minute check-in every day: what landed, what's next, what's blocking.
- Each handoff is a file in the shared repo, not a message. B pulls it and moves on.

&#91;embedded content: two-week plan · 2 people, 3 handoffs\]

Arrows show which way each handoff goes: A gives B phrases (end of day 3) and captions (end of day 6); B gives A generated samples to evaluate (end of day 10).

**Person A: data, captions, evaluation**

- [ ] Day 1: Agree file formats with B. Request Filosax access. Write the Weimar loader.
- [ ] Day 2: Chord-symbol normalizer; quantize onsets to 12 ticks per beat.
- [ ] Day 3: Phrase cutter, solo-level split, transposition augmentation. **Hand off H1: phrase files.**
- [ ] Day 4: Feature extractor for every caption feature (add Filosax if access came through).
- [ ] Day 5: Write 25 example captions by hand; draft the captioning prompt.
- [ ] Day 6: LLM captioning, 3 per phrase; spot-check 30 against the audio. **Hand off H2: captions.**
- [ ] Day 7: Write descriptions for the 50 human test phrases (B writes their own set for the same phrases).
- [ ] Day 8: Chord-fit and control-test scripts; sanity-check them on real phrases.
- [ ] Day 9: Diversity and copying scripts.
- [ ] Day 10: Build the listening-test form; recruit 6–8 musicians.
- [ ] Day 11: Run all metrics on B's samples (H3).
- [ ] Day 12: Run the listening test; make results tables and figures.
- [ ] Days 13–14: Write the data, captions, and results sections; rehearse the presentation.

**Person B: model and generation**

- [ ] Day 1: Agree file formats with A. Set up repo and environment; confirm GPU access.
- [ ] Day 2: REMI tokenizer with `miditok`; check tokens decode back to the same MIDI. Decoder-only model and training loop; overfit one batch of fake data.
- [ ] Day 3: Rendering script: MIDI to audio with swing, soundfont, simple backing.
- [ ] Day 4: Train the unconditioned baseline on H1.
- [ ] Day 5: Chord encoder + current-chord embedding; start the chords-only run.
- [ ] Day 6: Frozen text encoder, projection, cross-attention; test with placeholder captions.
- [ ] Day 7: Train the full model on H2 with caption dropout. Write your 50 human test descriptions (about an hour).
- [ ] Day 8: Sampling: token masks, temperature, top-p, classifier-free guidance.
- [ ] Day 9: Listen together with A; if the model ignores the text, switch on the fallback (feature values as an embedding) and retrain.
- [ ] Day 10: Generate evaluation sets from all three models and the demo solos. **Hand off H3: samples.**
- [ ] Day 11: Guidance-scale ablation (1, 2, 3); render the listening-test clips.
- [ ] Day 12: Code freeze; help run the listening test.
- [ ] Days 13–14: Write the model and training sections; build slides and the live demo.

**Where it can slip.** Day 9 is the riskiest day: if the text isn't steering the output, the retrain eats day 10. Days 11–12 have a little slack for that because the ablation is optional.

## Risks and open questions

The biggest risk is that the model ignores the text and leans entirely on the chords. The control test exists to catch exactly that.

| Risk | Sign it's happening | Fallback |
| --- | --- | --- |
| Model ignores the text | Control test shows no shift between wordings | Raise guidance scale; raise caption dropout; feed the feature values directly as a learned embedding (the attribute route) |
| Overfitting: \~700 solos is small | Validation loss rises early; copied phrases in samples | Smaller model, more dropout, more transposition, MidiCaps pretraining |
| Captions are circular | Model scores well on synthetic captions, poorly on human ones | Report both; say plainly in the write-up that training captions are synthetic |
| Filosax access delayed or no chords | No approval by end of week 1 | Train on Weimar alone; Filosax becomes a stretch goal |
| CPU-only compute | Runs take too long to iterate | Small configuration, fewer augmented keys, 4-bar phrases only |
| Chord symbol mismatch between datasets | Many unparseable symbols | Map rare qualities to the nearest of the 9; log what was mapped |

**Open questions**

- What is the due date, and is a live demo expected?
- Does Prof. Walsh want audio output, or is MIDI rendered to audio enough?
- Which pretrained text encoder: T5, CLAP's text branch, or a small sentence-embedding model? Pick the one that runs on our compute.
- Should the performer's name be allowed in descriptions ("in the style of Coltrane")? It's a strong signal, but it confounds the register issue noted in the Park paper.
