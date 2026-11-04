Jazz Combo Stem Splitter

A from-scratch U-Net that separates a jazz combo recording into four stems — drums, bass, piano, and horns — trained with a scale-invariant SDR loss on the JazzSAMBA dataset.

Final project for [course name], [term]. Group: [Name A], [Name B], [Name C].

1. Overview

Most source separators are trained on pop/rock (MUSDB18-style: vocals, drums, bass, "other"). Jazz breaks those assumptions: upright bass, brushed drums, piano comping in the same register as the horn, and live-room bleed. We train a compact spectrogram-masking U-Net, implemented from its equations, on jazz-combo stems and ask:

Does an SI-SDR loss beat a plain spectrogram loss?
How much do skip connections matter, and why (via gradient-flow diagnostics)?
How much does performance drop on live recordings (with mic bleed) versus overdubbed studio takes?

We do not expect to beat large pretrained separators (e.g. BS-RoFormer, HT-Demucs); we report them as a reference ceiling, not a baseline.

2. How this meets the project requirements
#	Requirement	How we meet it
1	New mechanism	U-Net (encoder–decoder with transposed convolutions and skip connections) + SI-SDR loss — neither taught, both build on CNNs
2	From scratch + numerical check	Transposed conv vs. nn.ConvTranspose2d; SI-SDR vs. torchmetrics; STFT→iSTFT round trip
3	Explain the math	See §3: equations, parameter count, gradient flow
4	Real GPU training	PyTorch on Colab Pro / instructor GPU (CPU only for inference latency)
5	Real data, enough of it	JazzSAMBA multitrack stems; random remixing yields tens of thousands of training crops
6	Fair test	Train/val/test split by song; all takes of a song stay in one split
7	Baselines + hypotheses	Supervised NMF (non-deep), oracle ideal ratio mask (ceiling); hypotheses in §5
8	Diagnostics	Per-layer gradient norms (skip vs. no-skip), decoder feature maps, mask visualizations
9	Scope	~10–15 GPU-hours total; uv environment; no non-Python tools needed
3. Method
3.1 Masking pipeline

For a mixture waveform x:

X = STFT(x)                          # complex spectrogram, F × T
M_k = σ( f_θ(|X|) )_k                # one mask per stem k ∈ {drums, bass, piano, horns}, values in [0, 1]
ŝ_k = iSTFT( M_k ⊙ X )               # estimate reuses the mixture phase

f_θ is the U-Net; σ is a sigmoid applied per stem.

3.2 U-Net
Encoder: 4 levels, each two 3×3 conv + BatchNorm + ReLU, then stride-2 downsampling. Channels 32 → 64 → 128 → 256, bottleneck 512.
Decoder: mirror image using transposed convolutions (stride 2) to upsample, then concatenation with the matching encoder feature map (skip connection), then two 3×3 convs.
Output: 1×1 conv to 4 channels (one mask per stem).

A transposed convolution is the gradient (adjoint) of a convolution with respect to its input — we implement it directly from that definition and check it against PyTorch.

3.3 Loss: SI-SDR

For reference s and estimate ŝ:

α       = ⟨ŝ, s⟩ / ‖s‖²
SI-SDR  = 10 · log₁₀( ‖α s‖² / ‖α s − ŝ‖² )
L       = − mean_k SI-SDR(ŝ_k, s_k)

SI-SDR ignores overall gain, so the model is not penalized for getting loudness slightly wrong.

3.4 Gradient flow

L → SI-SDR (differentiable in ŝ) → iSTFT (linear) → elementwise product with X → sigmoid → U-Net. Skip connections give early encoder layers a short path from the loss, which we expect to show up as larger early-layer gradient norms (§6).

3.5 Parameter count

With 3×3 kernels, one conv has 9 · C_in · C_out weights. Encoder ≈ 9 × (1·32 + 32·32 + 32·64 + 64·64 + … + 512·512) ≈ 4.7M; decoder is similar, for ~9–10M total. A half-width variant (16 → 256) is ~2.5M and is our fallback if training is slow.

4. Data
Dataset	Use	Notes
JazzSAMBA (Long et al., 2026) — demo & links	Main train/val/test	76 standards, 8 musicians; drums, bass, piano, trumpet, sax (alto or tenor). ~8.1 h preferred takes, ~16.1 h with alternates. 50 songs overdubbed (clean stems), 26 recorded live (bleed). 48 kHz. TODO Week 1: confirm license and download.
MoisesDB (GitHub)	Optional pretraining / pop-vs-jazz comparison	240 tracks, ~14 h, mostly rock/pop; CC BY-NC-SA 4.0
MUSDB18 (link)	Fallback if JazzSAMBA is unavailable	150 tracks, ~10 h; 4 pop stems

Preprocessing: resample to 24 kHz mono; STFT with n_fft=1024, hop=256, Hann window; 4-second crops.

Augmentation (training only): random remixing — draw each stem from a random crop/song, apply a random per-stem gain, sum to a mixture.

Splits: by song, never by take or crop. Overdubbed alternate takes are built on the same preferred takes, so splitting by take would leak. Remixing never crosses splits. The live (synchronous) songs are held out entirely as a second test set.

Excluded: the Jazz Trio Database — its stems were produced by a separation model, so they are not ground truth.

5. Baselines and experiments

Baselines

Supervised NMF (scikit-learn): learn a spectral dictionary per stem from training audio, decompose test mixtures, build Wiener-style masks. Non-deep baseline.
Oracle ideal ratio mask: computed from the true stems; an upper bound for any magnitude-masking model.
Reference only: published pretrained-separator scores on JazzSAMBA from the dataset paper.

Experiments (each hypothesis written before running)

ID	Conditions	Hypothesis
E1	L1 spectrogram loss vs. SI-SDR loss	SI-SDR loss gives higher test SI-SDR, because it optimizes the evaluation metric directly
E2	U-Net with vs. without skip connections	No-skip model is blurrier (worse on fast piano/horn lines) and its early encoder layers receive much smaller gradient norms
E3	Test on overdubbed vs. live songs	SI-SDR drops on live recordings because of mic bleed, most for piano and horns
E4 (stretch)	Train on MoisesDB only vs. JazzSAMBA only vs. both	Jazz-matched data beats pop-only on the jazz test set; combining helps most

Metric: SI-SDR (dB) per stem, reported per song as well as averaged, since the test set is small.

6. Diagnostics
Gradient norms per layer via backward hooks, logged to MLflow, for skip vs. no-skip (E2).
Feature maps from decoder levels on a short excerpt with a clear horn line.
Mask visualizations: predicted vs. oracle masks per stem.
7. Compute

~0.5–1 GPU-hour per training run on a T4-class GPU; ~10–15 runs including ablations → ~10–15 GPU-hours. Training requires a GPU. CPU is used for inference latency measurements only.

8. Setup
bash
# install uv: https://docs.astral.sh/uv/
uv sync                 # creates .venv from pyproject.toml
uv run mlflow ui        # experiment tracking at http://localhost:5000

Planned layout

data/            # downloaded datasets (git-ignored)
src/
  audio.py       # STFT / iSTFT helpers, resampling
  dataset.py     # song-level splits, random remixing
  unet.py        # U-Net and from-scratch transposed conv
  losses.py      # SI-SDR, L1 spectrogram loss
  nmf_baseline.py
  oracle.py      # ideal ratio mask
  train.py
  evaluate.py
  diagnostics.py # gradient-norm hooks, feature maps, mask plots
tests/           # numerical checks against PyTorch / torchmetrics
notebooks/       # figures for the report
9. Team roles

Each member owns one part and its math for the oral quiz.

Owner	Part	Equations to defend
A	Data pipeline, splits, remixing, NMF baseline, oracle mask, evaluation	NMF update rules, ratio masks
B	U-Net, from-scratch transposed conv, numerical checks, gradient-norm and feature-map diagnostics	Convolution / transposed conv, parameter count, gradient flow through skips
C	SI-SDR loss, training loop, MLflow, experiments E1–E4	SI-SDR and its gradient, masking pipeline

With two members, A and C merge.

10. Timeline
Week	A	B	C
1	Download JazzSAMBA, verify license, build song-level splits and remixer	From-scratch transposed conv, pass numerical check	uv env, MLflow, STFT and SI-SDR checks
2	NMF baseline, oracle mask	Full U-Net	First end-to-end training runs
3	Evaluation script, live-recording test (E3)	Skip ablation (E2), diagnostics	Loss comparison (E1), MoisesDB stretch (E4)
4	Real-record listening demo	Report: U-Net math	Report: SI-SDR math and results
11. Risks and fallbacks
JazzSAMBA license or download fails → switch to MUSDB18 (4 pop stems). Model and loss code are unchanged; only dataset.py changes.
Training too slow → half-width U-Net (~2.5M params) and shorter crops.
Small test set → report per-song scores and spread, avoid over-claiming.
12. Limitations and ethics
Limited coverage: 8 musicians and 76 standards. The model may not generalize to other players, instruments (e.g. guitar, vibes), or historical recordings like 1950s–60s jazz.
Ground truth only exists for studio and live JazzSAMBA takes. Results on commercial records are qualitative (listening only).
Misuse: separation tools can extract stems from copyrighted recordings. Our demo uses JazzSAMBA audio, whose musicians consented to public release.
13. References
Long et al., JazzSAMBA: A Synchronous and Asynchronous Multi-take Band Audio Dataset of Jazz Standards for Live Music Models, 2026. arXiv
Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015.
Jansson et al., Singing Voice Separation with Deep U-Net Convolutional Networks, ISMIR 2017.
Le Roux et al., SDR — Half-baked or Well Done?, ICASSP 2019.
Pereira et al., MoisesDB: A Dataset for Source Separation Beyond 4-Stems, ISMIR 2023.
Rafii et al., The MUSDB18 Corpus for Music Separation, 2017.
