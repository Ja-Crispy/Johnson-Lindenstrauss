# Carrier Decomposition Across Three Dense Models

**Status (2026-04-30):** Cross-model carrier-decomposition with dark-subspace test (Cancedda 2024 framework). n=3 case studies: Qwen2.5-1.5B, Gemma-3-1B-IT, Mistral-7B-Instruct-v0.3. Three different patterns of how the residual-stream carrier interacts with the unembedding spectrum. **The cleanest discriminator across families is whether the carrier is dark-aligned or bright-aligned in the unembedding's right-singular basis.**

This is a follow-up to:
- `docs/layer_structure_pilot.md` (n=1 pilot)
- `docs/layer_structure_carrier_test.md` (n=1 mechanism test on Qwen2.5-1.5B)

This document supersedes the prior n=3 cross-model write (which did not include the dark-subspace test). It now incorporates: extended (multi-rank) cosine measurement, hidden-state archival for downstream tests, unembedding spectral basis, and Cancedda 2024 positioning.

---

## Headline (one paragraph)

Three dense transformer families show three structurally different residual-stream regimes. **All have a low-rank persistent carrier subspace** (top-PC alignment ≥ 0.99 in the inner band; top-1 captures 84–98% of variance). **All show d_eff_c recovery from ~1 to tens or hundreds** when the carrier subspace is removed. But the carrier's **identity differs** along two axes:

1. **Dark-vs-bright alignment in the unembedding spectrum.** Qwen's carrier sits in W_unembed's tail at 5.4× the random baseline (partially dark — Cancedda-like sink). Mistral's at 1.7× (mildly dark). **Gemma's at 0.1× — actively *anti*-dark, in the bright subspace** that the unembedding reads.
2. **Position localization at first-token / position 0.** Qwen and Mistral show massive activations at position 0 (max/median 70–4000×). Gemma does not (max/median ≈ 1.7).

These two axes correlate: position-0-localized carriers are dark-aligned; the position-uniform carrier is bright-aligned. The cosine confound also tracks: removing the dark-aligned (sink-style) carrier barely moves cosine, but removing the bright-aligned (Gemma) carrier unflattens cosine by 0.1–0.34 across most middle transitions.

Cancedda 2024 demonstrated the dark-signal sink mechanism on **LLaMa2 only**. Our cross-family test extends his framework qualitatively to Qwen and Mistral (more weakly), and produces a clean counterexample in Gemma whose carrier mechanism is mechanistically distinct.

## Setup

All three runs use:
- 8 wikitext sequences × 512 tokens, fp32 forward + analysis
- Auto-detected collapsed band (`d_eff_c < 5.0`, longest contiguous run)
- Position-localization at three layers spaced through the detected band
- Multi-rank carrier-removed cosine, k ∈ {1, 2, 3, 5}
- Hidden states + W_unembed spectral basis archived to .npz for downstream tests
- Code: `scripts/analyze_carrier_structure.py`, `scripts/plot_carrier_structure.py`

**Important methodological caveat: BOS handling.** We use `tokenizer.encode(text, add_special_tokens=False)`, which means **no BOS token is prepended** to the input. The "position 0" we report is the first wikitext content token, not a BOS. Cancedda's analysis assumed LLaMa2's default tokenizer (which auto-prepends `<s>`), so his "BOS sink" is anchored on a deliberate special token. Our equivalent finding ("first-position sink") is similar in geometry but different in token identity. Whether the Qwen/Mistral position-0 sink moves to a BOS token if one were prepended is **untested in this run**.

## Cross-model summary

| | Qwen2.5-1.5B | Gemma-3-1B | Mistral-7B |
|---|---:|---:|---:|
| n_layers (incl. embed) | 29 | 27 | 33 |
| hidden_size | 1536 | 1152 | 4096 |
| Detected band | L2–L26 | L7–L22 | L1–L26 |
| Band length | 25 | 16 | 26 |
| Persistence (band, unc / cen) | 0.9995 / 0.9995 | 0.9942 / 0.9978 | 0.9242 / 0.9251 |
| Persistence (inner, unc / cen) | 0.9999 / 0.9999 | 0.9957 / 0.9985 | 0.9991 / 0.9996 |
| Top-1 var explained | 97.85% | 98.03% | **84.42%** |
| Top-2 var explained | 99.22% | 98.32% | 89.39% |
| Top-5 var explained | 99.41% | 98.68% | 94.29% |
| Position localization peak | pos 0 | pos 460 | pos 0 |
| Position max/median (mid-band) | **2403** | **1.7** | **252** |
| Δ_⊥ band fraction (mean) | 99.7% | 61.9% | 99.3% |
| **Carrier in W_unembed tail-100** | **5.4× random** | **0.1× random** | **1.7× random** |
| Cosine gap k=1 / k=2 / k=5 (band mean) | 0.0004 / 0.014 / 0.026 | **0.136** / 0.142 / 0.149 | 0.0012 / 0.002 / 0.006 |

## What's universal

1. **Persistent low-rank carrier in every model.** Inner-band persistence is 0.999+ in all three.
2. **d_eff_c recovery in every model.** After projecting off rank-1 (Qwen, Gemma) or rank-2 (Mistral), middle layers go from `d_eff_c ≈ 1` to tens or hundreds.
3. **Off-carrier updates dominate in sink-style models.** Qwen and Mistral both have Δ_⊥ at 99%+; the model is computing perpendicular to the carrier.

## What varies

### Carrier rank: rank-1 (Qwen, Gemma) vs rank-2/3 (Mistral)

Mistral's recovery curve at L7 jumps from 1.40 (k=1) to 74.78 (k=2). Top-1 captures only 84% of stacked variance vs Qwen/Gemma's 98%. Mistral's carrier subspace is genuinely 2-dimensional, not a single direction.

### Dark-subspace alignment: the new cleanest discriminator

For each model, we computed the squared projection of the rank-1 carrier `c_1` onto the bottom-100 right singular vectors of W_unembed. Random direction baseline = 100 / hidden_size.

| Model | hidden_size | Random tail-100 | Carrier in tail-100 | Dark multiplier |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1536 | 6.5% | 35.3% | **5.4× random** |
| Gemma-3-1B | 1152 | 8.7% | **0.88%** | **0.1× random (anti-dark)** |
| Mistral-7B | 4096 | 2.4% | 4.21% | **1.7× random** |

Qwen's carrier is partially dark — the unembedding mostly ignores it, consistent with Cancedda's sink-as-dark-signal picture. Mistral's is mildly dark — Cancedda's framework applies weakly. **Gemma's carrier is in the bright subspace**, the one the unembedding actually reads to produce logits. Whatever Gemma's carrier is for, it isn't a Cancedda-style attention sink.

### Cosine confound: response to carrier removal at k=1, 2, 3, 5

| Model | Mean band gap k=1 | k=2 | k=3 | k=5 |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.0004 | 0.014 | 0.018 | 0.026 |
| Gemma-3-1B | **0.136** | 0.142 | 0.144 | 0.149 |
| Mistral-7B | 0.0012 | 0.002 | 0.005 | 0.006 |

**Mistral's cosine is most immune to subspace removal** — even with rank-5 of the carrier projected off, cos sim moves <0.01. Gemma's cosine is most responsive — rank-1 removal already drops ~0.14, and higher ranks barely add. Qwen sits between: rank-1 doesn't move it, but rank-5 starts to (~0.03).

Notable: the "sink-style → cosine immune to carrier removal" rule holds robustly across rank sweeps, sharpening the n=3 dichotomy from the prior writeup.

## Where this puts us relative to Cancedda 2024

Cancedda showed in LLaMa2 7B/13B/70B that:
- The bottom 5% of W_unembed's right singular vectors form "U-dark"
- BOS attention sinks correspond to large-norm vectors lying entirely in U-dark, generated in specific MLP layers (L3 for 7B/13B)
- Sink-preserving spectral filters can suppress 25% of singular values with minimal NLL increase

Our test on three other families shows:
- The phenomenon **partially** generalizes to Qwen and Mistral. Their carriers are dark-aligned, but not "entirely in U-dark" the way Cancedda found for LLaMa2. Qwen's carrier puts 35% of energy in bottom-100 (= 6.5% of dimensions), whereas Cancedda's LLaMa2 sinks were essentially 100% in the bottom 5%. So Cancedda's framework applies in spirit but the carriers are noisier / more spread on Qwen.
- The phenomenon **does not** generalize to Gemma. Gemma's carrier is in the bright subspace, opposite to U-dark.

**Our potentially-new contribution after Cancedda:**
1. **Cross-family extension**: Cancedda's mechanism partially holds beyond LLaMa2. Qwen and Mistral confirm at weaker strengths.
2. **Counterexample (Gemma)**: an existing dense transformer with persistent low-rank carrier that is *not* a dark-signal sink. Its carrier is bright-aligned and unflattens adjacent-layer cosine when removed. This is genuinely outside Cancedda's framework.
3. **Methodological connection**: the carrier-vs-cosine relationship (whether removing carrier exposes layer variation) tracks the dark/bright alignment, which connects two literatures that were previously separate (Cancedda's spectral sinks; Curse-of-Depth-style cosine-similarity layer analysis).

## What this isn't

- **Not** a layer-utility metric.
- **Not** a "we found the right metric" claim. Multiple metrics (d_eff, cos sim, dark alignment, position localization) each capture a different aspect.
- **Not** universal generalization. n=3 across three families on Apple Silicon. No Llama, no Phi, no MoE, no hybrid.
- **Not** a BOS test. Our setup explicitly omits BOS (`add_special_tokens=False`); we report position-0-localization, which is *similar to* but not identical to a BOS sink in token identity. Whether Qwen/Mistral position-0 sinks shift to a BOS token under a default tokenizer is untested.
- **Not** a causal demonstration. We measure structure; we don't ablate the carrier and verify performance impact.

## Multi-rank carrier dark alignment (refining the rank-1 picture)

Rank-1 alignment with W_unembed tail-100 (above) is partial. The full top-5 carrier subspace is more dark-aligned in all three models, but the *ordering* differs by family:

| Model | rank-1 (tail-100) | rank-2 | rank-3 | rank-3 in tail-20 |
|---|---:|---:|---:|---:|
| Qwen | **5.4×** | 4.3× | 3.8× | 6.5× |
| Gemma | 0.1× | 2.4× | 2.1× | 4.9× |
| Mistral | 1.7× | 3.2× | **7.7×** | **23.6×** |

Mistral's third carrier direction is essentially Cancedda's darkest direction — 23× random in the very tail. Gemma's secondary carrier dimensions are dark too, but its primary (rank-1) is bright. **All three models have dark components in their carrier subspace; only the ranking and concentration differ.**

## CKA baseline comparison

We compared adjacent-layer linear CKA (Kornblith 2019) against cos sim, raw and carrier-removed, on all three models. Code: `scripts/compute_cka.py`. Plots: `docs/img/cka_vs_cos_<model>.png`.

| Model | raw cos sim band | raw CKA band | k=1 cos gap | k=1 CKA gap |
|---|---|---|---:|---:|
| Qwen2.5-1.5B | 0.85-0.96 (varies) | **0.998-1.000 (saturated)** | 0.0004 | ~0.05 |
| Mistral-7B | 0.83-0.95 (varies) | **0.999-1.000 (hyper-saturated)** | 0.001 | ~0.005 |
| Gemma-3-1B | **0.97-0.99 (flat)** | 0.4-0.99 (varies dramatically) | 0.14 | tracks raw |

**CKA is not a fix.** On Qwen and Mistral (sink-style), raw CKA is *even more* saturated than cos sim — 1.000 across the entire middle band. On Gemma (bright-carrier), raw CKA naturally exposes layer-to-layer variation that cos sim flattens.

The mathematical intuition: CKA centers and uses Gram matrices. When the carrier is hyper-localized (Qwen/Mistral position-0 outliers), centering doesn't fully remove its effect on the Gram structure — CKA gets dominated. When the carrier is position-uniform (Gemma), centering effectively discounts it, and CKA's behavior tracks the perpendicular subspace.

This means **every standard adjacent-layer similarity metric has a carrier-dependent failure mode**. The right answer isn't "switch metrics" — it's structural decomposition of what's underneath.

## BOS-included rerun

Original results used `tokenizer.encode(..., add_special_tokens=False)` — no BOS token, so position 0 is the first content token. Cancedda's setup includes BOS. To test whether the sink we observed shifts to BOS when present:

| Model | BOS available? | Without BOS (peak/ratio) | With BOS (peak/ratio) | Dark alignment unchanged? |
|---|---|---|---|---|
| Qwen2.5-1.5B | **No (tokenizer has no BOS token)** | pos 0, ratio 4083 (L8) | n/a | n/a — Qwen has no BOS option |
| Gemma-3-1B | Yes (`<bos>`) | uniform, ratio 1.7 (L11) | **pos 0, ratio 10.4** (L9) | yes (0.88% / 0.10× random in both) |
| Mistral-7B | Yes (`<s>`) | pos 0, ratio 987 (L7) | **pos 0, ratio 2496 (L6)** | yes (1.72× → 1.81× random) |

Two clean findings:

1. **Qwen2.5's tokenizer literally has no BOS token.** Whatever sink Qwen has must be anchored on the first content token, not a special token. The "position-0 sink" framing is necessarily about first-position rather than special-token absorption.
2. **Carrier identity (dark/bright) is BOS-invariant.** Both Gemma and Mistral show the same dark-alignment fraction with and without BOS. What changes is the *intensity* of position-0 absorption — BOS, when present, absorbs more carrier energy than a content token would. This is consistent with Cancedda's BOS-as-sink mechanism for the sink-aligned models.

So BOS handling matters for sink intensity but not carrier dark/bright identity. The structural finding (dark vs bright carrier) is robust.

## Outstanding questions worth flagging

1. **What is Gemma's carrier doing?** Bright-aligned regardless of BOS. The unembedding actively reads from it. We have no story for what it computes.
2. **Causal ablation.** Forward-pass intervention removing the carrier subspace, with PPL / NLL evaluation. Gold-standard test for whether the carrier is load-bearing. Multi-hour, more involved.
3. **Llama-3.2-1B.** Cancedda used LLaMa2; LLaMa3 family has different training. Free run if it loads. Not done yet.
4. **Why does CKA saturate on sink-style models?** Centering should discount uniform features but it doesn't kill outlier-position features. Worth a small theoretical note.

## Where this could go (not committed)

- Read Cancedda end-to-end (we have the abstract + key claims via webfetch but not the full mechanistic detail).
- Implement CKA comparison as a separate analysis script reading the saved hidden states.
- Implement causal ablation as a separate test (Qwen most likely target; sink is most extreme there).
- BOS-included rerun on Qwen and Mistral.
- Extend to one more dense family (Llama).

## Figures

For each model, five figures (`docs/img/carrier_*_<model>.png`):

- `carrier_persistence_*` — cross-layer top-PC alignment heatmap
- `carrier_recovery_*` — `d_eff_c` per layer at original and carrier-removed-rank-{1,2,3,5,10,20}
- `carrier_decomposition_*` — per-transition Δ_∥/Δ_⊥ fractions and `d_eff_c(Δ_⊥)`
- `carrier_cosine_comparison_*` — multi-rank carrier-removed cos sim per transition (raw + k ∈ {1, 2, 3, 5})
- `carrier_position_localization_*` — per-position carrier coefficient

## Adjacency

This work uses the same toolkit as the K/V asymmetry memo (`docs/kv_asymmetry_memo.md`) but operates on a different object (residual-stream activations, not K/V tensors). The methodological reuse is the only link.
