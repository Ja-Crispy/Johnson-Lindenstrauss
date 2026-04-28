# Layer-Structure Spectral Decomposition: A Methodology Probe

**Status (2026-04-25):** Bounded pilot, n=1 model. Not a layer-utility metric. Not a paper. A short note responding to a public methodology critique with a small, concrete decomposition that exposes structure the classical metric hides.

---

## What this is

Muyu He recently argued ([twitter.com/HeMuyu0327](https://x.com/HeMuyu0327)) that adjacent-layer cosine similarity, as used in the *Curse of Depth* paper to measure how effectively LLMs "utilize" their layers, is misleading on three grounds: (1) angle conflates orthogonality and cancellation, (2) angle is direction-blind, (3) a few dominant dimensions can mask real change.

This pilot tests, on one dense model (Qwen2.5-1.5B), whether a small spectral metric bundle exposes structural distinctions that a single cos-sim number flattens. **It does.** Concrete disagreement examples are below.

This note does not claim to define "layer utility." It claims that adjacent-layer cosine similarity, treated as a single summary, conflates layer-behavior types that have visibly different spectral signatures.

## Setup

- Model: `Qwen/Qwen2.5-1.5B`
- Calibration: 8 wikitext sequences × 512 tokens = 4096 positions
- Object: post-block residual hidden states from `output_hidden_states=True`, fp16 forward then upcast to fp32 for analysis (29 hidden states: embedding + 28 transformer blocks)
- Code: `scripts/analyze_layer_structure.py`, `scripts/plot_layer_structure.py`
- Raw output: `results/layer_structure_qwen25_1.5b.json`
- Figure: `docs/img/layer_structure_qwen25_1.5b.png`

## Metric bundle

Per layer `l`:
- `d_eff_uc(h_l) = (Σλ)² / Σλ²` on the uncentered covariance — concentration including any dominant mean drift
- `d_eff_c(h_l)` on the centered covariance — variance structure beyond the mean

Per transition `l → l+1`, with `Δ_l = h_{l+1} - h_l`:
- `cos(h_l, h_{l+1})` — classical adjacent-layer angle, the baseline
- `||Δ_l|| / ||h_l||` — relative magnitude of the layer's contribution
- `d_eff_c(Δ_l)` — **spectrum of the layer's contribution**: low ⇒ focused single-direction, high ⇒ diffuse multi-direction
- `||Δ_perp|| / ||Δ_l||` — fraction of the contribution outside the top-`k` centered subspace of `h_l`, with `k = ⌈d_eff_c(h_l)⌉`. High ⇒ injects new directions; low ⇒ refines existing structure

The two key new metrics relative to the classical baseline are `d_eff_c(Δ_l)` and `||Δ_perp|| / ||Δ_l||`. They directly address Muyu's critiques #2 (direction-blindness) and #3 (dominant-direction masking).

## Findings

### 1. Three regimes in per-layer concentration, invisible to cos sim

The activation's participation ratio across the 28 transformer blocks splits into three regimes:

| Layers | d_eff_uc | d_eff_c | Interpretation |
|---|---:|---:|---|
| L0 (embed), L1 | 28–36 | 42–110 | rich, multi-direction representation |
| L2–L26 | ~1 | ~1 | norm-collapsed onto a dominant direction |
| L27–L28 | 1.8–2.2 | 13–57 | structure re-emerges in the centered view at the output boundary |

The middle-layer collapse is consistent with the well-documented "outlier feature" phenomenon in transformers (a small number of channels carry disproportionate L2 norm). What's worth noting here is that **the centered view at L27 reveals a d_eff of 57 underneath an uncentered d_eff of 2.2** — the same dominant-direction-masks-real-variance pattern that motivated this whole exercise. cos-sim cannot see this regime structure at all.

### 2. cos sim is flat; the layer's contribution-spectrum is not

Across L2 → L26, adjacent-layer cosine similarity stays in a tight band of **0.85–0.96**. By the cos-sim story, every middle layer is doing roughly "the same amount" of similar-magnitude refinement.

Across the same span, `d_eff_c(Δ_l)` ranges from **1 to 222** — a more than two-orders-of-magnitude spread. Some layers make focused single-direction changes; others make broad multi-direction changes. Same angular distance, structurally different behavior.

Three concrete disagreement examples:

| Transition | cos sim | rel mag | `d_eff_c(Δ)` | Reading |
|---|---:|---:|---:|---|
| L2 → L3 | 0.872 | 0.63 | **1.01** | High similarity. Cos says "small change." Spectrum says "the change is concentrated in essentially one direction" — focused. |
| L7 → L8 | 0.904 | 0.45 | **204.4** | Same similarity neighborhood as L2→L3. Spectrum says "the change spans 200+ effective directions" — diffuse. |
| L26 → L27 | 0.955 | 0.39 | **1.04** | Highest cos sim of the run, but the spectrum says one focused direction — and this is the boundary back into the high-d_eff_c regime, so "one focused direction" plausibly *is* the operation that re-introduces variance structure. |

L7→L8 and L2→L3 have nearly identical cos sim (0.90 vs 0.87) and similar relative magnitudes. The classical metric calls them equivalent. The contribution spectrum says one is focused on a single direction and the other is spread across hundreds. **Whatever "layer utility" means, these two layers cannot be doing the same job.**

### 3. The novelty ratio is k-sensitive and needs care

`||Δ_perp|| / ||Δ_l||` was meant to read as "how much of the layer's change escapes the existing dominant subspace." On this model, `k = ⌈d_eff_c(h_l)⌉` collapses to 2 across most middle layers (because `d_eff_c(h_l) ≈ 1`), so any non-trivial change escapes the top-2 subspace and the metric saturates near 1. It only carries information at the model boundaries (L0→L1: 0.98 with k=42; L27→L28: 0.56 with k=57).

This is a real metric design issue, not a result. With a richer underlying activation (no outlier-feature collapse), the metric should differentiate. We do not claim it as a robust general metric here.

## What this pilot says

1. The classical adjacent-layer cos sim flattens across orders-of-magnitude variation in the structural complexity of what each layer contributes. On Qwen2.5-1.5B, layers with cos sim in the same 0.85–0.96 band differ by 200× in `d_eff_c(Δ)`. *Same angle, different work.*
2. Per-layer `d_eff_uc` and `d_eff_c` together reveal a three-regime structure in the residual stream that cos sim has no access to.
3. Both findings are direct concrete instances of Muyu's critiques #2 and #3 in action.

## What this pilot does not say

- This is not a metric for "layer utility." That framing requires a downstream definition of "useful" we have not engaged. We are claiming structural decomposition, not utility measurement.
- The middle-layer `d_eff ≈ 1` collapse is partly an artifact of the residual stream's outlier-feature norm distribution, well-documented in the quantization-outliers literature. The *variation* of `d_eff_c(Δ)` across layers is the real signal; the *absolute level* of `d_eff_c(h)` partly reflects activation norm pathologies.
- One model. Generalization untested. The middle-layer collapse depth and the L27 structural re-emergence may be Qwen-2.5 specific, family-general, or universal — we have not checked.
- The novelty-ratio metric saturates when `k = 2`. It needs either a fixed `k` (e.g. `k = 32` regardless of `d_eff_c`) or a richer underlying activation to be informative.

## Where this could go

Three possible follow-ups, ranked by cost-to-value:

1. **Same pilot on 2–3 more models** (Mistral-7B, Gemma-3-1B, Llama-3-something). 30 min each. Tells whether the three-regime structure and the cos-vs-spectrum disagreement are Qwen-specific or general. **High value, low cost.**
2. **Fix the novelty-ratio metric** by using fixed `k` instead of `k = ⌈d_eff_c⌉`. ~1 hour. Makes the fourth metric carry information across the whole layer range, not just at boundaries.
3. **Synthesize a "structural decomposition" of the residual stream** into mean-direction energy, dominant-subspace variance, and perpendicular-novelty fraction, all per-layer. ~half a day. Produces a small interpretable framework rather than four parallel metrics.

None of these are committed to. This pilot was a 4-hour bounded sidecar; further work is opt-in.

## Adjacency to other work

This is *not* connected to the K/V asymmetry memo (`docs/kv_asymmetry_memo.md`) or the Bench A null result (`docs/bench_v2_findings.md`). The metrics are reused, but the object of study (residual-stream activations vs. KV cache tensors) and the audience (interpretability / representation geometry vs. KV-compression engineering) are different. Conflating them would weaken both.

The methodological connection is upstream: the centered-vs-uncentered participation-ratio split was developed for KV-compression analysis and turns out to apply to a different problem with the same pathology shape (a few dominant directions hiding real variance structure). That's worth noting as a tooling adjacency, not a theoretical bridge.
