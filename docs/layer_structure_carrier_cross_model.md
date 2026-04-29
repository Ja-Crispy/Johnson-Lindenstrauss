# Carrier Decomposition Across Three Dense Models

**Status (2026-04-29):** Cross-model extension of the carrier-decomposition mechanism test. n=3 case studies from three families: Qwen, Gemma, Mistral. All three confirm the carrier phenomenon. The carrier's *identity* and the cosine-confound it produces vary across families.

This is a follow-up to:
- `docs/layer_structure_pilot.md` (n=1, original pilot)
- `docs/layer_structure_carrier_test.md` (n=1, mechanism test on Qwen2.5-1.5B)

The Qwen results in those documents are reproduced here in the cross-model table; the substantive new content is Gemma-3-1B and Mistral-7B-Instruct-v0.3.

---

## Headline (one paragraph)

Across three families on Apple Silicon (Qwen2.5-1.5B, Gemma-3-1B-IT, Mistral-7B-Instruct-v0.3), the residual stream's middle-layer apparent 1-D collapse is in every case explained by a low-rank persistent carrier subspace. After projecting off that subspace, per-layer `d_eff_c` jumps from ~1 to tens or hundreds in all three models — confirming "rich variance was hidden under a small dominant subspace" is general, not Qwen-specific. **What varies is the carrier's identity.** Two models (Qwen and Mistral) carry a position-0 / BOS-localized attention sink; one model (Gemma) carries a position-uniform feature. **The cosine-confound depends on this identity:** when the carrier is a sink, removing it does *not* unflatten adjacent-layer cosine (the perpendicular content also persists per-token); when the carrier is non-sink uniform, removing it *does* unflatten cosine substantially. This is a sharper version of Muyu's #2 critique: cosine is direction-blind in a way that depends on what the dominant direction is.

## Setup

All three runs use:
- 8 wikitext sequences × 512 tokens
- fp32 forward + analysis
- Auto-detected collapsed band (`d_eff_c < 5.0`, longest contiguous run)
- Position-localization at three layers spaced through the detected band
- Code: `scripts/analyze_carrier_structure.py`, `scripts/plot_carrier_structure.py`

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
| Effective carrier rank | 1 | 1 | 2–3 |
| Carrier-removed cos gap, band (max / mean) | 0.0016 / 0.0004 | **0.342 / 0.136** | 0.0063 / 0.0012 |
| Position-localization peaks | pos 0 / 0 / 0 | pos 460 / 460 / 46 | pos 0 / 0 / 0 |
| Position max/median ratio | 5172 / 2735 / 1749 | **1.7 / 1.7 / 1.5** | 987 / 252 / 73 |
| Δ_⊥ fraction band (min/mean/max) | 98.6 / 99.7 / 99.9% | 20.2 / 61.9 / 87.8% | 98.3 / 99.3 / 99.8% |

## What's universal

1. **Persistent low-rank carrier exists in every model.** Inner-band persistence is 0.999+ in all three. The full-band number being lower for Mistral (0.92) is driven by L1, which has a different top-PC than the rest of the band; trimming to L2–L25 brings persistence to 0.9991.
2. **d_eff_c recovery works in every model.** After projecting off rank-1 (Qwen, Gemma) or rank-2 (Mistral), middle layers go from `d_eff_c ≈ 1` to tens or hundreds. The "1-D collapse" was always a measurement artifact of one or two dominant directions.
3. **Layer updates compute mostly perpendicular to the carrier in sink-style models.** Δ_⊥ is 99%+ on Qwen and Mistral.

## What varies

### Carrier identity: sink vs. non-sink

| Model | Peak position (L_mid) | Max/median ratio |
|---|---:|---:|
| Qwen2.5-1.5B | 0 (BOS) | 2735 |
| Gemma-3-1B | 460 | 1.7 |
| Mistral-7B | 0 (BOS) | 252 |

**Qwen and Mistral**: massive activation at position 0 across all sampled middle layers, with max/median ratios 70× to 5000×. Textbook attention-sink phenomenon (StreamingLLM, Sun et al. 2024 "Massive Activations").

**Gemma**: peak position is not 0, varies between layers (460, 460, 46), and the localization is essentially absent (max/median ratio 1.5–1.7). Gemma's carrier is a **position-uniform feature** — a single direction in feature space that all positions activate roughly equally. Not a sink in the StreamingLLM sense.

### Carrier rank: rank-1 (Qwen, Gemma) vs rank-2/3 (Mistral)

Mistral's recovery curve has a distinctive shape: removing rank-1 barely improves d_eff_c (1.14 → 1.40 at L7), but removing rank-2 jumps it to 74.78. Rank-3 to 164. So Mistral has at least two carrier directions of comparable magnitude, and the d_eff collapse is dominated by their *combined* subspace, not by a single direction.

Qwen and Gemma are clean rank-1: top-1 captures 98% of variance.

### Cosine confound: response to carrier removal

This is the most interesting axis.

| Model | Mean carrier-removed cos gap (band) | Max gap | Carrier-removed cos behavior |
|---|---:|---:|---|
| Qwen2.5-1.5B | 0.0004 | 0.0016 | Effectively unchanged from raw |
| Mistral-7B | 0.0012 | 0.0063 | Effectively unchanged from raw |
| Gemma-3-1B | **0.136** | **0.342** | **Substantially below raw** |

On Qwen and Mistral (sink-carrier models), removing the carrier doesn't unflatten cosine. The off-carrier perpendicular content also persists per-token across layers, so per-token cosine remains high even after the sink is removed.

On Gemma (non-sink carrier), removing the carrier *does* unflatten cosine substantially. Specific transitions show carrier-removed cos as low as 0.64 while raw cos is 0.98 (L17→18). The original Muyu hypothesis — "cosine is hiding layer-to-layer change because of the dominant direction" — works as predicted on Gemma.

### Δ_⊥ fraction varies a lot in Gemma

Mean Δ_⊥ fraction in the band:
- Qwen: 99.7% (consistently perpendicular)
- Mistral: 99.3% (consistently perpendicular)
- Gemma: **61.9%** (varies layer-to-layer; some layers are mostly carrier-aligned)

Combined with Gemma's smaller cos band, this is consistent with Gemma's carrier being something more dynamic than a sink — layers actively read and write into the carrier direction in Gemma, while Qwen/Mistral mostly maintain it as infrastructure.

## What this means

The original Qwen-only writeup framed three things:
1. There is a single persistent carrier.
2. It is the BOS attention sink.
3. Removing it doesn't fix cosine.

After cross-model: only (1) is universal. (2) is sink-or-not depending on family. (3) is a *consequence* of (2) — when the carrier is the sink, both raw and carrier-removed cosine are dominated by structure that persists per-token; when it isn't, carrier-removed cosine reveals more.

The cleaner generalized claim:

**Adjacent-layer cosine in the residual stream is dominated by a small, persistent carrier subspace, but the carrier's identity (sink-like vs. position-uniform) determines whether removing it exposes layer-to-layer variation. Variance-spectrum metrics (`d_eff_c` on activations and on Δ) reveal layer behavior that cosine does not, regardless of carrier identity.**

That's the Muyu-facing claim now. Less elegant than the original n=1 sound bite, but more honest and more interesting.

## Figures

For each model, five figures (`docs/img/carrier_*_<model>.png`):

- `carrier_persistence_*` — cross-layer top-PC alignment heatmap
- `carrier_recovery_*` — `d_eff_c` per layer at original and carrier-removed-rank-{1,2,3,5,10,20}
- `carrier_decomposition_*` — per-transition Δ_∥/Δ_⊥ fractions and `d_eff_c(Δ_⊥)`
- `carrier_cosine_comparison_*` — raw vs carrier-removed cos sim per transition
- `carrier_position_localization_*` — per-position carrier coefficient

The Gemma cosine figure is the most striking visual artifact — gray (raw) and red (carrier-removed) curves separate by 0.1–0.3 across most middle transitions, in contrast to the perfectly overlapping curves on Qwen and Mistral.

## Limits

1. **n=3 across three families.** Real but not universal. Llama, Phi, hybrids untested.
2. **Cosine-removal test is rank-1 only.** Mistral's carrier is rank-2; we did not test rank-2 cosine removal on Mistral, which might unflatten it. Outstanding question: would removing the full carrier *subspace* (not just top-1) unflatten cosine even on sink-style models?
3. **Sample size 4096 positions.** Stable for spectrum-level claims; fine for the BOS spike (which is dramatic at any reasonable sample size). Less stable for fine position-distribution claims in the non-sink case (Gemma).
4. **fp32 forward.** All three models. Rules out fp16-precision artifacts.
5. **Wikitext only.** Domain effects untested.
6. **Position-localization heuristic.** "max/median ratio > 100" is a clean separation between the sink (Qwen 5000, Mistral 250–1000) and non-sink (Gemma 1.5–1.7) cases on this sample. Not claimed as a general taxonomy.

## What the cross-model picture clarifies and what it doesn't

**Clarifies:**
- The carrier phenomenon is a robust property of dense transformers in this size range, not a Qwen quirk.
- Sink-style and non-sink-style carriers exist in the wild within standard architectures.
- The cosine-confound has two regimes, not one.

**Doesn't clarify:**
- *Why* Gemma's carrier is non-sink while Mistral's and Qwen's are sinks. Could be training data, attention bias initialization, instruction-tuning, head architecture. Not investigated.
- Whether the rank-2 carrier in Mistral is two sinks, a sink + something, or two something-elses. We have the position-localization at rank-1 (BOS-dominated); we did not check rank-2's position pattern.
- What the perpendicular subspace actually encodes. We measured its complexity (`d_eff_c(Δ_⊥)`), not its content.
- Whether this generalizes to MoE, hybrid, vision-language, or 30B+ models.

## Suggested follow-ups (not committed)

1. **Rank-k cosine removal on Mistral.** Quick re-run, ~5 min — does removing rank-2 of the carrier subspace unflatten cosine? If yes, Mistral re-categorizes alongside Gemma after sufficient subspace removal. If no, the Qwen/Mistral pattern is intrinsically immune to subspace subtraction.
2. **Look at rank-2 position-localization on Mistral.** What is the second carrier direction? Another sink at a different position? An auxiliary feature?
3. **Investigate why Gemma's carrier is non-sink.** Compare attention patterns at L11 and L15 between Gemma and Qwen on the same input.
4. **Add Llama-3.2-1B.** Fourth family; would tell us if the sink/non-sink split is Mistral-vs-Gemma vs. universal.
5. **Hybrid models.** Already deferred; CUDA dependency is the blocker.

## Adjacency

This work uses the same toolkit as the K/V asymmetry memo (`docs/kv_asymmetry_memo.md`) but operates on a different object (residual-stream activations, not K/V tensors). The methodological reuse is the only link. The cross-model story here is independent of any compression conclusions.

The original Qwen-deep-dive (`docs/layer_structure_carrier_test.md`) remains the place to look for detailed mechanism and L2-anomaly notes; this document is the breadth pass.
