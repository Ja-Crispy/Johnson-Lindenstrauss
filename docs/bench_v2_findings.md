# Bench A: SpectralQuant replication + 3-tier extension — findings

**Status (2026-04-20):** null result on compressor improvements. Harness
validated qualitatively. Contribution reframed from "we beat SpectralQuant"
to "K/V asymmetry + geometry/methodology audit."

Live contribution after this run: the spectral geometry observation
(`d_eff_uc` ≈ 3.4 vs `d_eff_centered` ≈ 35) and independent corroboration
of the K/V asymmetry that Tom has already landed in TurboQuant+.

---

## 1. Setup

- **Metric**: attention-weight cosine similarity —
  `softmax(q @ k.T / sqrt(d))` for FP16 K and reconstructed K, then
  cosine on those weight vectors. Matches SpectralQuant's
  `phase3_exp1_attention_quality.py:232`. **Not** K reconstruction cosine,
  and **not** full attention-output cosine.
- **Quantizer**: 3σ uniform with `nan_to_num` on std to handle
  single-element tiers. Matches SQ `_uniform_quantize` except for the NaN
  guard (vendor code has the same latent bug but never hits it with their
  calibration).
- **Rotations**: Haar random orthogonal for T; uncentered eigenvectors +
  mean subtraction for S (matches vendor); centered eigenvectors + mean
  subtraction for Scv (our corrected variant); 3-tier Uncentered-V for O.
- **Calibration**: computed inline from the model's own KV cache,
  per (layer, head). No dependency on vendor phase0/1/2 artifacts.
- **Model**: Qwen2.5-1.5B (vendor exp1 default).
- **Sampling**: 5 sampled layers across 28, all KV heads (2), 8 random
  unit-vector query probes per head, 20 wikitext test sequences × 512
  tokens.
- **Split modes**: `same` (calibration and evaluation on the same
  sequences, matching vendor exp1) and `heldout` (10/10 split).

Result artifacts:
- `results/bench_v2_qwen2.5-1.5b_same_cV.json` (same split, with Scv)
- `results/bench_v2_qwen2.5-1.5b_heldout_cV.json` (heldout, with Scv)
- Harness: `scripts/benchmark_v2_weights.py`

## 2. Main results

| Config | nom | eff | same cos | heldout cos |
|---|---:|---:|---:|---:|
| T-3.0  | 3.0 | 3.000 | 0.97177 | 0.97279 |
| S-3.0  | 3.0 | 3.028 | 0.99956 | 0.99960 |
| Scv-3.0| 3.0 | 3.028 | 0.99948 | 0.99950 |
| O-3.0  | 3.0 | 3.000 | 0.99962 | 0.99962 |
| T-2.0  | 2.0 | 2.000 | 0.96604 | 0.96827 |
| S-2.0  | 2.0 | 2.028 | 0.99781 | 0.99790 |
| Scv-2.0| 2.0 | 2.028 | 0.99781 | 0.99788 |
| O-2.0  | 2.0 | 2.000 | 0.99792 | 0.99791 |

Calibration stats: `d_eff_uc` mean 3.38, median 3.36, range [1.00, 6.57].
`d_eff_centered` mean 35.03, median 33.18, range [21.82, 59.64]. Stable
across splits.

**What replicates:** the qualitative ordering T ≪ S is stable across
splits and bit levels. The geometry numbers (both d_effs) are stable
across splits. The `d_eff_uc` ≈ 4 observation from the SpectralQuant paper
shows up cleanly.

**What does not anchor-check:** absolute values. Vendor
`baseline_reproduction.json` targets TQ 3-bit cosine of `0.985` and
reports `0.9525` under their own reproduction (status:
`FAILED_REPRODUCTION`). Our T-3.0 = `0.972` sits between their target and
their reproduction. This is a legitimate range but not a tight anchor.
**We do not claim absolute replication.**

## 3. Negative ablations

Both candidate compressor improvements fail heldout.

**3-tier (O):** at integer bit budgets the budget solver collapses to
`b_high = b_med = b_low`, so the 3-tier method degenerates to
uniform-on-spectral. The only bit level where 3-tier structure actually
exercises is `2.5`, where the margin vs S appears positive on same-split
(+0.00082) but *reverses* under heldout (−0.00099). At integer bits the
margin is within noise. `2.5` is additionally confounded by effective-bit
mismatch between 2-tier and 3-tier solvers; see §5.

**Centered V (Scv):** the "wasted bit on mean direction" hypothesis was
that SQ's uncentered-V top eigenvector aligns with the mean direction, so
`b_high` bits on `(x − mean) @ v_1 ≈ 0` are wasted. Testing this with
centered-covariance eigenvectors under the same bit allocation showed Scv
**tied or slightly worse** than S on both splits, both bit levels. The
mean-variance mixture captured by uncentered V evidently carries real
signal that pure variance ordering loses.

Both are documented as tested-negative ablations. Do not claim either as
a compressor win.

## 4. Independent corroboration

SpectralQuant's own `results/comparison/proper_comparison.json` on the
same model (Qwen2.5-1.5B) reports `sq_mean = 0.8306`, `tq_mean = 0.8424`,
`win_rate = 0.42` across 100 per-head samples. Their own internal
head-level comparison shows SpectralQuant does not consistently beat
TurboQuant on this model.

Our heldout null on 3-tier and Scv is consistent with this. The broader
"SpectralQuant beats TurboQuant" story in the paper appears to be
model- and metric-dependent, not universal; our Bench A narrows this
specifically for the attention-weight cosine metric on Qwen2.5-1.5B.

## 5. Methodological audit items

- **Effective bits vs nominal bits.** The vendor 2-tier solver with
  `d_sem = 1` on some heads fails to hit fractional bit budgets cleanly
  (e.g. requested `2.5`, effective `2.028`). Fractional-bit comparisons
  need an effective-bits column prominently reported, not just nominal.
- **`int(round(avg_bits))` collapses `T-2.5` to `T-2.0`.** Vendor exp1
  has the same behavior; not a harness bug but a methodology point.
  Reporting fractional-bit TurboQuant as meaningfully different from the
  nearest integer is misleading.
- **1-element tier NaN in the quantizer.** Vendor `_uniform_quantize`
  would fail with NaN on heads where `d_sem = 1` because unbiased std over
  one element is undefined. Vendor never hits it with their calibration
  protocol; our inline calibration does. `nan_to_num` guard is the minimal
  fix.
- **Random query probes vs real Q.** The 8 random unit-vector probes are
  a softmax-forgiving stress test. Whether real Q vectors would amplify
  or attenuate the heldout null is **untested** and would be the most
  informative single next experiment if the SQ chapter reopens.

## 6. What remains live

- **K/V asymmetry** — Tom's `TurboQuant+ README` (post-update) and his
  shipping default (`q8_0` keys, `turbo3` values) now explicitly say V is
  effectively free, K is the quality axis. Our per-model profiling data
  characterizes K vs V spectral structure separately and can corroborate
  or refine the design. Highest leverage, lowest effort.
- **Geometry / theory writeup** (Plan D) — unchanged in scope. The
  `d_eff_uc` vs `d_eff_centered` split is a stable geometric observation
  independent of whether it translates to bit allocation wins. Should
  frame this as "what the geometry says about K," not "a new compressor."
- **Methodology audit** — the effective-bits, NaN-tier, and
  `int(round())` issues are small but real. Worth a short PR or issue
  against vendor.

## 7. What is paused

- **Bench B (WHT)** — no longer informative without a validated spectral
  win to beat real TurboQuant with. Resume only if a new compressor
  hypothesis appears.
- **C++ integration of our variants into `llama-cpp-turboquant`** — no
  validated win to integrate.
- **MLX port of SpectralQuant with 3-tier** — no validated win to port.
- **Further sweeps over 3-tier / centered-V variants** — two hypotheses
  tested and both failed heldout; further sweeps are a fishing expedition
  absent a new principled hypothesis.
- **STILL (learned Perceiver KV compression)** — interesting different
  paradigm (Conor O'Neill group, ~7M params across all layers, 8×
  compression in milliseconds via KL distillation). Not pursued here;
  would be a project pivot, not a continuation.

## 8. Recommended single next action

Profile K and V separately across the 12 models we already have geometry
data for and write a one-page note on why asymmetric K/V is the right
design. Inputs are already cached in `archive/vendor_untracked_backup_2026-04-20/`
and `profiles/`. This is the highest-leverage action that uses the
infrastructure we already built and matches the direction Tom has already
landed on.
