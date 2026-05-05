# johnson-lindenstrauss

This repo started as an exploration of the Johnson-Lindenstrauss lemma and dimension-reduction via random projections (forked from [nick yoder's work](https://nickyoder.com/johnson-lindenstrauss/)). It then became the working tree for KV-cache compression research targeting [tom turney's turboquant+](https://github.com/TheTom/turboquant_plus). It then became the home of a residual-stream geometry case study that grew out of that. The repo name no longer describes what's in it; the contents are layered like archaeology.

This README is a guide to what's actually here.

## current research output

The active branches are:

- **`research/residual-carriers`** — current work, public-facing
- **`preservation/2026-04-23`** — older alias of the same commit, kept for stability of links in published writeups

The original JL work and the early KV-compression iterations sit in the git history of these branches.

## three layers

### layer 1: johnson-lindenstrauss exploration

The original purpose. PyTorch + MLX implementations of random projections (Gaussian, Hadamard / FJLT, learned linear), 4-mode experiment framework (A/B/C/D), comparisons against Yoder's original numerics. Lives in:

- `cuda/` — CUDA / PyTorch implementations
- `mlx_jl/` — MLX (Apple Silicon) ports
- `archive/` — preserved originals from before refactoring
- `jl_experiments.py`, `jl_full_comparison.py`, `Hadamard random projection.py` (root-level legacy files)

Working but not actively developed.

### layer 2: kv-cache compression research

The pivot. Goal was to extend [turboquant](https://arxiv.org/abs/2504.04658) with non-uniform per-head bit allocation using the centered-vs-uncentered participation ratio gap as a tier boundary. The compressor variants we tried (3-tier, centered-V) didn't beat the 2-tier baseline under heldout evaluation. The structural per-head spectral profiler we built across 8 dense + 2 hybrid models did survive.

Key artifacts:

- `scripts/spectral_profiler.py` — per-head per-layer eigendecomposition of post-RoPE K and V across N models
- `scripts/benchmark_v2_weights.py` — Bench A: SpectralQuant exp1 replication + 3-tier extension
- `scripts/aggregate_kv_asymmetry.py` — cross-model K/V synthesis + dumbbell chart
- `profiles/spectral_*.json` — saved per-head profiles (8 models)
- `docs/bench_v2_findings.md` — Bench A null result writeup
- `docs/kv_asymmetry_memo.md` — cross-model K/V synthesis memo
- `docs/img/kv_asymmetry_dumbbell.png` — the cross-model figure
- Public writeup: [free v, stubborn k](https://yourblog.com/blog/free-v-stubborn-k) (link to your blog)

Honest summary: no compressor win. K/V asymmetry corroboration on n=8. One n=2 hybrid K observation worth following up if hardware allows.

### layer 3: residual carrier work (current)

The thing that actually landed. Triggered by [muyu he's twitter thread](https://x.com/HeMuyu0327/status/2048615865222938972) on adjacent-layer cosine being a misleading layer-utility metric. Reused the spectral toolkit from layer 2 on residual stream activations instead of K/V. Structural finding (every dense model has a low-rank persistent carrier dominating its middle band) extended to causal grounding (carrier is load-bearing infrastructure).

Key artifacts:

- `scripts/analyze_carrier_structure.py` — carrier basis + persistence + dark-subspace alignment + multi-rank cosine
- `scripts/compute_cka.py` — CKA baseline comparison
- `scripts/ablate_carrier.py` — forward-pass causal intervention with norm-restored sanity check
- `scripts/plot_*.py` — figure generators
- `results/carrier_structure_*.json` and `.npz` — Qwen2.5-1.5B, Gemma-3-1B, Mistral-7B (both no-BOS and +BOS where applicable)
- `results/cka_*.json` — CKA at multi-rank carrier-removed
- `results/ablation_*.json` — causal ablation results
- `docs/layer_structure_pilot.md` — original pilot writeup (Qwen-only)
- `docs/layer_structure_carrier_test.md` — n=1 mechanism test on Qwen
- `docs/layer_structure_carrier_cross_model.md` — n=3 cross-model writeup with CKA + BOS sections + multi-rank dark-subspace alignment
- `docs/layer_structure_carrier_ablation.md` — causal ablation across all three families + sanity checks
- `docs/img/carrier_*.png`, `docs/img/cka_vs_cos_*.png`, `docs/img/ablation_*.png` — figures
- Public writeup: [what's under the cosine](https://yourblog.com/blog/whats-under-the-cosine) (link to your blog)

Headline:

- Persistent low-rank carrier in residual stream of every dense family tested (Qwen, Gemma, Mistral)
- Carrier identity differs (dark/sink-style on Qwen + Mistral, bright-aligned on Gemma)
- Adjacent-layer cosine + linear CKA both have carrier-dependent failure modes
- Carrier is causally load-bearing: random direction removal at-baseline, carrier removal up to +18 nats per token, direction swap consistently worse than removal

## negative results worth noting

Documented because they cost real time and might save someone else from repeating them:

- **PCA dimension pruning of K**: 8 layers of pruned K on Llama-3.2-1B = +552% PPL. JL preserves pairwise distances; attention needs Q-K dot product preservation across the downstream stack.
- **3-tier bit allocation (centered/uncentered d_eff gap)**: passes on `same` split, fails on `heldout`. We were overfitting the medium-tier boundary to calibration data.
- **Centered-V eigenvectors as a SpectralQuant fix**: ties or slightly underperforms vanilla SQ. The wasted-bit-on-mean-direction effect is real geometry but not a practical compression lever.
- **SQ vs TQ on Qwen2.5-1.5B**: SpectralQuant's own released `proper_comparison.json` shows `sq_mean = 0.8306` vs `tq_mean = 0.8424`, win_rate = 0.42. Their published claim is shakier than it reads.

## limits we hit

- Apple Silicon hardware ceiling on hybrid models. RecurrentGemma, Zamba2, Hymba all failed to load (no `past_key_values` exposed for Griffin, transformers tied-weights validator bug for Zamba2, CUDA-only deps for Hymba). Hybrid breadth past n=2 needs CUDA.
- Mistral-7B fp32 ablation infeasible on M5 Max (~21 hr compute estimate). Switched to fp16 at single layer. The Mistral causal claim is a lower bound.
- Wikitext only. No NIAH or task-level eval on any compressor or carrier intervention.

## external references

- [TurboQuant — google research, ICLR 2026](https://arxiv.org/abs/2504.04658)
- [TurboQuant+ — tom turney](https://github.com/TheTom/turboquant_plus)
- [SpectralQuant — arxiv 2402.09221](https://arxiv.org/abs/2402.09221)
- [Cancedda 2024 — spectral filters, dark signals, attention sinks](https://aclanthology.org/2024.acl-long.263/)
- [Curse of Depth — Sun et al. 2025](https://arxiv.org/abs/2502.05795)
- [StreamingLLM / attention sinks — Xiao et al. 2023](https://arxiv.org/abs/2309.17453)
- [Massive Activations — Sun et al. 2024](https://arxiv.org/abs/2402.17762)
- [LLM.int8() / outlier features — Dettmers et al. 2022](https://arxiv.org/abs/2208.07339)
- [Linear CKA — Kornblith et al. 2019](https://arxiv.org/abs/1905.00414)
- [Johnson-Lindenstrauss exploration — nick yoder](https://nickyoder.com/johnson-lindenstrauss/)

## license

Inherited from the upstream JL repo. Research code, no warranty, share if useful.
