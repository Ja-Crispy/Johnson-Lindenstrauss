#!/usr/bin/env python3
"""
Aggregate K/V spectral-asymmetry statistics across profiled models.

Reads profiles/spectral_*.json (one per model, produced by
scripts/spectral_profiler.py), emits:

  docs/kv_asymmetry_memo.md    — action-oriented memo, markdown tables
  docs/img/kv_asymmetry_dumbbell.png  — cross-model K vs V d_eff_centered
                                        dumbbell chart

No SpectralQuant/TurboQuant dependency. Self-contained.
"""

from __future__ import annotations

import glob
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
PROFILE_GLOB = str(ROOT / "profiles" / "spectral_*.json")
MEMO_PATH = ROOT / "docs" / "kv_asymmetry_memo.md"
FIG_PATH = ROOT / "docs" / "img" / "kv_asymmetry_dumbbell.png"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    path: str
    model_id: str
    short_name: str
    architecture: str
    is_hybrid: bool
    n_model_layers: int
    n_attn_layers: int
    n_heads: int
    head_dim: int

    # Summary-level stats (whole-model means)
    k_deff_uc: float
    k_deff_c: float
    v_deff_uc: float
    v_deff_c: float
    asym_uc: float
    asym_c: float

    # Per-layer arrays of K_avg_d_eff, V_avg_d_eff (centered by default)
    layer_k_deff: List[float]
    layer_v_deff: List[float]

    # Head-level compressibility buckets (K centered d_eff)
    k_heads_highly_compressible: float  # fraction with d_eff_c < 16
    k_heads_stubborn: float              # fraction with d_eff_c >= 32
    v_heads_stubborn: float              # fraction with d_eff_c >= 40


SHORT_NAME_FIXUPS = {
    "google/gemma-3-1b-it": "Gemma-3-1B",
    "LiquidAI/LFM2.5-1.2B-Instruct": "LFM-2.5-1.2B (hybrid)",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B (hybrid)",
}


def short_name(model_id: str) -> str:
    return SHORT_NAME_FIXUPS.get(model_id, model_id.split("/")[-1])


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_profile(path: str) -> ModelProfile:
    with open(path) as f:
        d = json.load(f)

    s = d["summary"]
    layers = d["layers"]

    # Keys come as string ints in JSON; sort numerically
    layer_keys = sorted(layers.keys(), key=int)
    layer_k = [layers[k]["K_avg_d_eff"] for k in layer_keys]
    layer_v = [layers[k]["V_avg_d_eff"] for k in layer_keys]

    # Head-level compressibility: flatten K_heads and V_heads across layers
    k_heads_deff_c = []
    v_heads_deff_c = []
    for k in layer_keys:
        for h in layers[k]["K_heads"]:
            k_heads_deff_c.append(h["d_eff_centered"])
        for h in layers[k]["V_heads"]:
            v_heads_deff_c.append(h["d_eff_centered"])

    def frac(xs, predicate):
        return sum(1 for x in xs if predicate(x)) / max(1, len(xs))

    return ModelProfile(
        path=path,
        model_id=d["model"],
        short_name=short_name(d["model"]),
        architecture=d.get("architecture", "unknown"),
        is_hybrid=bool(d.get("is_hybrid", False)),
        n_model_layers=d.get("n_model_layers", 0),
        n_attn_layers=d.get("n_attn_layers", len(layer_keys)),
        n_heads=d.get("n_heads", 0),
        head_dim=d.get("head_dim", 0),
        k_deff_uc=s["K_d_eff_uncentered_mean"],
        k_deff_c=s["K_d_eff_centered_mean"],
        v_deff_uc=s["V_d_eff_uncentered_mean"],
        v_deff_c=s["V_d_eff_centered_mean"],
        asym_uc=s["KV_asymmetry_uncentered"],
        asym_c=s["KV_asymmetry_centered"],
        layer_k_deff=layer_k,
        layer_v_deff=layer_v,
        k_heads_highly_compressible=frac(k_heads_deff_c, lambda x: x < 16),
        k_heads_stubborn=frac(k_heads_deff_c, lambda x: x >= 32),
        v_heads_stubborn=frac(v_heads_deff_c, lambda x: x >= 40),
    )


# ---------------------------------------------------------------------------
# Boundary view
# ---------------------------------------------------------------------------

def boundary_buckets(
    xs: List[float], boundary_n: int = 3
) -> Tuple[float, float, float]:
    """Return (first_N_mean, middle_mean, last_N_mean). Middle excludes both
    boundaries. For layer counts <= 2*boundary_n+1, middle is whatever remains
    (possibly empty, in which case NaN)."""
    if len(xs) <= 2 * boundary_n:
        return (
            statistics.mean(xs[:boundary_n]) if xs[:boundary_n] else float("nan"),
            float("nan"),
            statistics.mean(xs[-boundary_n:]) if xs[-boundary_n:] else float("nan"),
        )
    first = statistics.mean(xs[:boundary_n])
    middle = statistics.mean(xs[boundary_n : -boundary_n])
    last = statistics.mean(xs[-boundary_n:])
    return (first, middle, last)


# ---------------------------------------------------------------------------
# Dumbbell figure
# ---------------------------------------------------------------------------

def render_dumbbell(profiles: List[ModelProfile], out_path: Path) -> None:
    # Sort dense first, hybrid last; within groups by K d_eff_uc ascending.
    # Uncentered d_eff makes the dense/hybrid contrast visually sharp:
    # dense K clusters at 4-5, hybrid K at 10-11.
    dense = sorted(
        [p for p in profiles if not p.is_hybrid], key=lambda p: p.k_deff_uc
    )
    hybrid = sorted(
        [p for p in profiles if p.is_hybrid], key=lambda p: p.k_deff_uc
    )
    ordered = dense + hybrid

    labels = [p.short_name for p in ordered]
    k_vals = [p.k_deff_uc for p in ordered]
    v_vals = [p.v_deff_uc for p in ordered]
    is_hybrid = [p.is_hybrid for p in ordered]

    y = list(range(len(ordered)))
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (k, v, hy) in enumerate(zip(k_vals, v_vals, is_hybrid)):
        edge = "tab:orange" if hy else "tab:gray"
        ax.plot([k, v], [i, i], color=edge, linewidth=1.5, alpha=0.6, zorder=1)
    ax.scatter(k_vals, y, s=80, color="tab:blue", label="K d_eff (uncentered)", zorder=2)
    ax.scatter(v_vals, y, s=80, color="tab:red", label="V d_eff (uncentered)", zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("d_eff (participation ratio, uncentered covariance)")
    ax.set_title(
        "K/V spectral asymmetry across 8 models (uncentered d_eff)\n"
        "Dense: K ≈ 4–5, V 5–10× wider. Hybrid: K ≈ 10–11, V only ≈ 2× wider."
    )
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Memo rendering
# ---------------------------------------------------------------------------

def render_memo(profiles: List[ModelProfile], fig_rel: str) -> str:
    dense = [p for p in profiles if not p.is_hybrid]
    hybrid = [p for p in profiles if p.is_hybrid]

    def fmt(x: float, nd: int = 2) -> str:
        if x != x:  # NaN
            return "—"
        return f"{x:.{nd}f}"

    def pct(x: float) -> str:
        return f"{x * 100:.0f}%"

    # Headline stats
    dense_asym_uc = [p.asym_uc for p in dense]
    dense_asym_c = [p.asym_c for p in dense]
    hybrid_asym_uc = [p.asym_uc for p in hybrid]

    lines: List[str] = []

    lines += [
        "# K/V Asymmetry: Cross-Model Profiling Memo",
        "",
        f"*{Path(__file__).name} output, {len(profiles)} models, "
        "2048-token wikitext calibration via scripts/spectral_profiler.py.*",
        "",
        "## TL;DR",
        "",
        "On every model tested, V's effective dimension is greater than or "
        "equal to K's — V carries more variance directions than K. The gap "
        "is large on dense attention models and much smaller on hybrid "
        "(Mamba+Attention / DeltaNet) models. This corroborates the "
        "`asymmetric K/V` direction TurboQuant+ has already landed on "
        "(`q8_0` keys, `turbo3` values); the refinement worth flagging is "
        "that hybrid models need a different policy than dense because their "
        "K geometry is genuinely different (K `d_eff_uc` ≈ 10–11 on hybrids "
        "vs ≈ 4–5 on dense).",
        "",
        f"- **Dense models**: V/K uncentered d_eff ratio = "
        f"{min(dense_asym_uc):.1f}×–{max(dense_asym_uc):.1f}× "
        f"(median {statistics.median(dense_asym_uc):.1f}×).",
        f"- **Hybrid models**: V/K uncentered d_eff ratio = "
        f"{min(hybrid_asym_uc):.1f}×–{max(hybrid_asym_uc):.1f}× only.",
        f"- Dense K `d_eff_uncentered` is tightly clustered around 4–5; "
        f"hybrid K sits at ~10–11 — hybrids do **not** have the "
        "single-dominant-direction K structure that makes dense K so "
        "compressible.",
        "",
        "## Data",
        "",
        f"{len(profiles)} models profiled per-head, per-layer, on 2048 "
        "wikitext tokens of calibration data (post-RoPE K from KV cache). "
        "Raw profiles: `profiles/spectral_*.json`. This memo consumes only "
        "the summary statistics and per-layer K/V d_eff means; per-head "
        "detail is available in the raw files for follow-ups.",
        "",
        "## 1. Cross-model K/V spectral summary",
        "",
        "`d_eff_uc` (participation ratio of **uncentered** covariance) "
        "measures structural concentration including the mean direction. "
        "`d_eff_c` (centered) measures variance-around-mean concentration. "
        "Asymmetry columns are `V / K` ratios — higher means V is more "
        "diffuse relative to K.",
        "",
        "| Model | Arch | Attn layers | Heads × dim | K d_eff_uc | K d_eff_c | V d_eff_uc | V d_eff_c | Asym_uc | Asym_c |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in profiles:
        arch = "hybrid" if p.is_hybrid else "dense"
        lines.append(
            f"| {p.short_name} | {arch} | {p.n_attn_layers} | "
            f"{p.n_heads} × {p.head_dim} | "
            f"{fmt(p.k_deff_uc)} | {fmt(p.k_deff_c)} | "
            f"{fmt(p.v_deff_uc)} | {fmt(p.v_deff_c)} | "
            f"{fmt(p.asym_uc, 1)}× | {fmt(p.asym_c, 2)}× |"
        )

    lines += [
        "",
        f"![K/V d_eff dumbbell]({fig_rel})",
        "",
        "## 2. Boundary-layer behavior (dense models)",
        "",
        "Tom's `TurboQuant+` docs flag that first and last layers behave "
        "differently. We split each dense model's layers into first-3 / "
        "middle / last-3 buckets and report mean K and V centered d_eff. "
        "Hybrids have too few attention layers for a meaningful split and "
        "are omitted.",
        "",
        "| Model | K first3 | K mid | K last3 | V first3 | V mid | V last3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in dense:
        k_first, k_mid, k_last = boundary_buckets(p.layer_k_deff)
        v_first, v_mid, v_last = boundary_buckets(p.layer_v_deff)
        lines.append(
            f"| {p.short_name} | {fmt(k_first, 1)} | {fmt(k_mid, 1)} | "
            f"{fmt(k_last, 1)} | {fmt(v_first, 1)} | {fmt(v_mid, 1)} | "
            f"{fmt(v_last, 1)} |"
        )

    # Interpretation summary for boundary view
    lines += [
        "",
        "**Pattern**: on most dense models V `d_eff_c` grows noticeably in "
        "the last 3 layers — V becomes *less* compressible near the output. "
        "K is more stable across layers. Implication: V aggression has to "
        "taper at the last layers; a flat V policy will cost more quality "
        "per saved bit at the tail than at the middle.",
        "",
        "## 3. Head-level compressibility distribution",
        "",
        "What fraction of K heads are easy targets (d_eff_c < 16) versus "
        "stubborn (d_eff_c ≥ 32)? Same for V (using ≥ 40 since V is "
        "generally more diffuse).",
        "",
        "| Model | K heads w/ d_eff_c < 16 | K heads w/ d_eff_c ≥ 32 | V heads w/ d_eff_c ≥ 40 |",
        "|---|---:|---:|---:|",
    ]
    for p in profiles:
        lines.append(
            f"| {p.short_name} | {pct(p.k_heads_highly_compressible)} | "
            f"{pct(p.k_heads_stubborn)} | {pct(p.v_heads_stubborn)} |"
        )

    lines += [
        "",
        "## 4. Implications for quantization policy",
        "",
        "Read this as structural evidence, not as specific bit-width "
        "prescriptions — the mapping from `d_eff` to optimal bits depends on "
        "the quantizer (uniform, Lloyd-Max, TCQ), the rotation (WHT vs "
        "spectral), and the downstream metric (attention cosine, PPL, NIAH).",
        "",
        "1. **V bits can be pushed harder than K bits on dense models.** "
        "Every dense model has V `d_eff_c` higher than K `d_eff_c`, and "
        "uncentered V/K ratios of 6–10×. This is the structural fact behind "
        "`q8_0 keys + turbo3 values`.",
        "2. **Hybrid K is a different geometry.** K `d_eff_uc` on LFM-2.5 "
        "and Qwen3.5-4B is 10–11 rather than 4–5. The aggressive K "
        "compression schemes that work on dense attention will extract less "
        "gain from hybrid K, and the V/K asymmetry narrows (2× vs 6–10×). "
        "If a codebase is shipping dense policies by default, hybrid models "
        "deserve a separate calibration pass or a conservative fallback. "
        "TurboQuant+'s `layer-aware-v-compression.md` already caveats "
        "that Boundary V currently mis-targets hybrid architectures "
        "(e.g. Qwen3.5) because only a subset of layers carry KV attention "
        "caches; the different K spectral structure reported here is "
        "independent mechanistic evidence that dense-tuned K policy will "
        "not transfer cleanly to hybrids.",
        "3. **V needs layer-awareness at the output boundary.** V `d_eff_c` "
        "rises in the last 3 layers across dense models (see §2). A "
        "uniform-across-layers V bit budget will be more lossy at the tail "
        "than at the middle. A one-knob escape hatch (\"last N layers get "
        "+2 V bits\") likely captures most of the quality loss.",
        "4. **Per-head variance is large.** Even on dense Qwen2.5 models, "
        "a non-trivial minority of K heads are stubborn (d_eff_c ≥ 32). A "
        "per-head bit policy captures real signal that a per-layer policy "
        "cannot, but requires metadata overhead. If the codebase is already "
        "per-layer, the next rung up is boundary-layer-aware rather than "
        "per-head.",
        "",
        "## 5. What this memo does not claim",
        "",
        "- Nothing here tells you whether spectral rotation beats WHT in "
        "production. Our `Bench A` (docs/bench_v2_findings.md) shows "
        "SpectralQuant-style variants are not a clear win over TurboQuant "
        "under heldout evaluation on Qwen2.5-1.5B. The asymmetric K/V story "
        "is orthogonal to that: it's about *bit budget allocation per "
        "tensor*, not about *how to quantize a given tensor*.",
        "- Nothing here measures downstream task quality (NIAH, PPL, "
        "generation). Head-level d_eff is a structural proxy; the actual "
        "bit choice needs an end-to-end metric.",
        "- Only 8 models. No 30B+, no MoE, no vision-language. The dense-"
        "vs-hybrid split is real on this sample but the boundary of "
        "generalization is not established.",
        "",
        "## 6. Suggested follow-ups (ranked)",
        "",
        "1. **Cross-check with TurboQuant+'s existing K/V docs before "
        "socializing.** The relevant references are "
        "`vendor/turboquant_plus/docs/papers/asymmetric-kv-compression.md` "
        "(the \"V is free, K is everything\" case, 7 models, 1.5B–104B) "
        "and `layer-aware-v-compression.md` (Boundary V: protect first 2 "
        "and last 2 layers, validated on pure-attention phi-4 and "
        "Qwen2.5-7B). Align terminology and units before any external "
        "writeup. TriAttention V3 is token eviction and is the wrong "
        "citation for K/V bit allocation.",
        "2. **Validate the boundary-V hypothesis against Tom's data.** "
        "`layer-aware-v-compression.md` protects first-2 and last-2 "
        "layers; our §2 data shows V `d_eff_c` rising specifically in the "
        "last 3 layers. Two independent signals pointing at the same "
        "place. A small overlay — plot our per-layer V `d_eff_c` against "
        "his PPL-delta curve from the same model — would tighten the "
        "story and is nearly zero new compute.",
        "3. **Cross-check our hybrid finding against MoE results.** Tom "
        "has already tested Boundary V on Qwen3.5-35B-A3B MoE "
        "(`moe-v-compression-frontier.md`); we have no MoE in our "
        "profile set, and MoE ≠ hybrid attention. Extending our profiler "
        "to one MoE model (e.g. Qwen3-MoE-A3B or Qwen3.5-35B-A3B) would "
        "tell us whether MoE K spectral structure looks dense-like or "
        "hybrid-like, and whether Tom's positive MoE Boundary V result "
        "is consistent with the `d_eff_uc ≈ 4–5` regime.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    paths = sorted(glob.glob(PROFILE_GLOB))
    if not paths:
        raise SystemExit(f"No profiles matched {PROFILE_GLOB}")

    profiles = [load_profile(p) for p in paths]
    # Dense first, then hybrid, stable ordering within
    profiles.sort(key=lambda p: (p.is_hybrid, p.short_name))

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    render_dumbbell(profiles, FIG_PATH)

    fig_rel = str(FIG_PATH.relative_to(MEMO_PATH.parent))
    memo = render_memo(profiles, fig_rel)
    MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMO_PATH.write_text(memo)

    print(f"Profiles aggregated: {len(profiles)}")
    print(f"Memo:    {MEMO_PATH}")
    print(f"Figure:  {FIG_PATH}")


if __name__ == "__main__":
    main()
