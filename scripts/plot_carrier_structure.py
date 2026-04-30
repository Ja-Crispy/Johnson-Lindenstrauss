#!/usr/bin/env python3
"""Plot the carrier decomposition mechanism test output as 5 figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_persistence(d: dict, out_dir: Path, suffix: str = "") -> None:
    """Two heatmaps side by side: uncentered and centered top-PC alignment."""
    p = d["persistence"]
    # Backward compatibility with old key names
    M_uc = np.array(p.get("uncentered_band_matrix") or p["uncentered_L2_L26_matrix"])
    M_c = np.array(p.get("centered_band_matrix") or p["centered_L2_L26_matrix"])
    band_label = p.get("band_label", "band")
    middle = d["middle_layers"]

    off_uc = p.get("uncentered_band_mean_offdiag",
                   p.get("uncentered_L2_L26_mean_offdiag"))
    off_c = p.get("centered_band_mean_offdiag",
                  p.get("centered_L2_L26_mean_offdiag"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, M, title, off in zip(
        axes,
        [M_uc, M_c],
        ["Uncentered top-PC alignment", "Centered top-PC alignment"],
        [off_uc, off_c],
    ):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
        step = max(1, len(middle) // 5)
        ax.set_xticks(range(0, len(middle), step))
        ax.set_xticklabels([f"L{middle[i]}" for i in range(0, len(middle), step)])
        ax.set_yticks(range(0, len(middle), step))
        ax.set_yticklabels([f"L{middle[i]}" for i in range(0, len(middle), step)])
        ax.set_title(f"{title}\nmean off-diag = {off:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Step 1: Cross-layer top-PC alignment, {band_label} ({d['model']})",
        fontsize=12,
    )
    plt.tight_layout()
    out = out_dir / f"carrier_persistence{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_recovery(d: dict, out_dir: Path, suffix: str = "") -> None:
    """Per-layer d_eff_c, original + carrier-removed for k in {1,2,3,5,10,20}."""
    n_layers = d["n_layers"]
    layer_idx = list(range(n_layers))
    orig = [b["d_eff_c"] for b in d["baseline_per_layer"]]
    k_sweep = d["recovery"]["k_sweep"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(k_sweep)))

    for ax, key, label in zip(
        axes,
        ["d_eff_c_per_layer", "d_eff_uc_per_layer"],
        ["d_eff_c (centered)", "d_eff_uc (uncentered)"],
    ):
        if key == "d_eff_c_per_layer":
            ax.semilogy(layer_idx, orig, "k--", lw=2, label="original", marker="o", mfc="white")
        else:
            orig_uc = [b["d_eff_uc"] for b in d["baseline_per_layer"]]
            ax.semilogy(layer_idx, orig_uc, "k--", lw=2, label="original", marker="o", mfc="white")
        for color, k in zip(cmap, k_sweep):
            ys = d["recovery"][key][str(k)]
            ax.semilogy(layer_idx, ys, "o-", color=color, label=f"k={k} removed", ms=4)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(f"{label} (log)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", ncol=2, fontsize=8)
        ax.set_title(f"Recovery: {label} after carrier removal")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Step 3: d_eff after projecting off the top-k carrier subspace",
        fontsize=12,
    )
    plt.tight_layout()
    out = out_dir / f"carrier_recovery{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_decomposition(d: dict, out_dir: Path, suffix: str = "") -> None:
    """Update decomposition: parallel/perp fractions + d_eff_c(Δ_perp)."""
    r1 = d["update_decomposition_rank1"]
    r5 = d["update_decomposition_rank5"]
    t = [u["layer_from"] + 0.5 for u in r1]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, [u["parallel_frac"] for u in r1], "o-", color="tab:blue",
            label="||Δ_∥|| / ||Δ|| (rank-1)", ms=4)
    ax.plot(t, [u["perp_frac"] for u in r1], "s-", color="tab:red",
            label="||Δ_⊥|| / ||Δ|| (rank-1)", ms=4)
    ax.plot(t, [u["parallel_frac"] for u in r5], "o-", color="tab:cyan",
            label="||Δ_∥|| / ||Δ|| (rank-5)", alpha=0.5, ms=3)
    ax.plot(t, [u["perp_frac"] for u in r5], "s-", color="tab:orange",
            label="||Δ_⊥|| / ||Δ|| (rank-5)", alpha=0.5, ms=3)
    ax.set_ylabel("Fraction of update")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8, ncol=2)
    ax.set_title("Update decomposition: how much of Δ is along the carrier?")

    ax = axes[1]
    ax.semilogy(t, [u["d_eff_c_delta"] for u in r1], "o-", color="tab:purple",
                label="d_eff_c(Δ) — pilot metric", ms=4)
    ax.semilogy(t, [u["d_eff_c_delta_perp"] for u in r1], "s-", color="tab:green",
                label="d_eff_c(Δ_⊥) — off-carrier complexity (rank-1)", ms=4)
    ax.semilogy(t, [u["d_eff_c_delta_perp"] for u in r5], "s-", color="tab:olive",
                label="d_eff_c(Δ_⊥) (rank-5)", alpha=0.5, ms=3)
    ax.set_xlabel("Layer transition l → l+1")
    ax.set_ylabel("d_eff_c (log)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Off-carrier work complexity")

    fig.suptitle(
        "Step 4: Update decomposition (Δ_∥ along carrier, Δ_⊥ orthogonal)",
        fontsize=12,
    )
    plt.tight_layout()
    out = out_dir / f"carrier_decomposition{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_cosine(d: dict, out_dir: Path, suffix: str = "") -> None:
    """The headline figure: raw cos vs carrier-removed cos per transition."""
    c = d["cosine_comparison"]
    t = [r["layer_from"] + 0.5 for r in c]
    cos_raw = [r["cos_raw"] for r in c]

    # Detect available ranks
    ranks = []
    for k in [1, 2, 3, 5]:
        if all(f"cos_perp_k{k}" in r for r in c):
            ranks.append(k)
    if not ranks and "cos_perp" in c[0]:
        ranks = [1]
        for r in c:
            r["cos_perp_k1"] = r["cos_perp"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t, cos_raw, "o-", color="tab:gray", lw=2, ms=5,
            label="raw cos(h_l, h_{l+1})")

    palette = plt.cm.Reds(np.linspace(0.4, 0.9, len(ranks)))
    for color, k in zip(palette, ranks):
        cos_k = [r[f"cos_perp_k{k}"] for r in c]
        ax.plot(t, cos_k, "s-", color=color, lw=1.8, ms=4,
                label=f"carrier removed, k={k}")
    ax.set_xlabel("Layer transition l → l+1")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(
        f"Step 5: Carrier-removed cosine across ranks ({d['model']})\n"
        "Raw cos vs cos with top-k carrier dimensions projected off."
    )
    plt.tight_layout()
    out = out_dir / f"carrier_cosine_comparison{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_position(d: dict, out_dir: Path, suffix: str = "") -> None:
    """Per-position carrier coefficient for representative middle layers."""
    pl = d["position_localization"]
    layers = sorted(int(k) for k in pl.keys())

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(layers)))
    for color, l in zip(cmap, layers):
        prof = pl[str(l)]
        mean_pp = prof["mean_per_pos"]
        ax.plot(
            range(len(mean_pp)),
            mean_pp,
            "-",
            color=color,
            label=f"L{l}: max@pos {prof['global_max_pos']}, "
                  f"max/median = {prof['ratio_max_to_median']:.1f}",
            alpha=0.85,
        )
    ax.set_xlabel("Within-sequence position")
    ax.set_ylabel("|h_l[t] · c_1| — mean across 8 sequences")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Step 6: Carrier coefficient per position\n"
        "(BOS / early-position spike → sink-like; broad → generic carrier)"
    )
    plt.tight_layout()
    out = out_dir / f"carrier_position_localization{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", default="results/carrier_structure_qwen25_1.5b.json"
    )
    p.add_argument("--out-dir", default="docs/img")
    p.add_argument(
        "--suffix",
        default="",
        help='Filename suffix (e.g. "_gemma3_1b" -> carrier_persistence_gemma3_1b.png)',
    )
    args = p.parse_args()

    d = load(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures:")
    fig_persistence(d, out_dir, suffix=args.suffix)
    fig_recovery(d, out_dir, suffix=args.suffix)
    fig_decomposition(d, out_dir, suffix=args.suffix)
    fig_cosine(d, out_dir, suffix=args.suffix)
    fig_position(d, out_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
