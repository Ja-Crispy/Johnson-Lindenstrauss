#!/usr/bin/env python3
"""Plot causal-ablation results: PPL bar chart + per-position NLL heatmap."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRETTY = {
    "baseline": "baseline",
    "carrier_remove": "carrier removal h - (h·c)c",
    "random_remove": "random direction removal",
    "direction_swap": "direction swap (energy preserved)",
    "centered_remove": "centered top-PC removal",
}

COLORS = {
    "baseline": "#666666",
    "carrier_remove": "#d62728",
    "random_remove": "#1f77b4",
    "direction_swap": "#9467bd",
    "centered_remove": "#ff7f0e",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/ablation_qwen25_1.5b.json")
    ap.add_argument("--out-dir", default="docs/img")
    ap.add_argument("--suffix", default="_qwen25_1.5b")
    args = ap.parse_args()

    d = json.loads(Path(args.input).read_text())
    results = d["results"]
    targets = d["target_layers"]

    base = next(r for r in results if r["condition"] == "baseline")
    base_ppl = base["ppl"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.18
    x = np.arange(len(targets), dtype=float)

    ax.axhline(base_ppl, color=COLORS["baseline"], linestyle="--", linewidth=1.4,
               label=f"baseline (PPL={base_ppl:.2f})")

    for ci, cond in enumerate(["random_remove", "direction_swap",
                                "centered_remove", "carrier_remove"]):
        ys, errs = [], []
        for tl in targets:
            row = [r for r in results
                   if r.get("target_layer") == tl and r.get("condition") == cond]
            if not row:
                ys.append(np.nan)
                errs.append(0.0)
                continue
            r = row[0]
            if "ppl" in r:
                ys.append(r["ppl"])
                errs.append(0.0)
            else:
                ys.append(r["ppl_mean"])
                errs.append(r.get("ppl_std", 0.0))
        offset = (ci - 1.5) * width
        ax.bar(x + offset, ys, width, yerr=errs, capsize=3,
               label=PRETTY[cond], color=COLORS[cond],
               edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{t}" for t in targets])
    ax.set_xlabel("target layer (intervention applied to layer output)")
    ax.set_ylabel("perplexity (wikitext-2, 8 × 512 sequences)")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("carrier ablation on Qwen2.5-1.5B — PPL by target layer × condition")
    plt.tight_layout()
    out1 = Path(args.out_dir) / f"ablation_ppl{args.suffix}.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out1}")

    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4),
                              sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, tl in zip(axes, targets):
        ax.plot(base["per_position_mean_nll"], color=COLORS["baseline"],
                lw=1.5, label="baseline", alpha=0.85)
        for cond in ["centered_remove", "carrier_remove"]:
            row = [r for r in results
                   if r.get("target_layer") == tl and r.get("condition") == cond
                   and "per_position_mean_nll" in r]
            if not row:
                continue
            ax.plot(row[0]["per_position_mean_nll"],
                    color=COLORS[cond], lw=1.2, alpha=0.85,
                    label=cond.replace("_", " "))
        ax.set_xlabel("within-sequence position")
        ax.set_title(f"L{tl} intervention")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("per-position NLL (mean across 8 sequences)")
    fig.suptitle("per-position NLL after carrier-targeted intervention", fontsize=11)
    plt.tight_layout()
    out2 = Path(args.out_dir) / f"ablation_per_position{args.suffix}.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out2}")


if __name__ == "__main__":
    main()
