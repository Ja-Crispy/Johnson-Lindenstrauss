#!/usr/bin/env python3
"""Plot the layer-structure pilot output as a 2x2 figure."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/layer_structure_qwen25_1.5b.json")
    p.add_argument("--output", default="docs/img/layer_structure_qwen25_1.5b.png")
    args = p.parse_args()

    d = json.loads(Path(args.input).read_text())
    layers = d["per_layer"]
    trans = d["per_transition"]

    layer_idx = [s["layer"] for s in layers]
    deff_uc = [s["d_eff_uc"] for s in layers]
    deff_c = [s["d_eff_c"] for s in layers]

    t_idx = [t["layer_from"] + 0.5 for t in trans]
    cos = [t["cos_sim"] for t in trans]
    deff_delta = [t["d_eff_c_delta"] for t in trans]
    perp = [t["novelty_ratio"] for t in trans]
    relmag = [t["rel_magnitude"] for t in trans]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.semilogy(layer_idx, deff_uc, "o-", label="d_eff (uncentered)", color="tab:blue")
    ax.semilogy(layer_idx, deff_c, "s-", label="d_eff (centered)", color="tab:red")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Participation ratio (log)")
    ax.set_title("Per-layer activation concentration")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

    ax = axes[0, 1]
    ax.plot(t_idx, cos, "o-", color="tab:gray")
    ax.set_xlabel("Layer transition (l → l+1)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Classical adjacent-layer cos sim")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(t_idx, deff_delta, "o-", color="tab:purple")
    ax.set_xlabel("Layer transition (l → l+1)")
    ax.set_ylabel("d_eff_c(Δ_l) (log)")
    ax.set_title("Spectrum of the layer's contribution\n(low = focused, high = diffuse)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_idx, perp, "o-", color="tab:green", label="||Δ_perp|| / ||Δ||")
    ax.plot(t_idx, relmag, "s-", color="tab:orange", alpha=0.6, label="||Δ|| / ||h||")
    ax.set_xlabel("Layer transition (l → l+1)")
    ax.set_ylabel("Ratio")
    ax.set_title("Update novelty + relative magnitude")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle(
        f"Layer-structure spectral decomposition: {d['model']}\n"
        f"({d['n_seqs']} × {d['seq_len']}-token sequences, {d['n_positions']} positions)",
        fontsize=12,
    )
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
