#!/usr/bin/env python3
"""Cross-model ablation summary: ΔNLL bar chart across Qwen, Gemma, Mistral."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = [
    # (label, json_path, target_layer)
    ("Qwen2.5-1.5B (L7, fp32)", "results/ablation_qwen25_1.5b.json", 7),
    ("Gemma-3-1B (L9, fp32, no BOS)", "results/ablation_gemma3_1b.json", 9),
    ("Mistral-7B (L13, fp16)", "results/ablation_mistral_7b.json", 13),
]

CONDITIONS = ["random_remove", "carrier_remove", "direction_swap", "centered_remove"]
COLORS = {
    "random_remove": "#1f77b4",
    "carrier_remove": "#d62728",
    "direction_swap": "#9467bd",
    "centered_remove": "#ff7f0e",
}
PRETTY = {
    "random_remove": "random direction removal",
    "carrier_remove": "carrier removal h - (h·c)c",
    "direction_swap": "direction swap (energy preserved)",
    "centered_remove": "centered top-PC removal",
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.18
    x = np.arange(len(CONFIGS), dtype=float)

    for ci, cond in enumerate(CONDITIONS):
        ys = []
        for label, path, tl in CONFIGS:
            d = json.loads(Path(path).read_text())
            base = next(r for r in d["results"] if r["condition"] == "baseline")["ppl"]
            base_nll = math.log(base)
            row = [r for r in d["results"]
                   if r.get("target_layer") == tl and r.get("condition") == cond]
            if not row:
                ys.append(np.nan)
                continue
            v = row[0].get("ppl_median") or row[0].get("ppl") or row[0].get("ppl_mean")
            if v is None or v <= 0:
                ys.append(np.nan)
            else:
                ys.append(math.log(v) - base_nll)

        offset = (ci - (len(CONDITIONS) - 1) / 2) * width
        ax.bar(x + offset, ys, width, color=COLORS[cond],
               label=PRETTY[cond], edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=10)
    ax.set_ylabel("ΔNLL (nats per token, vs baseline)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(math.log(151936), color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(2.45, math.log(151936), "log(vocab) = uniform random predictor",
            ha="right", va="bottom", fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "carrier ablation across three dense families (ΔNLL, single layer per model)\n"
        "random direction removal is at-baseline everywhere; carrier removal is graded by carrier identity"
    )
    plt.tight_layout()
    out = Path("docs/img/ablation_cross_model_dnll.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
