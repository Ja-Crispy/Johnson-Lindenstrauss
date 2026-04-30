#!/usr/bin/env python3
"""Plot CKA vs cosine similarity, raw and carrier-removed, per model.

For each of the three models, produces a 2-panel figure:
  Top:    raw CKA + raw cos sim, plus carrier-removed at k=1,5
  Bottom: same but linear y-axis zoomed to where the action is

Output: docs/img/cka_vs_cos_<suffix>.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cka", required=True, help="CKA JSON from compute_cka.py")
    p.add_argument("--carrier", required=True,
                   help="Carrier-test JSON (for cos sim values)")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    cka = json.loads(Path(args.cka).read_text())
    car = json.loads(Path(args.carrier).read_text())

    cka_results = cka["results"]
    cos_results = car["cosine_comparison"]

    # Align by layer_from
    cka_by_l = {r["layer_from"]: r for r in cka_results}
    cos_by_l = {r["layer_from"]: r for r in cos_results}

    common = sorted(set(cka_by_l.keys()) & set(cos_by_l.keys()))
    t = [l + 0.5 for l in common]

    cka_raw = [cka_by_l[l]["cka_raw"] for l in common]
    cka_p1 = [cka_by_l[l]["cka_perp_k1"] for l in common]
    cka_p5 = [cka_by_l[l].get("cka_perp_k5", cka_by_l[l]["cka_perp_k1"]) for l in common]
    cos_raw = [cos_by_l[l]["cos_raw"] for l in common]
    cos_p1 = [cos_by_l[l].get("cos_perp_k1", cos_by_l[l].get("cos_perp", 0)) for l in common]
    cos_p5 = [cos_by_l[l].get("cos_perp_k5", cos_p1[i]) for i, l in enumerate(common)]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, cos_raw, "o-", color="tab:gray", label="cos raw", lw=2, ms=4)
    ax.plot(t, cos_p1, "s-", color="tab:red", alpha=0.7,
            label="cos carrier-removed (k=1)", lw=1.5, ms=3)
    ax.plot(t, cos_p5, "s--", color="tab:red", alpha=0.4,
            label="cos carrier-removed (k=5)", lw=1.5, ms=3)
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_title("Adjacent-layer cosine similarity")

    ax = axes[1]
    ax.plot(t, cka_raw, "o-", color="tab:blue", label="CKA raw", lw=2, ms=4)
    ax.plot(t, cka_p1, "s-", color="tab:cyan", alpha=0.7,
            label="CKA carrier-removed (k=1)", lw=1.5, ms=3)
    ax.plot(t, cka_p5, "s--", color="tab:cyan", alpha=0.4,
            label="CKA carrier-removed (k=5)", lw=1.5, ms=3)
    ax.set_xlabel("Layer transition l → l+1")
    ax.set_ylabel("Linear CKA (Kornblith 2019)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_title("Adjacent-layer linear CKA")

    title = args.title or "CKA vs cosine, raw and carrier-removed"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
