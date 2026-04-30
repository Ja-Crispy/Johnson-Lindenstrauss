#!/usr/bin/env python3
"""
Linear CKA (Kornblith et al. 2019) baseline for the carrier-decomposition
cross-model test.

Reads saved hidden states + carrier basis from the .npz produced by
analyze_carrier_structure.py. For each layer transition l -> l+1, computes:
  - linear_cka(h_l, h_{l+1})              — raw
  - linear_cka(h_l_perp_k, h_{l+1}_perp_k) for k in {1, 2, 3, 5}
where h_perp_k is h with the top-k carrier subspace projected off.

Comparison target: do the metrics rank layer transitions the same way
adjacent-layer cosine does? If raw CKA is also flat across the middle band
(like cos sim on Qwen/Mistral), then carrier removal adds something CKA
misses. If raw CKA already varies, then CKA is doing what our explicit
carrier removal does.

Usage:
  python scripts/compute_cka.py --input results/carrier_structure_qwen25_1.5b.npz \\
    --output results/cka_qwen25_1.5b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered linear CKA between (N, D_x) and (N, D_y) matrices.

    Standard Kornblith 2019 definition: HSIC-based, value in [0, 1].
    Computed in the small-D form (avoids materializing N x N Gram).
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    num = float(np.sum(XtY * XtY))
    den = float(np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro"))
    if den < 1e-18:
        return 0.0
    return num / den


def project_off_subspace(h: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    coords = h @ V_k.T
    parallel = coords @ V_k
    return h - parallel


def cos_sim_per_position_mean(h1: np.ndarray, h2: np.ndarray) -> float:
    n1 = np.linalg.norm(h1, axis=1, keepdims=True)
    n2 = np.linalg.norm(h2, axis=1, keepdims=True)
    h1n = h1 / np.clip(n1, 1e-12, None)
    h2n = h2 / np.clip(n2, 1e-12, None)
    return float((h1n * h2n).sum(axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .npz from analyze_carrier_structure.py")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--ranks", default="1,2,3,5",
                    help="Carrier ranks for projection (comma-separated)")
    args = ap.parse_args()

    npz_path = Path(args.input)
    if not npz_path.exists():
        raise SystemExit(f"NPZ not found: {npz_path}")

    print(f"Loading {npz_path}")
    npz = np.load(npz_path)
    V_20 = npz["V_20"].astype(np.float64)
    # Find hidden state arrays
    layer_keys = sorted(
        (k for k in npz.files if k.startswith("hidden_state_layer_")),
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    n_layers = len(layer_keys)
    print(f"  found {n_layers} hidden-state layers, V_20 shape {V_20.shape}")

    ranks = [int(r) for r in args.ranks.split(",")]
    print(f"  computing CKA at carrier ranks: {ranks}")

    print(f"\n{'l→l+1':>6}  {'cka_raw':>9}  " + "  ".join(f"cka⊥k={k}".rjust(10) for k in ranks)
          + f"  {'cos_raw':>8}")

    results = []
    h_l = npz[layer_keys[0]].astype(np.float64)
    for li in range(n_layers - 1):
        h_lp1 = npz[layer_keys[li + 1]].astype(np.float64)
        cka_raw = linear_cka(h_l, h_lp1)
        cos_raw = cos_sim_per_position_mean(h_l, h_lp1)
        entry = {
            "layer_from": li,
            "cka_raw": cka_raw,
            "cos_raw": cos_raw,
        }
        cells = []
        for k in ranks:
            Vk = V_20[:k]
            h_l_perp = project_off_subspace(h_l, Vk)
            h_lp1_perp = project_off_subspace(h_lp1, Vk)
            cka_perp = linear_cka(h_l_perp, h_lp1_perp)
            entry[f"cka_perp_k{k}"] = cka_perp
            entry[f"cka_gap_k{k}"] = cka_raw - cka_perp
            cells.append(f"{cka_perp:>10.4f}")
        results.append(entry)
        print(f"  {li:>2}→{li+1:<2}  {cka_raw:>9.4f}  " + "  ".join(cells)
              + f"  {cos_raw:>8.4f}")
        h_l = h_lp1

    out = {
        "input_npz": str(npz_path),
        "n_layers": n_layers,
        "ranks": ranks,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
