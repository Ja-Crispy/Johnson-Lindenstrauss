#!/usr/bin/env python3
"""
Carrier decomposition mechanism test.

Phase-2 follow-up to scripts/analyze_layer_structure.py. Tests whether the
L2-L26 d_eff~=1 collapse seen in the pilot is explained by a stable low-rank
carrier subspace persisting across middle layers. Six tests:

  1. Persistence: cross-layer top-PC alignment matrices (centered + uncentered)
     for L2-L26. If alignment is high, the carrier story has structural support.
  2. Carrier basis: top-20 right singular vectors of stacked uncentered
     L2-L26 activations. Cross-checked against sign-aligned mean of per-layer
     top-1 PCs.
  3. Recovery: per-layer d_eff_c after projecting off the top-k carrier, for
     k in {1,2,3,5,10,20}. If middle layers jump from ~1 to tens or hundreds,
     "rich variance hidden under carrier" is confirmed.
  4. Update decomposition: per-transition Δ split into Δ_parallel (carrier-
     aligned, rank-1) and Δ_perp; report magnitudes and d_eff_c(Δ_perp).
  5. Carrier-removed cosine: per-transition raw cos(h_l, h_{l+1}) vs
     cos(h_l_perp, h_{l+1}_perp). If raw is flat but carrier-removed varies,
     this is the cleanest demonstration of what cos sim was hiding.
  6. Position localization: per-position carrier coefficients for representative
     middle layers. If concentrated on BOS / early positions, sink-like; if
     spread, generic carrier.

Runs in fp32 (forward + analysis) for precision, on the same 8x512 wikitext
calibration as the pilot.

Usage:
  python scripts/analyze_carrier_structure.py --model Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None

from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_calibration_text(n_chars: int = 200_000) -> str:
    candidates = [
        ROOT / "vendor" / "llama-cpp-turboquant" / "wikitext-2-raw" / "wiki.test.raw",
        ROOT / "wikitext-2-raw" / "wiki.test.raw",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text()[:n_chars]
    raise FileNotFoundError("wikitext-2 not found")


def extract_hidden_states_per_seq(model, tokenizer, n_seqs, seq_len, device):
    """Returns list of length n_seqs, each a list of n_layers numpy arrays
    (seq_len, hidden_size). All in fp32.
    """
    text = load_calibration_text()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < n_seqs * seq_len:
        raise ValueError(f"Need {n_seqs * seq_len} tokens, have {len(tokens)}")

    per_seq = []
    for i in range(n_seqs):
        chunk = tokens[i * seq_len : (i + 1) * seq_len]
        ids = torch.tensor([chunk], device=device)
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        per_layer = [
            h[0].detach().cpu().to(torch.float32).numpy() for h in out.hidden_states
        ]
        per_seq.append(per_layer)
        del out
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
    return per_seq


def concatenate_per_layer(per_seq: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Convert [seq][layer](T, D) -> [layer](N=n_seqs*T, D)."""
    n_seqs = len(per_seq)
    n_layers = len(per_seq[0])
    return [
        np.concatenate([per_seq[i][li] for i in range(n_seqs)], axis=0)
        for li in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Spectral primitives
# ---------------------------------------------------------------------------

def participation_ratio(eigvals: np.ndarray) -> float:
    s = float(eigvals.sum())
    s2 = float((eigvals ** 2).sum())
    return (s ** 2) / s2 if s2 > 0 else float(len(eigvals))


def d_eff_uncentered(h: np.ndarray) -> Tuple[float, np.ndarray]:
    """Returns (d_eff_uc, top right singular vector v_1)."""
    h64 = h.astype(np.float64)
    _, sigma, Vt = np.linalg.svd(h64, full_matrices=False)
    eig = sigma ** 2
    return participation_ratio(eig), Vt[0]


def d_eff_centered(h: np.ndarray) -> Tuple[float, np.ndarray]:
    """Returns (d_eff_c, top centered eigenvector)."""
    h64 = h.astype(np.float64)
    h_c = h64 - h64.mean(axis=0)
    _, sigma, Vt = np.linalg.svd(h_c, full_matrices=False)
    eig = sigma ** 2
    return participation_ratio(eig), Vt[0]


def d_eff_centered_only(h: np.ndarray) -> float:
    h64 = h.astype(np.float64)
    h_c = h64 - h64.mean(axis=0)
    _, sigma, _ = np.linalg.svd(h_c, full_matrices=False)
    return participation_ratio(sigma ** 2)


# ---------------------------------------------------------------------------
# Step 1: persistence matrix
# ---------------------------------------------------------------------------

def persistence_matrix(
    per_layer: List[np.ndarray], layer_indices: List[int], centered: bool
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Build alignment matrix |v_1^(i) . v_1^(j)| over the given layers.
    Returns (matrix, dict layer -> v_1).
    """
    top_pcs: Dict[int, np.ndarray] = {}
    for l in layer_indices:
        h = per_layer[l].astype(np.float64)
        if centered:
            h = h - h.mean(axis=0)
        _, _, Vt = np.linalg.svd(h, full_matrices=False)
        top_pcs[l] = Vt[0]

    n = len(layer_indices)
    M = np.zeros((n, n), dtype=np.float64)
    for i, li in enumerate(layer_indices):
        for j, lj in enumerate(layer_indices):
            M[i, j] = abs(float(np.dot(top_pcs[li], top_pcs[lj])))
    return M, top_pcs


def mean_offdiag(M: np.ndarray) -> float:
    n = M.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(M[mask].mean())


# ---------------------------------------------------------------------------
# Step 2: carrier basis from stacked uncentered SVD
# ---------------------------------------------------------------------------

def stacked_carrier_basis(
    per_layer: List[np.ndarray], layer_indices: List[int], k_max: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k_max right singular vectors of stacked uncentered activations.
    Returns (V_k_max with shape (k_max, D), eigenvalue spectrum first 50).
    """
    stacked = np.concatenate(
        [per_layer[l] for l in layer_indices], axis=0
    ).astype(np.float64)
    _, sigma, Vt = np.linalg.svd(stacked, full_matrices=False)
    return Vt[:k_max].copy(), (sigma[:50] ** 2).copy()


def sign_aligned_mean_pc(
    top_pcs: Dict[int, np.ndarray], reference_layer: int
) -> np.ndarray:
    """Sign-flip per-layer top PCs so they align with reference, then mean."""
    ref = top_pcs[reference_layer]
    aligned = []
    for v in top_pcs.values():
        sgn = float(np.sign(np.dot(v, ref)))
        if sgn == 0:
            sgn = 1.0
        aligned.append(sgn * v)
    avg = np.mean(np.stack(aligned, axis=0), axis=0)
    return avg / max(np.linalg.norm(avg), 1e-12)


# ---------------------------------------------------------------------------
# Step 3: activation recovery — d_eff after carrier removal
# ---------------------------------------------------------------------------

def project_off_subspace(h: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """h: (N, D), V_k: (k, D). Returns h_perp = h - h V_k^T V_k."""
    coords = h @ V_k.T
    parallel = coords @ V_k
    return h - parallel


# ---------------------------------------------------------------------------
# Step 4: update decomposition
# ---------------------------------------------------------------------------

def update_decomposition(
    h_l: np.ndarray, h_lp1: np.ndarray, V_k: np.ndarray
) -> Dict:
    """Decompose Δ = h_lp1 - h_l into parallel and perp components w.r.t. V_k."""
    h_l64 = h_l.astype(np.float64)
    delta = h_lp1.astype(np.float64) - h_l64

    coords = delta @ V_k.T
    delta_parallel = coords @ V_k
    delta_perp = delta - delta_parallel

    norm_delta = np.linalg.norm(delta, axis=1)
    norm_par = np.linalg.norm(delta_parallel, axis=1)
    norm_perp = np.linalg.norm(delta_perp, axis=1)

    # Per-position ratios then mean (avoids dominance by single positions)
    parallel_frac = float((norm_par / np.clip(norm_delta, 1e-12, None)).mean())
    perp_frac = float((norm_perp / np.clip(norm_delta, 1e-12, None)).mean())

    return {
        "parallel_frac": parallel_frac,
        "perp_frac": perp_frac,
        "d_eff_c_delta": d_eff_centered_only(delta),
        "d_eff_c_delta_perp": d_eff_centered_only(delta_perp),
        "rel_magnitude_delta": float(
            (norm_delta / np.clip(np.linalg.norm(h_l, axis=1), 1e-12, None)).mean()
        ),
    }


# ---------------------------------------------------------------------------
# Step 5: carrier-removed cosine
# ---------------------------------------------------------------------------

def cos_per_position(h1: np.ndarray, h2: np.ndarray) -> float:
    n1 = np.linalg.norm(h1, axis=1, keepdims=True)
    n2 = np.linalg.norm(h2, axis=1, keepdims=True)
    h1n = h1 / np.clip(n1, 1e-12, None)
    h2n = h2 / np.clip(n2, 1e-12, None)
    return float((h1n * h2n).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Step 6: position localization
# ---------------------------------------------------------------------------

def position_carrier_profile(
    per_seq: List[List[np.ndarray]],
    layer: int,
    c: np.ndarray,
) -> Dict:
    """For one layer, compute |h[t] @ c| per within-seq position, aggregated
    across sequences.
    """
    # per_seq[i][layer]: (seq_len, D)
    n_seqs = len(per_seq)
    seq_len = per_seq[0][layer].shape[0]
    coeffs = np.zeros((n_seqs, seq_len), dtype=np.float64)
    for i in range(n_seqs):
        h = per_seq[i][layer].astype(np.float64)
        coeffs[i] = np.abs(h @ c)
    return {
        "mean_per_pos": coeffs.mean(axis=0).tolist(),
        "max_per_pos": coeffs.max(axis=0).tolist(),
        "global_max_pos": int(np.argmax(coeffs.mean(axis=0))),
        "ratio_max_to_median": float(
            coeffs.mean(axis=0).max() / max(np.median(coeffs.mean(axis=0)), 1e-12)
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--n-seqs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--output", default="results/carrier_structure_qwen25_1.5b.json"
    )
    args = parser.parse_args()

    print(f"Loading {args.model} in fp32")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    ).to(args.device)
    model.train(False)

    print(f"Extracting hidden states (fp32 forward): "
          f"{args.n_seqs} seqs x {args.seq_len} tokens")
    per_seq = extract_hidden_states_per_seq(
        model, tokenizer, args.n_seqs, args.seq_len, args.device
    )
    n_layers = len(per_seq[0])
    seq_len = per_seq[0][0].shape[0]
    hidden_size = per_seq[0][0].shape[1]
    print(f"  {n_layers} layers, {args.n_seqs} seqs of {seq_len} tokens, "
          f"hidden_dim={hidden_size}")

    del model
    if args.device == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    per_layer = concatenate_per_layer(per_seq)

    middle_layers = list(range(2, 27))  # L2-L26 inclusive
    middle_layers_inner = list(range(3, 26))  # L3-L25, sanity check

    # Sanity: per-layer baseline d_eff (matches pilot)
    print("\n[baseline] per-layer d_eff_uc / d_eff_c:")
    baseline = []
    for l in range(n_layers):
        d_uc, _ = d_eff_uncentered(per_layer[l])
        d_c, _ = d_eff_centered(per_layer[l])
        baseline.append({"layer": l, "d_eff_uc": d_uc, "d_eff_c": d_c,
                         "norm_mean": float(np.linalg.norm(per_layer[l], axis=1).mean())})
        print(f"  L{l:>2}: d_eff_uc={d_uc:>7.2f}  d_eff_c={d_c:>7.2f}  "
              f"||h||={baseline[-1]['norm_mean']:>7.2f}")

    # ---------- Step 1: persistence matrices ----------
    print("\n[step 1] persistence matrices for L2-L26")
    M_uc, top_pcs_uc = persistence_matrix(per_layer, middle_layers, centered=False)
    M_c, top_pcs_c = persistence_matrix(per_layer, middle_layers, centered=True)
    M_uc_inner, _ = persistence_matrix(per_layer, middle_layers_inner, centered=False)
    M_c_inner, _ = persistence_matrix(per_layer, middle_layers_inner, centered=True)
    print(f"  uncentered  L2-L26 mean off-diag alignment: {mean_offdiag(M_uc):.4f}")
    print(f"  uncentered  L3-L25 mean off-diag alignment: {mean_offdiag(M_uc_inner):.4f}")
    print(f"  centered    L2-L26 mean off-diag alignment: {mean_offdiag(M_c):.4f}")
    print(f"  centered    L3-L25 mean off-diag alignment: {mean_offdiag(M_c_inner):.4f}")

    # ---------- Step 2: carrier basis ----------
    print("\n[step 2] stacked uncentered carrier basis (top-20)")
    V_20, eigs_top50 = stacked_carrier_basis(per_layer, middle_layers, k_max=20)
    var_explained = (eigs_top50 / eigs_top50.sum()).cumsum()
    for k_show in [1, 2, 3, 5, 10, 20]:
        print(f"  top-{k_show:>2} explains "
              f"{var_explained[k_show - 1] * 100:.2f}% of stacked variance "
              f"(within top-50 only)")

    c_avg = sign_aligned_mean_pc(top_pcs_uc, reference_layer=middle_layers[0])
    align_to_stacked = abs(float(np.dot(c_avg, V_20[0])))
    print(f"  sign-aligned mean PC alignment to stacked top-1: {align_to_stacked:.4f}")

    # ---------- Step 3: activation recovery ----------
    print("\n[step 3] activation recovery — d_eff_c after carrier removal")
    k_sweep = [1, 2, 3, 5, 10, 20]
    recovery = {k: [] for k in k_sweep}
    recovery_uc = {k: [] for k in k_sweep}
    header_cells = "  ".join(f"{'k=' + str(k):>7}" for k in k_sweep)
    print(f"  {'layer':>5}  {'orig':>7}  {header_cells}")
    for l in range(n_layers):
        h = per_layer[l]
        orig_d_c, _ = d_eff_centered(h)
        d_c_by_k = {}
        d_uc_by_k = {}
        for k in k_sweep:
            h_perp = project_off_subspace(h.astype(np.float64), V_20[:k])
            d_c_by_k[k] = d_eff_centered_only(h_perp)
            d_uc_by_k[k] = participation_ratio(np.linalg.svd(h_perp, full_matrices=False)[1] ** 2)
            recovery[k].append({"layer": l, "d_eff_c": d_c_by_k[k]})
            recovery_uc[k].append({"layer": l, "d_eff_uc": d_uc_by_k[k]})
        cells = "  ".join(f"{d_c_by_k[k]:>7.2f}" for k in k_sweep)
        print(f"  L{l:>3}  {orig_d_c:>7.2f}  {cells}")

    # ---------- Step 4: update decomposition (rank-1 primary, rank-5 cross-check) ----------
    print("\n[step 4] update decomposition (rank-1 carrier)")
    decomp_r1 = []
    decomp_r5 = []
    print(f"  {'l→l+1':>6}  {'parl%':>6}  {'perp%':>6}  "
          f"{'d_eff(Δ)':>9}  {'d_eff(Δ⊥1)':>11}  {'d_eff(Δ⊥5)':>11}")
    for l in range(n_layers - 1):
        d1 = update_decomposition(per_layer[l], per_layer[l + 1], V_20[:1])
        d5 = update_decomposition(per_layer[l], per_layer[l + 1], V_20[:5])
        d1["layer_from"] = l
        d5["layer_from"] = l
        decomp_r1.append(d1)
        decomp_r5.append(d5)
        print(f"  {l:>2}→{l+1:<2}  "
              f"{d1['parallel_frac']*100:>5.1f}%  "
              f"{d1['perp_frac']*100:>5.1f}%  "
              f"{d1['d_eff_c_delta']:>9.2f}  "
              f"{d1['d_eff_c_delta_perp']:>11.2f}  "
              f"{d5['d_eff_c_delta_perp']:>11.2f}")

    # ---------- Step 5: carrier-removed cosine ----------
    print("\n[step 5] carrier-removed cosine (rank-1)")
    cosines = []
    print(f"  {'l→l+1':>6}  {'cos_raw':>8}  {'cos_perp':>9}  {'gap':>8}")
    for l in range(n_layers - 1):
        cos_raw = cos_per_position(per_layer[l], per_layer[l + 1])
        h_l_perp = project_off_subspace(per_layer[l].astype(np.float64), V_20[:1])
        h_lp1_perp = project_off_subspace(per_layer[l + 1].astype(np.float64), V_20[:1])
        cos_perp = cos_per_position(h_l_perp, h_lp1_perp)
        cosines.append({
            "layer_from": l,
            "cos_raw": cos_raw,
            "cos_perp": cos_perp,
            "gap": cos_raw - cos_perp,
        })
        print(f"  {l:>2}→{l+1:<2}  {cos_raw:>8.4f}  {cos_perp:>9.4f}  "
              f"{cos_raw - cos_perp:>+8.4f}")

    # ---------- Step 6: position localization ----------
    print("\n[step 6] position localization for L7, L13, L20 (rank-1 carrier)")
    pos_layers = [7, 13, 20]
    position_profiles = {}
    for pl in pos_layers:
        prof = position_carrier_profile(per_seq, pl, V_20[0])
        position_profiles[pl] = prof
        print(f"  L{pl}: max-position={prof['global_max_pos']}, "
              f"max/median ratio={prof['ratio_max_to_median']:.2f}")

    # ---------- Save ----------
    out = {
        "model": args.model,
        "precision_forward": "fp32",
        "n_seqs": args.n_seqs,
        "seq_len": args.seq_len,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "middle_layers": middle_layers,
        "baseline_per_layer": baseline,
        "persistence": {
            "uncentered_L2_L26_matrix": M_uc.tolist(),
            "centered_L2_L26_matrix": M_c.tolist(),
            "uncentered_L2_L26_mean_offdiag": mean_offdiag(M_uc),
            "centered_L2_L26_mean_offdiag": mean_offdiag(M_c),
            "uncentered_L3_L25_mean_offdiag": mean_offdiag(M_uc_inner),
            "centered_L3_L25_mean_offdiag": mean_offdiag(M_c_inner),
        },
        "carrier": {
            "stacked_top50_eigs": eigs_top50.tolist(),
            "var_explained_cum_top50": var_explained.tolist(),
            "mean_aligned_PC_to_stacked_alignment": align_to_stacked,
        },
        "recovery": {
            "k_sweep": k_sweep,
            "d_eff_c_per_layer": {
                k: [r["d_eff_c"] for r in recovery[k]] for k in k_sweep
            },
            "d_eff_uc_per_layer": {
                k: [r["d_eff_uc"] for r in recovery_uc[k]] for k in k_sweep
            },
        },
        "update_decomposition_rank1": decomp_r1,
        "update_decomposition_rank5": decomp_r5,
        "cosine_comparison": cosines,
        "position_localization": position_profiles,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Numpy-aware JSON serializer
    def default_serialize(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not serializable: {type(o)}")

    out_path.write_text(json.dumps(out, indent=2, default=default_serialize))

    # Save the carrier basis separately (npz, not in JSON)
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(npz_path, V_20=V_20, c_avg=c_avg)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {npz_path} (carrier basis V_20 + sign-aligned mean PC)")


if __name__ == "__main__":
    main()
