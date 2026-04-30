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
    """Returns (per_seq_hidden_states, token_ids).

    per_seq_hidden_states: list of length n_seqs, each a list of n_layers
        numpy arrays (seq_len, hidden_size) in fp32.
    token_ids: numpy array (n_seqs, seq_len) of int32 token ids used.
    """
    text = load_calibration_text()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < n_seqs * seq_len:
        raise ValueError(f"Need {n_seqs * seq_len} tokens, have {len(tokens)}")

    per_seq = []
    token_ids_per_seq = []
    for i in range(n_seqs):
        chunk = tokens[i * seq_len : (i + 1) * seq_len]
        token_ids_per_seq.append(chunk)
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
    return per_seq, np.array(token_ids_per_seq, dtype=np.int32)


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


def _gram_eig(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and right singular vectors of X via Gram matrix.

    Memory: O(D^2) for the Gram matrix instead of O(N*D) for full SVD's U.
    Returns (eigvals_descending, top_eigenvecs_descending) where each row
    of eigvecs is a right singular vector. Singular values squared = eigvals.
    """
    X64 = X.astype(np.float64, copy=False)
    G = X64.T @ X64  # (D, D)
    eigvals, eigvecs = np.linalg.eigh(G)
    # eigh returns ascending; flip to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvals, eigvecs.T  # rows = right singular vectors


def d_eff_uncentered(h: np.ndarray) -> Tuple[float, np.ndarray]:
    """Returns (d_eff_uc, top right singular vector v_1)."""
    eigvals, vecs = _gram_eig(h)
    return participation_ratio(eigvals), vecs[0]


def d_eff_centered(h: np.ndarray) -> Tuple[float, np.ndarray]:
    """Returns (d_eff_c, top centered eigenvector)."""
    h64 = h.astype(np.float64, copy=False)
    h_c = h64 - h64.mean(axis=0)
    eigvals, vecs = _gram_eig(h_c)
    return participation_ratio(eigvals), vecs[0]


def d_eff_centered_only(h: np.ndarray) -> float:
    h64 = h.astype(np.float64, copy=False)
    h_c = h64 - h64.mean(axis=0)
    eigvals, _ = _gram_eig(h_c)
    return participation_ratio(eigvals)


# ---------------------------------------------------------------------------
# Step 1: persistence matrix
# ---------------------------------------------------------------------------

def persistence_matrix(
    per_layer: List[np.ndarray], layer_indices: List[int], centered: bool
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Build alignment matrix |v_1^(i) . v_1^(j)| over the given layers.
    Returns (matrix, dict layer -> v_1). Uses Gram-matrix eigh (memory O(D^2)).
    """
    top_pcs: Dict[int, np.ndarray] = {}
    for l in layer_indices:
        h = per_layer[l].astype(np.float64, copy=False)
        if centered:
            h = h - h.mean(axis=0)
        _, vecs = _gram_eig(h)
        top_pcs[l] = vecs[0]

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


def detect_collapsed_band(
    d_eff_c_per_layer: List[float], threshold: float = 5.0, min_length: int = 3
) -> Tuple[int, int]:
    """Find the longest contiguous run of layers with d_eff_c < threshold.

    Returns (start, end_inclusive). If no band of min_length found, returns
    (None, None) — caller should fall back to a sensible default or error.
    """
    n = len(d_eff_c_per_layer)
    best_start, best_end, best_len = None, None, 0
    i = 0
    while i < n:
        if d_eff_c_per_layer[i] < threshold:
            j = i
            while j < n and d_eff_c_per_layer[j] < threshold:
                j += 1
            run_len = j - i
            if run_len > best_len:
                best_len = run_len
                best_start = i
                best_end = j - 1
            i = j
        else:
            i += 1
    if best_len < min_length:
        return None, None
    return best_start, best_end


# ---------------------------------------------------------------------------
# Step 2: carrier basis from stacked uncentered SVD
# ---------------------------------------------------------------------------

def stacked_carrier_basis(
    per_layer: List[np.ndarray], layer_indices: List[int], k_max: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k_max right singular vectors of stacked uncentered activations.

    Memory-efficient: builds Gram matrix X^T X (D x D) by accumulating
    per-layer X^T X without materializing the full stacked tensor.
    For Mistral D=4096 the Gram is 134 MB instead of (N*D, D) = 3.4+ GB.

    Returns (V_k_max with shape (k_max, D), eigenvalue spectrum first 50).
    """
    if not layer_indices:
        raise ValueError("layer_indices empty")
    D = per_layer[layer_indices[0]].shape[1]
    G = np.zeros((D, D), dtype=np.float64)
    for l in layer_indices:
        Xl = per_layer[l].astype(np.float64, copy=False)
        G += Xl.T @ Xl
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    Vt = eigvecs.T  # rows = right singular vectors
    return Vt[:k_max].copy(), eigvals[:50].copy()


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
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=5.0,
        help="d_eff_c threshold for auto-detecting the collapsed band",
    )
    parser.add_argument(
        "--band-override",
        default=None,
        help='Override auto-detection. Format "start,end" (inclusive).',
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
    per_seq, token_ids = extract_hidden_states_per_seq(
        model, tokenizer, args.n_seqs, args.seq_len, args.device
    )
    n_layers = len(per_seq[0])
    seq_len = per_seq[0][0].shape[0]
    hidden_size = per_seq[0][0].shape[1]
    print(f"  {n_layers} layers, {args.n_seqs} seqs of {seq_len} tokens, "
          f"hidden_dim={hidden_size}")

    # Capture unembedding (output embedding) BEFORE deleting the model.
    # We save its SVD spectral basis, not raw weights, for the dark-subspace
    # test (Cancedda 2024).
    output_emb = model.get_output_embeddings()
    if output_emb is None or output_emb.weight is None:
        print("  warning: no output embedding (lm_head) found; spectral basis skipped")
        lm_head_weight = None
    else:
        lm_head_weight = output_emb.weight.detach().cpu().to(torch.float32).numpy()
        print(f"  unembedding W: {lm_head_weight.shape}")
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    bos_token = getattr(tokenizer, "bos_token", None)

    del model
    if args.device == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    per_layer = concatenate_per_layer(per_seq)

    # Per-layer baseline d_eff first — needed for auto-detecting the collapsed band
    print("\n[baseline] per-layer d_eff_uc / d_eff_c:")
    baseline = []
    for l in range(n_layers):
        d_uc, _ = d_eff_uncentered(per_layer[l])
        d_c, _ = d_eff_centered(per_layer[l])
        baseline.append({"layer": l, "d_eff_uc": d_uc, "d_eff_c": d_c,
                         "norm_mean": float(np.linalg.norm(per_layer[l], axis=1).mean())})
        print(f"  L{l:>2}: d_eff_uc={d_uc:>7.2f}  d_eff_c={d_c:>7.2f}  "
              f"||h||={baseline[-1]['norm_mean']:>7.2f}")

    # ---------- Detect collapsed band ----------
    if args.band_override:
        start, end = (int(x) for x in args.band_override.split(","))
        band_source = "manual override"
    else:
        d_eff_cs = [b["d_eff_c"] for b in baseline]
        start, end = detect_collapsed_band(
            d_eff_cs, threshold=args.collapse_threshold, min_length=3
        )
        band_source = f"auto (d_eff_c < {args.collapse_threshold})"

    if start is None:
        print(
            "\n[warn] No collapsed band detected with threshold "
            f"{args.collapse_threshold}. Using middle 80% of layers as fallback."
        )
        start = max(1, n_layers // 10)
        end = max(start + 2, n_layers - n_layers // 10 - 1)
        band_source = "fallback (middle 80%)"

    middle_layers = list(range(start, end + 1))
    inner_start = start + 1 if (end - start) >= 4 else start
    inner_end = end - 1 if (end - start) >= 4 else end
    middle_layers_inner = list(range(inner_start, inner_end + 1))

    print(f"\n[band detection] collapsed band: L{start}-L{end} "
          f"({len(middle_layers)} layers) — {band_source}")
    print(f"  inner sanity check: L{inner_start}-L{inner_end}")

    # ---------- Step 1: persistence matrices ----------
    band_label = f"L{start}-L{end}"
    inner_label = f"L{inner_start}-L{inner_end}"
    print(f"\n[step 1] persistence matrices for {band_label}")
    M_uc, top_pcs_uc = persistence_matrix(per_layer, middle_layers, centered=False)
    M_c, top_pcs_c = persistence_matrix(per_layer, middle_layers, centered=True)
    M_uc_inner, _ = persistence_matrix(per_layer, middle_layers_inner, centered=False)
    M_c_inner, _ = persistence_matrix(per_layer, middle_layers_inner, centered=True)
    print(f"  uncentered  {band_label} mean off-diag alignment: {mean_offdiag(M_uc):.4f}")
    print(f"  uncentered  {inner_label} mean off-diag alignment: {mean_offdiag(M_uc_inner):.4f}")
    print(f"  centered    {band_label} mean off-diag alignment: {mean_offdiag(M_c):.4f}")
    print(f"  centered    {inner_label} mean off-diag alignment: {mean_offdiag(M_c_inner):.4f}")

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
            uc_eigvals, _ = _gram_eig(h_perp)
            d_uc_by_k[k] = participation_ratio(uc_eigvals)
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

    # ---------- Step 5: carrier-removed cosine (multi-rank) ----------
    cos_ranks = [1, 2, 3, 5]
    print(f"\n[step 5] carrier-removed cosine, k in {cos_ranks}")
    cosines = []
    cells_header = "  ".join(f"cos⊥k={k}".rjust(8) for k in cos_ranks)
    print(f"  {'l→l+1':>6}  {'cos_raw':>8}  {cells_header}")
    for l in range(n_layers - 1):
        cos_raw = cos_per_position(per_layer[l], per_layer[l + 1])
        entry = {
            "layer_from": l,
            "cos_raw": cos_raw,
            # legacy single-rank field (rank-1) for backward compat with old plotter
            "cos_perp": None,
            "gap": None,
        }
        cells = []
        for k in cos_ranks:
            h_l_perp = project_off_subspace(per_layer[l].astype(np.float64), V_20[:k])
            h_lp1_perp = project_off_subspace(per_layer[l + 1].astype(np.float64), V_20[:k])
            cos_perp_k = cos_per_position(h_l_perp, h_lp1_perp)
            entry[f"cos_perp_k{k}"] = cos_perp_k
            entry[f"gap_k{k}"] = cos_raw - cos_perp_k
            cells.append(f"{cos_perp_k:>8.4f}")
            if k == 1:
                entry["cos_perp"] = cos_perp_k
                entry["gap"] = cos_raw - cos_perp_k
        cosines.append(entry)
        print(f"  {l:>2}→{l+1:<2}  {cos_raw:>8.4f}  " + "  ".join(cells))

    # ---------- Step 6: position localization ----------
    # Pick 3 layers spaced through the detected band: 1/4, 1/2, 3/4 of the way
    band_len = end - start + 1
    pos_layers = [
        start + max(1, band_len // 4),
        start + band_len // 2,
        start + (3 * band_len) // 4,
    ]
    pos_layers = sorted(set(pos_layers))
    print(f"\n[step 6] position localization for layers {pos_layers} "
          "(rank-1 carrier; report peak position, no a-priori labeling)")
    position_profiles = {}
    for pl in pos_layers:
        prof = position_carrier_profile(per_seq, pl, V_20[0])
        position_profiles[pl] = prof
        print(f"  L{pl}: peak-position={prof['global_max_pos']}, "
              f"max/median ratio={prof['ratio_max_to_median']:.2f}")

    # ---------- Step 7: unembedding spectral basis (for Cancedda dark-subspace test) ----------
    if lm_head_weight is not None:
        print("\n[step 7] unembedding W spectral decomposition")
        # SVD of W_unembed in float64. Right singular vectors (rows of Vt_lm)
        # live in residual-stream space; ordered by descending singular value.
        # Top-k = "bright" directions (project to logits with high gain).
        # Bottom-k = "dark" directions (Cancedda 2024).
        # Gram-matrix SVD: avoids materializing U which is (vocab_size, D)
        # and would be ~1 GB for Mistral. We only need sigma and Vt.
        W64 = lm_head_weight.astype(np.float64, copy=False)
        G_lm = W64.T @ W64
        eig_lm, eigvecs_lm = np.linalg.eigh(G_lm)
        eig_lm = np.clip(eig_lm[::-1], 0.0, None)
        eigvecs_lm = eigvecs_lm[:, ::-1]
        sigma_lm = np.sqrt(eig_lm).astype(np.float32)
        Vt_lm = eigvecs_lm.T.astype(np.float32)
        del W64, G_lm, eig_lm, eigvecs_lm
        sv_max = float(sigma_lm[0])
        sv_min = float(sigma_lm[-1])
        sv_ratio = sv_max / max(sv_min, 1e-12)
        print(f"  W_unembed shape: {lm_head_weight.shape}")
        print(f"  singular values: max={sv_max:.4f}, min={sv_min:.6f}, ratio={sv_ratio:.2f}")
        # Quick alignment of carrier top-1 with unembedding tail
        for k_tail in [1, 5, 20, 50, 100]:
            tail_basis = Vt_lm[-k_tail:]  # (k_tail, D)
            c1_in_tail = float(np.sum((V_20[0] @ tail_basis.T) ** 2))
            print(f"  carrier c_1 fraction in W_unembed tail-{k_tail}: "
                  f"{c1_in_tail:.4f}")
    else:
        sigma_lm = None
        Vt_lm = None

    # ---------- Save ----------
    out = {
        "model": args.model,
        "precision_forward": "fp32",
        "n_seqs": args.n_seqs,
        "seq_len": args.seq_len,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "collapsed_band": {
            "start": int(start),
            "end_inclusive": int(end),
            "n_layers": len(middle_layers),
            "source": band_source,
            "threshold": float(args.collapse_threshold),
        },
        "middle_layers": middle_layers,
        "baseline_per_layer": baseline,
        "persistence": {
            "uncentered_band_matrix": M_uc.tolist(),
            "centered_band_matrix": M_c.tolist(),
            "uncentered_band_mean_offdiag": mean_offdiag(M_uc),
            "centered_band_mean_offdiag": mean_offdiag(M_c),
            "uncentered_inner_mean_offdiag": mean_offdiag(M_uc_inner),
            "centered_inner_mean_offdiag": mean_offdiag(M_c_inner),
            "band_label": band_label,
            "inner_label": inner_label,
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
        "cosine_ranks_swept": cos_ranks,
        "position_localization": position_profiles,
        "tokenizer": {
            "bos_token_id": bos_token_id,
            "bos_token": bos_token,
        },
        "unembedding_summary": (
            None if sigma_lm is None
            else {
                "shape": list(lm_head_weight.shape),
                "sigma_max": float(sigma_lm[0]),
                "sigma_min": float(sigma_lm[-1]),
                "sigma_ratio": float(sigma_lm[0] / max(sigma_lm[-1], 1e-12)),
                "n_singular_values": int(len(sigma_lm)),
                "carrier_c1_fraction_in_tail_k1": (
                    float(np.sum((V_20[0] @ Vt_lm[-1:].T) ** 2))
                ),
                "carrier_c1_fraction_in_tail_k20": (
                    float(np.sum((V_20[0] @ Vt_lm[-20:].T) ** 2))
                ),
                "carrier_c1_fraction_in_tail_k100": (
                    float(np.sum((V_20[0] @ Vt_lm[-100:].T) ** 2))
                ),
            }
        ),
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

    # Save numerical artifacts (npz, not in JSON):
    #   - V_20: carrier basis (top-20 right singular vectors of stacked
    #     uncentered band activations)
    #   - c_avg: sign-aligned mean of per-layer top PCs
    #   - sigma_unembed, Vt_unembed: W_unembed SVD spectral basis (for
    #     Cancedda dark-subspace test)
    #   - token_ids: input ids used during forward (n_seqs, seq_len)
    #   - bos_token_id: tokenizer's BOS id (or -1 if none)
    #   - hidden_state_layer_<l>: per-layer concatenated activations
    #     (n_seqs * seq_len, hidden_size) in fp32. Saved per-layer rather
    #     than as a stacked tensor to allow lazy per-layer loading later.
    npz_path = out_path.with_suffix(".npz")
    payload = {
        "V_20": V_20,
        "c_avg": c_avg,
        "token_ids": token_ids,
        "bos_token_id": np.array([bos_token_id if bos_token_id is not None else -1],
                                  dtype=np.int64),
    }
    if sigma_lm is not None:
        payload["sigma_unembed"] = sigma_lm
        payload["Vt_unembed"] = Vt_lm
    for l in range(n_layers):
        payload[f"hidden_state_layer_{l}"] = per_layer[l].astype(np.float32)
    np.savez_compressed(npz_path, **payload)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {npz_path}")
    print(f"  carrier basis V_20, c_avg, token_ids ({token_ids.shape})")
    if sigma_lm is not None:
        print(f"  W_unembed spectral basis Vt_unembed {Vt_lm.shape} + sigma "
              f"({len(sigma_lm)} values)")
    print(f"  hidden states for {n_layers} layers (fp32, {per_layer[0].shape})")


if __name__ == "__main__":
    main()
