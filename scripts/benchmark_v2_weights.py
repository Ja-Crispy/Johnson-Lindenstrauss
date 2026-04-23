#!/usr/bin/env python3
"""
Bench A: SpectralQuant exp1 replication + 3-tier extension.

Metric:     attention weight cosine similarity, softmax(q @ k.T / sqrt(d))
Quantizer:  3-sigma uniform (matches SQ exp1 _uniform_quantize)
Rotations:
  T (TurboQuant): Haar random orthogonal, per (layer, head)
  S (SpectralQuant): uncentered eigvecs V + mean subtraction
  O (Ours 3-tier): same V/mean as S + 3-tier bit allocation using
                   (d_eff_uncentered, d_eff_centered) as tier boundaries

Calibration: per-head (V, mean, d_eff_uc, d_eff_centered) computed inline
from the model's KV cache. No pre-computed calibration artifacts required.

Split modes:
  same    - calibrate and assess on the same sequences (exp1 style,
            used for the replication gate)
  heldout - split sequences 50/50 for calibration vs assessment (cleaner
            but halves sample size)

Reference: vendor/spectralquant/experiments/phase3_exp1_attention_quality.py
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# MPS huge-buffer allocator warmup causes a crash on 7B+ models
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _uniform_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """3-sigma uniform quantizer, per-token scale on last dim. Matches SQ exp1.

    nan_to_num handles the 1-element tier case where unbiased std returns NaN.
    """
    if bits >= 16:
        return x
    n_levels = 2 ** bits
    std = x.std(dim=-1, keepdim=True)
    std = torch.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6).clamp(min=1e-6)
    x_clamp = x.clamp(-3 * std, 3 * std)
    x_norm = (x_clamp / (3 * std) + 1) / 2
    x_int = (x_norm * (n_levels - 1)).round().clamp(0, n_levels - 1)
    return x_int / (n_levels - 1) * 2 * 3 * std - 3 * std


# ---------------------------------------------------------------------------
# Bit-budget solvers
# ---------------------------------------------------------------------------

def solve_bits_2tier(avg_bits: float, d: int, d_sig: int) -> Tuple[int, int]:
    """SQ-style (b_high, b_low) for [0:d_sig] and [d_sig:d]."""
    d_sig = max(1, min(d_sig, d - 1))
    budget = d * avg_bits
    best_err, best = float("inf"), (4, 2)
    for bh in range(2, 9):
        for bl in range(1, bh):
            err = abs(d_sig * bh + (d - d_sig) * bl - budget)
            if err < best_err:
                best_err, best = err, (bh, bl)
    return best


def solve_bits_3tier(
    avg_bits: float, d: int, d_sig: int, d_med_end: int
) -> Tuple[int, int, int]:
    """(b_high, b_med, b_low) for [0:d_sig], [d_sig:d_med_end], [d_med_end:d].

    Enforces b_high >= b_med >= b_low so the medium tier is really a
    middle tier.
    """
    d_sig = max(1, min(d_sig, d - 1))
    d_med_end = max(d_sig + 1, min(d_med_end, d))
    n_sig = d_sig
    n_med = d_med_end - d_sig
    n_noise = d - d_med_end
    budget = d * avg_bits
    best_err, best = float("inf"), (4, 3, 2)
    for bh in range(2, 9):
        for bm in range(1, bh + 1):
            for bl in range(1, bm + 1):
                err = abs(n_sig * bh + n_med * bm + n_noise * bl - budget)
                if err < best_err:
                    best_err, best = err, (bh, bm, bl)
    return best


# ---------------------------------------------------------------------------
# Compressors
# ---------------------------------------------------------------------------

def _haar_random_orthogonal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    Z = torch.randn(d, d, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(Z)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs.unsqueeze(0)


class TurboQuantCompressor:
    """Haar random orthogonal + uniform quantization. SQ paper's TQ baseline."""

    def __init__(self, head_dim: int, avg_bits: float, seed: int):
        self.head_dim = head_dim
        self.avg_bits = avg_bits
        self.bits = max(1, int(round(avg_bits)))
        self.Pi = _haar_random_orthogonal(head_dim, seed=seed)

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        Pi = self.Pi.to(x.device)
        x_rot = x @ Pi
        x_q = _uniform_quantize(x_rot, self.bits)
        return x_q @ Pi.T

    def bits_per_vector(self) -> float:
        return float(self.bits)


class SpectralQuantCompressor:
    """Spectral rotation + mean subtraction + 2-tier uniform quantization.

    use_centered_V=False (default): uncentered eigenvectors, matches SQ exp1 exactly.
    use_centered_V=True: centered-covariance eigenvectors (Scv corrected variant).
        Fixes the "wasted bit on mean direction" issue in vanilla SQ, where the
        top uncentered eigvec aligns with the mean direction and (x - m) @ v_1 ~ 0.
    """

    def __init__(
        self,
        eigenvectors: torch.Tensor,
        mean: torch.Tensor,
        d_eff: float,
        avg_bits: float,
    ):
        self.V = eigenvectors.float()
        self.mean = mean.float()
        self.head_dim = eigenvectors.shape[0]
        self.d_sem = max(1, int(round(d_eff)))
        self.avg_bits = avg_bits
        self.b_high, self.b_low = solve_bits_2tier(
            avg_bits, self.head_dim, self.d_sem
        )

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        V = self.V.to(x.device)
        mean = self.mean.to(x.device)
        x_rot = (x - mean) @ V
        x_sem = _uniform_quantize(x_rot[..., : self.d_sem], self.b_high)
        x_tail = _uniform_quantize(x_rot[..., self.d_sem :], self.b_low)
        x_q = torch.cat([x_sem, x_tail], dim=-1)
        return x_q @ V.T + mean

    def bits_per_vector(self) -> float:
        d = self.head_dim
        return (self.d_sem * self.b_high + (d - self.d_sem) * self.b_low) / d


class ThreeTierCompressor:
    """Spectral rotation + mean sub + 3-tier split at (d_eff_uc, d_eff_centered)."""

    def __init__(
        self,
        eigenvectors: torch.Tensor,
        mean: torch.Tensor,
        d_eff_uc: float,
        d_eff_centered: float,
        avg_bits: float,
    ):
        self.V = eigenvectors.float()
        self.mean = mean.float()
        d = eigenvectors.shape[0]
        self.head_dim = d
        self.d_sig = max(1, min(int(round(d_eff_uc)), d - 2))
        self.d_med_end = max(
            self.d_sig + 1, min(int(round(d_eff_centered)), d)
        )
        self.avg_bits = avg_bits
        self.b_high, self.b_med, self.b_low = solve_bits_3tier(
            avg_bits, d, self.d_sig, self.d_med_end
        )

    def compress_decompress(self, x: torch.Tensor) -> torch.Tensor:
        V = self.V.to(x.device)
        mean = self.mean.to(x.device)
        x_rot = (x - mean) @ V

        parts = [_uniform_quantize(x_rot[..., : self.d_sig], self.b_high)]
        if self.d_med_end > self.d_sig:
            parts.append(
                _uniform_quantize(
                    x_rot[..., self.d_sig : self.d_med_end], self.b_med
                )
            )
        if self.d_med_end < self.head_dim:
            parts.append(
                _uniform_quantize(x_rot[..., self.d_med_end :], self.b_low)
            )
        x_q = torch.cat(parts, dim=-1)
        return x_q @ V.T + mean

    def bits_per_vector(self) -> float:
        d = self.head_dim
        n_sig = self.d_sig
        n_med = max(0, self.d_med_end - self.d_sig)
        n_noise = d - n_sig - n_med
        return (n_sig * self.b_high + n_med * self.b_med + n_noise * self.b_low) / d


# ---------------------------------------------------------------------------
# Calibration (inline)
# ---------------------------------------------------------------------------

def compute_calibration(k_vectors: torch.Tensor) -> Dict:
    """Per-head calibration: uncentered V + mean + d_eff_uc + d_eff_centered.

    k_vectors: (N, d) float32 on CPU.
    """
    k = k_vectors.double()
    n, d = k.shape

    mean = k.mean(dim=0)

    # Uncentered covariance, matches SQ EigenspectralCalibrator
    C_uc = (k.T @ k) / n
    eigvals_uc, V_uc = torch.linalg.eigh(C_uc)
    eigvals_uc = eigvals_uc.flip(0).clamp(min=0.0)
    V_uc = V_uc.flip(1)

    s = eigvals_uc.sum()
    s2 = (eigvals_uc ** 2).sum().clamp(min=1e-18)
    d_eff_uc = float((s * s) / s2)

    # Centered covariance: eigvecs for corrected-SQ variant, also d_eff boundary
    k_c = k - mean
    C_c = (k_c.T @ k_c) / n
    eigvals_c, V_c = torch.linalg.eigh(C_c)
    eigvals_c = eigvals_c.flip(0).clamp(min=0.0)
    V_c = V_c.flip(1)
    sc = eigvals_c.sum()
    sc2 = (eigvals_c ** 2).sum().clamp(min=1e-18)
    d_eff_centered = float((sc * sc) / sc2)

    return {
        "eigenvectors": V_uc.float().contiguous(),
        "eigenvectors_centered": V_c.float().contiguous(),
        "eigenvalues": eigvals_uc.float(),
        "eigenvalues_centered": eigvals_c.float(),
        "mean": mean.float(),
        "d_eff_uc": d_eff_uc,
        "d_eff_centered": d_eff_centered,
    }


# ---------------------------------------------------------------------------
# KV extraction
# ---------------------------------------------------------------------------

def _normalize_past_kv(kv) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Normalize HF past_key_values into a list of (K, V) pairs per layer."""
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if hasattr(kv, "layers"):
        for i in range(len(kv.layers)):
            k = getattr(kv.layers[i], "keys", None)
            v = getattr(kv.layers[i], "values", None)
            if isinstance(k, torch.Tensor) and k.ndim == 4:
                out.append((k[0].detach().cpu().float(), v[0].detach().cpu().float()))
    elif hasattr(kv, "key_cache"):
        for i in range(len(kv.key_cache)):
            k = kv.key_cache[i]
            v = kv.value_cache[i]
            if isinstance(k, torch.Tensor) and k.ndim == 4:
                out.append((k[0].detach().cpu().float(), v[0].detach().cpu().float()))
    elif isinstance(kv, (list, tuple)):
        for item in kv:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                if isinstance(k, torch.Tensor) and k.ndim == 4:
                    out.append((k[0].detach().cpu().float(), v[0].detach().cpu().float()))
    return out


def extract_kv_cache(
    model, tokenizer, texts: List[str], seq_len: int, device: torch.device
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """One forward pass per text; returns per-seq list of per-layer (K, V)."""
    cached = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 16:
            continue
        with torch.no_grad():
            out = model(**enc, use_cache=True, output_hidden_states=False)
        cached.append(_normalize_past_kv(out.past_key_values))
    return cached


def collect_keys_per_head(
    cached: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    sampled_layers: List[int],
    n_kv_heads: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Concatenate K vectors across all sequences, per (layer, head)."""
    buf: Dict[Tuple[int, int], List[torch.Tensor]] = {
        (l, h): [] for l in sampled_layers for h in range(n_kv_heads)
    }
    for seq_kvs in cached:
        for l in sampled_layers:
            if l >= len(seq_kvs):
                continue
            K_layer = seq_kvs[l][0]
            if K_layer.ndim != 3:
                continue
            if K_layer.shape[0] == n_kv_heads:
                for h in range(n_kv_heads):
                    buf[(l, h)].append(K_layer[h])
            elif K_layer.shape[1] == n_kv_heads:
                for h in range(n_kv_heads):
                    buf[(l, h)].append(K_layer[:, h, :])
    return {k: torch.cat(v, dim=0) for k, v in buf.items() if v}


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def softmax_attention_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q: (n_q, d), k: (T, d) -> (n_q, T)."""
    scale = k.shape[-1] ** -0.5
    return torch.softmax(q @ k.T * scale, dim=-1)


def run_assessment(
    eval_cached: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    compressors: Dict[str, Dict[Tuple[int, int], object]],
    sampled_layers: List[int],
    n_kv_heads: int,
    query_probes: torch.Tensor,
    device: torch.device,
) -> Dict[str, Dict]:
    """Run every compressor against every (seq, layer, head); return per-method stats."""
    per_method = {name: {"cos": [], "max_err": []} for name in compressors}

    q = query_probes.to(device)

    for seq_kvs in eval_cached:
        for l in sampled_layers:
            if l >= len(seq_kvs):
                continue
            K_layer = seq_kvs[l][0]
            if K_layer.ndim != 3:
                continue

            if K_layer.shape[0] == n_kv_heads:
                def get_head(h, KL=K_layer):
                    return KL[h]
            elif K_layer.shape[1] == n_kv_heads:
                def get_head(h, KL=K_layer):
                    return KL[:, h, :]
            else:
                continue

            for h in range(n_kv_heads):
                k_head = get_head(h).to(device)
                w_ref = softmax_attention_weights(q, k_head)

                for name, by_head in compressors.items():
                    comp = by_head.get((l, h))
                    if comp is None:
                        continue
                    k_recon = comp.compress_decompress(k_head)
                    w_comp = softmax_attention_weights(q, k_recon)
                    cos = F.cosine_similarity(w_ref, w_comp, dim=-1).mean().item()
                    max_err = (w_ref - w_comp).abs().max().item()
                    per_method[name]["cos"].append(cos)
                    per_method[name]["max_err"].append(max_err)

    return per_method


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bench A: SQ exp1 replication + 3-tier extension",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-seqs", type=int, default=20)
    p.add_argument("--n-probe", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--avg-bits", default="3.0,2.0")
    p.add_argument("--split", choices=["same", "heldout"], default="same")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/bench_v2_weights.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.train(False)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = (
        getattr(cfg, "head_dim", None)
        or cfg.hidden_size // cfg.num_attention_heads
    )
    sampled_layers = sorted(
        np.linspace(0, n_layers - 1, args.n_layers, dtype=int).tolist()
    )
    print(
        f"  layers={n_layers}  kv_heads={n_kv_heads}  head_dim={head_dim}  "
        f"sampled={sampled_layers}"
    )

    # Test data
    print("Loading wikitext test split")
    from datasets import load_dataset

    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="test", streaming=False
    )
    texts = [
        r["text"].strip() for r in ds if len(r["text"].strip()) > 200
    ][: args.n_seqs]
    print(f"  loaded {len(texts)} sequences")

    # KV extraction
    t0 = time.time()
    print("Extracting KV cache")
    all_cached = extract_kv_cache(
        model, tokenizer, texts, args.seq_len, device
    )
    print(f"  {len(all_cached)} sequences cached in {time.time() - t0:.1f}s")

    del model
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # Split
    if args.split == "same":
        calib_cached = all_cached
        eval_cached = all_cached
    else:
        mid = len(all_cached) // 2
        calib_cached = all_cached[:mid]
        eval_cached = all_cached[mid:]
    print(
        f"Split={args.split}  calib={len(calib_cached)}  eval={len(eval_cached)}"
    )

    # Per-head calibration
    print("Computing per-head calibration")
    keys_per_head = collect_keys_per_head(
        calib_cached, sampled_layers, n_kv_heads
    )
    calibration: Dict[Tuple[int, int], Dict] = {}
    for (l, h), k_vecs in keys_per_head.items():
        if k_vecs.shape[0] < head_dim:
            continue
        calibration[(l, h)] = compute_calibration(k_vecs)

    deff_uc = [c["d_eff_uc"] for c in calibration.values()]
    deff_c = [c["d_eff_centered"] for c in calibration.values()]
    print(
        f"  d_eff_uc:        mean={np.mean(deff_uc):.2f}  "
        f"median={np.median(deff_uc):.2f}  min={min(deff_uc):.2f}  "
        f"max={max(deff_uc):.2f}"
    )
    print(
        f"  d_eff_centered:  mean={np.mean(deff_c):.2f}  "
        f"median={np.median(deff_c):.2f}  min={min(deff_c):.2f}  "
        f"max={max(deff_c):.2f}"
    )

    # Fixed probes for all heads (matches SQ exp1)
    g = torch.Generator().manual_seed(args.seed + 1)
    query_probes = F.normalize(
        torch.randn(args.n_probe, head_dim, generator=g), dim=-1
    )

    bits_list = [float(b) for b in args.avg_bits.split(",")]

    all_results: List[Dict] = []
    for bits in bits_list:
        print(f"\n=== avg_bits = {bits:.1f} ===")

        tq = {
            (l, h): TurboQuantCompressor(head_dim, bits, seed=l * 100 + h)
            for l in sampled_layers
            for h in range(n_kv_heads)
        }
        sq = {
            (l, h): SpectralQuantCompressor(
                c["eigenvectors"], c["mean"], c["d_eff_uc"], bits
            )
            for (l, h), c in calibration.items()
        }
        scv = {
            (l, h): SpectralQuantCompressor(
                c["eigenvectors_centered"], c["mean"], c["d_eff_uc"], bits
            )
            for (l, h), c in calibration.items()
        }
        ours = {
            (l, h): ThreeTierCompressor(
                c["eigenvectors"],
                c["mean"],
                c["d_eff_uc"],
                c["d_eff_centered"],
                bits,
            )
            for (l, h), c in calibration.items()
        }
        compressors = {
            f"T-{bits:.1f}": tq,
            f"S-{bits:.1f}": sq,
            f"Scv-{bits:.1f}": scv,
            f"O-{bits:.1f}": ours,
        }

        per_method = run_assessment(
            eval_cached,
            compressors,
            sampled_layers,
            n_kv_heads,
            query_probes,
            device,
        )

        # Sample bit allocations from first head for logging
        sample_key = next(iter(calibration))
        sample_c = calibration[sample_key]
        d_sig = max(1, int(round(sample_c["d_eff_uc"])))
        d_med = min(
            head_dim,
            max(d_sig + 1, int(round(sample_c["d_eff_centered"]))),
        )
        sq_alloc = solve_bits_2tier(bits, head_dim, d_sig)
        ours_alloc = solve_bits_3tier(bits, head_dim, d_sig, d_med)

        for name, stats in per_method.items():
            if not stats["cos"]:
                continue
            cos_mean = float(np.mean(stats["cos"]))
            cos_std = float(np.std(stats["cos"]))
            err_mean = float(np.mean(stats["max_err"]))

            # Effective bits: average bits_per_vector across all heads for this method
            by_head = compressors[name]
            bpv_per_head = [c.bits_per_vector() for c in by_head.values()]
            effective_bits = float(np.mean(bpv_per_head)) if bpv_per_head else float("nan")

            r = {
                "config": name,
                "avg_bits_nominal": bits,
                "avg_bits_effective": effective_bits,
                "cos_mean": cos_mean,
                "cos_std": cos_std,
                "max_err_mean": err_mean,
                "n_samples": len(stats["cos"]),
            }
            if name.startswith("S-") or name.startswith("Scv-"):
                r["bits_alloc_sample"] = {
                    "b_high": sq_alloc[0],
                    "b_low": sq_alloc[1],
                    "d_sem_sample": d_sig,
                }
            elif name.startswith("O-"):
                r["bits_alloc_sample"] = {
                    "b_high": ours_alloc[0],
                    "b_med": ours_alloc[1],
                    "b_low": ours_alloc[2],
                    "d_sig_sample": d_sig,
                    "d_med_end_sample": d_med,
                }
            all_results.append(r)
            print(
                f"  {name:>8}  nom={bits:.1f}  eff={effective_bits:.3f}  "
                f"cos={cos_mean:.5f}  std={cos_std:.5f}  max_err={err_mean:.5f}"
            )

    # Summary table
    print("\n=== Summary ===")
    print(
        f"{'config':>8}  {'nom':>4}  {'eff':>6}  {'cos_mean':>10}  "
        f"{'cos_std':>9}  {'max_err':>9}  {'n':>7}"
    )
    for r in all_results:
        print(
            f"{r['config']:>8}  {r['avg_bits_nominal']:>4.1f}  "
            f"{r['avg_bits_effective']:>6.3f}  "
            f"{r['cos_mean']:>10.5f}  {r['cos_std']:>9.5f}  "
            f"{r['max_err_mean']:>9.5f}  {r['n_samples']:>7d}"
        )

    # Save
    output = {
        "model": args.model,
        "split": args.split,
        "seq_len": args.seq_len,
        "n_seqs": len(all_cached),
        "sampled_layers": sampled_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "n_probe": args.n_probe,
        "calib_summary": {
            "d_eff_uc_mean": float(np.mean(deff_uc)),
            "d_eff_uc_median": float(np.median(deff_uc)),
            "d_eff_centered_mean": float(np.mean(deff_c)),
            "d_eff_centered_median": float(np.median(deff_c)),
        },
        "results": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
