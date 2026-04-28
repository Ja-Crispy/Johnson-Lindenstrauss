#!/usr/bin/env python3
"""
Layer-structure pilot — spectral decomposition of residual-stream activations.

Methodology probe in response to Muyu He's critique that adjacent-layer cosine
similarity is a misleading proxy for layer utility. We compute four metrics
per layer transition and look for disagreements with the classical cosine.

Per layer l:
  d_eff_uc(h_l)  — concentration of activation including dominant mean drift
  d_eff_c(h_l)   — variance structure of activation after removing the mean

Per transition l -> l+1 (Δ_l = h_{l+1} - h_l):
  cos(h_l, h_{l+1})         — classical adjacent-layer angle (baseline)
  ||Δ_l|| / ||h_l||         — relative update magnitude
  d_eff_c(Δ_l)              — update complexity (spectrum of the layer's contribution)
  ||Δ_perp|| / ||Δ_l||      — update novelty: fraction of Δ orthogonal to the
                              top-k centered subspace of h_l, where
                              k = ceil(d_eff_c(h_l)). High = layer injects new
                              directions; low = layer refines existing structure.

Pilot scope: one model (Qwen2.5-1.5B), 8 wikitext sequences x 512 tokens.
NOT a layer-utility metric. NOT connected to the K/V asymmetry memo.

Usage:
  python scripts/analyze_layer_structure.py --model Qwen/Qwen2.5-1.5B --device mps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None

from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent


def load_calibration_text(n_chars: int = 200_000) -> str:
    candidates = [
        ROOT / "vendor" / "llama-cpp-turboquant" / "wikitext-2-raw" / "wiki.test.raw",
        ROOT / "wikitext-2-raw" / "wiki.test.raw",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text()[:n_chars]
    raise FileNotFoundError("wikitext-2 not found in expected locations")


def extract_hidden_states(model, tokenizer, n_seqs, seq_len, device):
    """Returns list of n_layers numpy arrays, each (total_positions, hidden_size).

    Uses output_hidden_states=True. Layer 0 is the embedding output;
    layers 1..n are post-residual outputs of each transformer block.
    """
    text = load_calibration_text()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < n_seqs * seq_len:
        raise ValueError(f"Need {n_seqs * seq_len} tokens, only have {len(tokens)}")

    accumulated = None
    for i in range(n_seqs):
        chunk = tokens[i * seq_len : (i + 1) * seq_len]
        ids = torch.tensor([chunk], device=device)
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if accumulated is None:
            accumulated = [[] for _ in hs]
        for li, h in enumerate(hs):
            accumulated[li].append(h[0].detach().cpu().float())
        del out, hs
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

    return [torch.cat(layer_chunks, dim=0).numpy() for layer_chunks in accumulated]


def participation_ratio(eigvals: np.ndarray) -> float:
    s = float(eigvals.sum())
    s2 = float((eigvals ** 2).sum())
    if s2 <= 0:
        return float(len(eigvals))
    return (s ** 2) / s2


def per_layer_stats(h: np.ndarray, top_k_keep: int = 200) -> dict:
    """h: (N, D). Returns d_eff stats and trimmed top-k centered eigenvectors."""
    h64 = h.astype(np.float64)

    _, sigma_uc, _ = np.linalg.svd(h64, full_matrices=False)
    eig_uc = sigma_uc ** 2
    d_eff_uc = participation_ratio(eig_uc)

    mean = h64.mean(axis=0)
    h_c = h64 - mean
    _, sigma_c, Vt_c = np.linalg.svd(h_c, full_matrices=False)
    eig_c = sigma_c ** 2
    d_eff_c = participation_ratio(eig_c)

    Vt_c_trimmed = Vt_c[: min(top_k_keep, Vt_c.shape[0])]

    return {
        "d_eff_uc": d_eff_uc,
        "d_eff_c": d_eff_c,
        "mean": mean,
        "Vt_c": Vt_c_trimmed,
        "norm_mean": float(np.linalg.norm(h64, axis=1).mean()),
    }


def transition_stats(h_l: np.ndarray, h_lp1: np.ndarray, layer_l: dict) -> dict:
    """Compute per-transition metrics."""
    h_l64 = h_l.astype(np.float64)
    delta = h_lp1.astype(np.float64) - h_l64

    norm_h = np.linalg.norm(h_l64, axis=1)
    norm_delta = np.linalg.norm(delta, axis=1)
    rel_magnitude = float((norm_delta / np.clip(norm_h, 1e-12, None)).mean())

    h_l_n = h_l64 / np.clip(norm_h[:, None], 1e-12, None)
    h_lp1_n = h_lp1.astype(np.float64) / np.clip(
        np.linalg.norm(h_lp1.astype(np.float64), axis=1, keepdims=True), 1e-12, None
    )
    cos_sim = float((h_l_n * h_lp1_n).sum(axis=1).mean())

    delta_mean = delta.mean(axis=0)
    delta_c = delta - delta_mean
    _, sigma_d, _ = np.linalg.svd(delta_c, full_matrices=False)
    eig_d = sigma_d ** 2
    d_eff_c_delta = participation_ratio(eig_d)

    k = max(1, int(np.ceil(layer_l["d_eff_c"])))
    k = min(k, layer_l["Vt_c"].shape[0])
    Vk = layer_l["Vt_c"][:k]
    proj = delta @ Vk.T
    delta_parallel = proj @ Vk
    delta_perp = delta - delta_parallel
    norm_perp = np.linalg.norm(delta_perp, axis=1)
    novelty_ratio = float((norm_perp / np.clip(norm_delta, 1e-12, None)).mean())

    return {
        "cos_sim": cos_sim,
        "rel_magnitude": rel_magnitude,
        "d_eff_c_delta": d_eff_c_delta,
        "novelty_ratio": novelty_ratio,
        "k_used": int(k),
    }


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
        "--output", default="results/layer_structure_qwen25_1.5b.json"
    )
    args = parser.parse_args()

    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if args.device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device)
    model.train(False)

    print(f"Extracting hidden states: {args.n_seqs} seqs x {args.seq_len} tokens")
    hidden_states = extract_hidden_states(
        model, tokenizer, args.n_seqs, args.seq_len, args.device
    )
    n_layers = len(hidden_states)
    N, D = hidden_states[0].shape
    print(f"  layers={n_layers} (incl. embedding output)  positions={N}  hidden_dim={D}")

    del model
    if args.device == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    print("Computing per-layer stats")
    layer_stats = []
    for li, h in enumerate(hidden_states):
        s = per_layer_stats(h)
        s["layer"] = li
        layer_stats.append(s)
        print(
            f"  L{li:>2}: d_eff_uc={s['d_eff_uc']:.2f}  "
            f"d_eff_c={s['d_eff_c']:.2f}  ||h||={s['norm_mean']:.2f}"
        )

    print("Computing per-transition stats")
    trans_stats = []
    for li in range(n_layers - 1):
        t = transition_stats(hidden_states[li], hidden_states[li + 1], layer_stats[li])
        t["layer_from"] = li
        t["layer_to"] = li + 1
        trans_stats.append(t)
        print(
            f"  L{li:>2}->{li+1:<2}: cos={t['cos_sim']:.4f}  "
            f"rel_mag={t['rel_magnitude']:.4f}  "
            f"d_eff_c(Δ)={t['d_eff_c_delta']:.2f}  "
            f"perp_frac={t['novelty_ratio']:.4f}  k={t['k_used']}"
        )

    payload = {
        "model": args.model,
        "n_seqs": args.n_seqs,
        "seq_len": args.seq_len,
        "n_layers": n_layers,
        "hidden_size": D,
        "n_positions": N,
        "per_layer": [
            {k: v for k, v in s.items() if k not in ("mean", "Vt_c")}
            for s in layer_stats
        ],
        "per_transition": trans_stats,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
