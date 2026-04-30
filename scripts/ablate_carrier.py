#!/usr/bin/env python3
"""
Causal ablation: carrier removal vs controls on Qwen2.5-1.5B.

Forward-pass interventions on the residual stream after a target middle layer
finishes its block but before the next block reads it. Five conditions per
target layer:

  1. baseline          — no hook
  2. carrier_remove    — h' = h - (h · c) c, c = stacked-uncentered top-1
  3. random_remove     — h' = h - (h · r) r, r random unit, averaged over seeds
  4. direction_swap    — h' = h - (h · c) c + (h · c) r, r ⊥ c, averaged
                          (preserves coefficient energy, scrambles direction)
  5. centered_remove   — h' = h - (h · c_c) c_c, c_c = stacked-centered top-1

Response: per-token NLL on the saved 8 × 512 wikitext sequences from
results/carrier_structure_qwen25_1.5b.npz, aggregated to PPL. Predictions
under the carrier-as-infrastructure hypothesis: condition 2 catastrophic,
condition 3 mild (random direction has tiny coefficient mass), condition 4
discriminates whether the specific direction matters or just the energy,
condition 5 says whether the dominant low-rank structure generally is
load-bearing.

Targets: single-layer interventions at L7, L13, L20.

Usage:
  python scripts/ablate_carrier.py --model Qwen/Qwen2.5-1.5B \\
    --carrier-npz results/carrier_structure_qwen25_1.5b.npz \\
    --target-layers 7,13,20 --n-random-seeds 8 \\
    --output results/ablation_qwen25_1.5b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None

from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Carrier basis helpers
# ---------------------------------------------------------------------------

def compute_centered_carrier(
    npz: np.lib.npyio.NpzFile, middle_layers: List[int]
) -> np.ndarray:
    """Sign-aligned mean of per-layer top centered eigenvectors over middle band."""
    aligned: List[np.ndarray] = []
    ref = None
    for l in middle_layers:
        key = f"hidden_state_layer_{l}"
        if key not in npz.files:
            continue
        h = npz[key].astype(np.float64)
        h = h - h.mean(axis=0)
        G = h.T @ h
        _, eigvecs = np.linalg.eigh(G)
        v = eigvecs[:, -1]
        if ref is None:
            ref = v
            aligned.append(v)
        else:
            sgn = float(np.sign(np.dot(v, ref)))
            if sgn == 0:
                sgn = 1.0
            aligned.append(sgn * v)
    if not aligned:
        raise ValueError("no middle-layer hidden states found in npz")
    avg = np.mean(np.stack(aligned, axis=0), axis=0)
    norm = np.linalg.norm(avg)
    if norm < 1e-12:
        raise ValueError("centered carrier has zero norm after averaging")
    return avg / norm


def random_unit_direction(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def random_orthogonal_to(c: np.ndarray, seed: int) -> np.ndarray:
    """Random unit vector orthogonal to c."""
    d = c.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    v = v - np.dot(v, c) * c
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        # Extremely unlikely; resample
        return random_orthogonal_to(c, seed + 1_000_000)
    return v / norm


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def make_remove_hook(direction: torch.Tensor):
    """Project off `direction` from the layer output's residual stream."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        d = direction.to(h.device).to(h.dtype)
        proj = (h * d).sum(dim=-1, keepdim=True)  # (b, s, 1)
        h_new = h - proj * d
        if rest is None:
            return h_new
        return (h_new,) + rest
    return hook


def make_swap_hook(c: torch.Tensor, r: torch.Tensor):
    """Replace c-component with r-component (preserves coefficient magnitude)."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        cd = c.to(h.device).to(h.dtype)
        rd = r.to(h.device).to(h.dtype)
        proj = (h * cd).sum(dim=-1, keepdim=True)
        h_new = h - proj * cd + proj * rd
        if rest is None:
            return h_new
        return (h_new,) + rest
    return hook


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_per_token_nll(
    model, token_ids: np.ndarray, device: torch.device
) -> np.ndarray:
    """Returns (n_seqs, seq_len - 1) array of per-position NLL.

    NLL[i, t] = -log P(token_ids[i, t+1] | token_ids[i, :t+1]) under the model.
    """
    nlls = []
    for seq in token_ids:
        ids = torch.tensor(seq, device=device, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=False, return_dict=True)
        logits = out.logits  # (1, seq_len, vocab)
        # next-token prediction
        targets = ids[:, 1:]                 # (1, seq_len - 1)
        log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # fp32 for stability
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq_len - 1)
        nlls.append(nll[0].cpu().numpy())
        del out, logits, log_probs, nll
    return np.stack(nlls, axis=0)


def stats_from_nll(nll: np.ndarray) -> Dict:
    flat = nll.reshape(-1)
    return {
        "mean_nll": float(flat.mean()),
        "ppl": float(np.exp(flat.mean())),
        "median_nll": float(np.median(flat)),
        "p95_nll": float(np.percentile(flat, 95)),
        "n_tokens": int(flat.size),
    }


def per_position_mean(nll: np.ndarray) -> List[float]:
    """Mean NLL per within-sequence position, averaged across n_seqs."""
    return nll.mean(axis=0).tolist()


# ---------------------------------------------------------------------------
# Run a single condition
# ---------------------------------------------------------------------------

def run_condition(
    model,
    target_layer: int,
    hook_fn,
    token_ids: np.ndarray,
    device: torch.device,
    label: str,
) -> Dict:
    """Register hook, eval, unregister. Returns stats dict."""
    if hook_fn is None:
        nll = eval_per_token_nll(model, token_ids, device)
    else:
        # Qwen2.5: model.model.layers[l] — same for Mistral, Gemma, Llama
        target_module = model.model.layers[target_layer]
        handle = target_module.register_forward_hook(hook_fn)
        try:
            nll = eval_per_token_nll(model, token_ids, device)
        finally:
            handle.remove()

    s = stats_from_nll(nll)
    s["per_position_mean_nll"] = per_position_mean(nll)
    s["target_layer"] = target_layer
    s["condition"] = label
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--carrier-npz", required=True,
                    help="NPZ from analyze_carrier_structure.py")
    ap.add_argument("--target-layers", default="7,13,20")
    ap.add_argument("--n-random-seeds", type=int, default=8,
                    help="Random direction control: number of seeds to average")
    ap.add_argument("--device",
                    default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    target_layers = [int(s) for s in args.target_layers.split(",")]

    print(f"Loading carrier data: {args.carrier_npz}")
    npz = np.load(args.carrier_npz)
    if "V_20" not in npz.files:
        raise SystemExit("npz missing V_20 (carrier basis)")
    if "token_ids" not in npz.files:
        raise SystemExit("npz missing token_ids")

    c_uncentered = npz["V_20"][0]  # rank-1 carrier
    token_ids = npz["token_ids"]
    D = c_uncentered.shape[0]
    print(f"  carrier dim: {D}")
    print(f"  token_ids shape: {token_ids.shape}")

    # Detect collapsed band from baseline_per_layer if available;
    # otherwise hardcode L2-L26 for Qwen2.5-1.5B.
    middle_layers = [l for l in range(2, 27) if f"hidden_state_layer_{l}" in npz.files]
    if not middle_layers:
        middle_layers = [l for l in range(token_ids.shape[1] // 16)]
    print(f"  middle layers used for centered carrier: "
          f"L{middle_layers[0]}-L{middle_layers[-1]} ({len(middle_layers)} layers)")

    print("Computing centered carrier from saved hidden states")
    c_centered = compute_centered_carrier(npz, middle_layers)
    align_uc_c = float(np.abs(np.dot(c_uncentered, c_centered)))
    print(f"  alignment (uncentered c · centered c_c): {align_uc_c:.4f}")

    # Convert to torch tensors
    c_uc_t = torch.tensor(c_uncentered, dtype=torch.float32)
    c_c_t = torch.tensor(c_centered, dtype=torch.float32)

    print(f"\nLoading {args.model} in {args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    model.train(False)

    device = torch.device(args.device)

    # Sanity: confirm a transformer layer module exists
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise SystemExit("Unexpected model structure (no model.model.layers)")
    n_layers = len(model.model.layers)
    print(f"  {n_layers} transformer layers")

    results: List[Dict] = []

    # Baseline once (independent of target_layer)
    print("\n[baseline] no intervention")
    base = run_condition(model, -1, None, token_ids, device, "baseline")
    base["target_layer"] = None
    print(f"  baseline PPL = {base['ppl']:.4f}")
    results.append(base)

    for target in target_layers:
        if target >= n_layers:
            print(f"  skip L{target} (out of range)")
            continue
        print(f"\n=== Target layer L{target} ===")

        # 2. carrier_remove
        hook = make_remove_hook(c_uc_t)
        s = run_condition(model, target, hook, token_ids, device, "carrier_remove")
        print(f"  carrier_remove        PPL={s['ppl']:.4f}  ΔPPL/baseline={s['ppl']/base['ppl']:.2f}×")
        results.append(s)

        # 3. random_remove (averaged over seeds)
        random_remove_runs = []
        for seed in range(args.n_random_seeds):
            r = random_unit_direction(D, seed=10_000 + seed)
            r_t = torch.tensor(r, dtype=torch.float32)
            hook = make_remove_hook(r_t)
            sr = run_condition(model, target, hook, token_ids, device,
                               f"random_remove_seed{seed}")
            random_remove_runs.append(sr["ppl"])
        agg = {
            "condition": "random_remove",
            "target_layer": target,
            "ppl_mean": float(np.mean(random_remove_runs)),
            "ppl_std": float(np.std(random_remove_runs)),
            "n_seeds": args.n_random_seeds,
            "per_seed_ppl": random_remove_runs,
        }
        print(f"  random_remove (n={args.n_random_seeds}) PPL={agg['ppl_mean']:.4f} "
              f"± {agg['ppl_std']:.4f}  ΔPPL={agg['ppl_mean']/base['ppl']:.2f}×")
        results.append(agg)

        # 4. direction_swap (averaged over seeds, r ⊥ c_uncentered)
        swap_runs = []
        for seed in range(args.n_random_seeds):
            r = random_orthogonal_to(c_uncentered, seed=20_000 + seed)
            r_t = torch.tensor(r, dtype=torch.float32)
            hook = make_swap_hook(c_uc_t, r_t)
            sr = run_condition(model, target, hook, token_ids, device,
                               f"direction_swap_seed{seed}")
            swap_runs.append(sr["ppl"])
        agg = {
            "condition": "direction_swap",
            "target_layer": target,
            "ppl_mean": float(np.mean(swap_runs)),
            "ppl_std": float(np.std(swap_runs)),
            "n_seeds": args.n_random_seeds,
            "per_seed_ppl": swap_runs,
        }
        print(f"  direction_swap (n={args.n_random_seeds}) PPL={agg['ppl_mean']:.4f} "
              f"± {agg['ppl_std']:.4f}  ΔPPL={agg['ppl_mean']/base['ppl']:.2f}×")
        results.append(agg)

        # 5. centered_remove
        hook = make_remove_hook(c_c_t)
        s = run_condition(model, target, hook, token_ids, device, "centered_remove")
        print(f"  centered_remove       PPL={s['ppl']:.4f}  ΔPPL={s['ppl']/base['ppl']:.2f}×")
        results.append(s)

        if args.device == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

    out = {
        "model": args.model,
        "carrier_npz": args.carrier_npz,
        "target_layers": target_layers,
        "n_random_seeds": args.n_random_seeds,
        "carrier_uncentered_centered_alignment": align_uc_c,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
