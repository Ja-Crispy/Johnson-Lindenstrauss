#!/usr/bin/env python3
"""
jl_experiments.py

Comprehensive experiments for Johnson-Lindenstrauss reproduction and exploration.

Modes:
 A - reproduce: optimize N unit vectors in k dims with different losses (observe pathologies)
 B - compare: compare Gaussian JL, Hadamard/FJLT, and learned linear projection; estimate empirical C
 C - losses: battery test (exp_penalty, hinge, gram, rbf, logbar) with temperature scheduling and sampling
 D - scaling: grid over (N,k) to collect max/median angle statistics

This script will try to import functions from the uploaded files:
 - /mnt/data/JL optimizer.py
 - /mnt/data/Hadamard random projection.py

If they exist, some functions will be used. Otherwise, internal fallbacks are used.

Author: generated for user by ChatGPT
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from functools import partial

import numpy as np
import torch
from tqdm import trange, tqdm

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ------------------ Helpers & utils ------------------

def try_import_module_from_path(path, module_name):
    """Dynamically import Python module from path; return module or None."""
    if not path:
        return None
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[warn] Couldn't import {path}: {e}")
        return None

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if x is None:
        return None
    return x.to(device)

def normalize_rows(X, eps=1e-12):
    # X: (..., d)
    norm = X.norm(dim=-1, keepdim=True).clamp_min(eps)
    return X / norm

# ------------------ Fast Walsh-Hadamard (FWHT) ------------------
# We'll provide an internal FWHT in case Hadamard file isn't present.

def fwht_batch(x: torch.Tensor):
    """
    In-place FWHT on last dimension. x shape (..., n) where n is power of 2.
    Returns transformed tensor (same view).
    """
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n & (n - 1) == 0, "FWHT requires power-of-two length"
    x = x.view(-1, n)  # (batch, n)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[:, i:i+h]
            b = x[:, i+h:i+2*h]
            x[:, i:i+h] = a + b
            x[:, i+h:i+2*h] = a - b
        h *= 2
    return x.view(*orig_shape)

# ------------------ Pair sampling & evaluation ------------------

def sample_pairs(N, M, device='cpu'):
    """Sample M random unordered pairs (i,j) with i!=j."""
    i = torch.randint(0, N, (M,), device=device)
    j = torch.randint(0, N, (M,), device=device)
    neq = (i != j)
    if not torch.all(neq):
        # ensure i != j by resampling duplicates
        while not torch.all(neq):
            idxs = torch.where(~neq)[0]
            i[idxs] = torch.randint(0, N, (idxs.shape[0],), device=device)
            j[idxs] = torch.randint(0, N, (idxs.shape[0],), device=device)
            neq = (i != j)
    return i, j

def evaluate_pairwise_stats(X, pairs=None, eval_pairs=100000, device='cuda'):
    """
    Evaluate pairwise dot statistics by sampling 'eval_pairs' random pairs if
    pairs not provided. X is (N,k) tensor (unit-normalized).
    Returns dict of stats (mean, median, quantiles, max_angle_deg, hist)
    """
    N = X.shape[0]
    if pairs is None:
        i, j = sample_pairs(N, min(eval_pairs, N*(N-1)//2), device=device)
    else:
        i, j = pairs
    dots = (X[i] * X[j]).sum(dim=1).float().cpu().numpy()
    # clip to [-1,1]
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))  # degrees
    stats = {
        "dot_mean": float(dots.mean()),
        "dot_median": float(np.median(dots)),
        "dot_std": float(dots.std()),
        "angle_mean": float(angles.mean()),
        "angle_median": float(np.median(angles)),
        "angle_5pct": float(np.percentile(angles, 5)),
        "angle_95pct": float(np.percentile(angles, 95)),
        "angle_max": float(np.max(angles)),
        "angle_min": float(np.min(angles)),
    }
    return stats, dots, angles

# ------------------ Loss functions for vector optimization (mode A/C) ------------------

def loss_exp_penalty(dots, alpha=20.0):
    # Yoder style: exp(alpha * dot^2). Use clamping for numerical safety.
    # dots in [-1,1]
    val = (alpha * (dots ** 2)).clamp(max=50.0)  # avoid huge exponent
    return torch.exp(val).sum()

def loss_hinge_margin(dots, tau=0.01):
    # hinge on |dot| > tau : sum(max(0, |dot| - tau))
    return torch.nn.functional.relu(dots.abs() - tau).sum()

def loss_gram_fro(X):
    # X: (N,k) unit normalized rows. Gram = X X^T -> we want Gram approx I
    G = X @ X.t()
    N = G.shape[0]
    I = torch.eye(N, device=G.device)
    # penalize off-diagonals only
    off = (G - I)
    return (off ** 2).sum()

def loss_rbf_repulsive(dots, beta=10.0):
    # repulsive potential: sum(exp(beta * (dot - 1))) but shift so it's positive
    # This penalizes dots near 1 strongly.
    # Use stable clamp
    vals = beta * (dots - 1.0)
    vals = vals.clamp(min=-50.0, max=50.0)
    return torch.exp(vals).sum()

def loss_log_barrier(dots, eps=1e-3):
    # log barrier penalizing dots -> 1: sum(-log(1 - |dot| + eps))
    inside = (1.0 - dots.abs()).clamp(min=1e-6)
    return (-torch.log(inside)).sum()

# ------------------ Vector optimizer (mode A & C core) ------------------

def optimize_vectors(N, k, device='cuda', iters=2000, pairs_per_step=20000,
                     eval_pairs=100000, lr=1e-3, loss_type='exp_penalty',
                     loss_kwargs=None, reinit=True, seed=0, verbose=True):
    """
    Optimize N unit vectors in k dims to maximize mutual "orthogonality"
    via different loss functions. Returns final X (N,k), history dict.
    """
    set_seed(seed)
    device = torch.device(device)
    # initialize random gaussian then normalize
    X = torch.randn(N, k, device=device)
    X = normalize_rows(X)
    X = X.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([X], lr=lr)
    history = {"loss": [], "eval_stats": []}

    loss_kwargs = loss_kwargs or {}

    for step in range(iters):
        # sample pairs
        m = min(pairs_per_step, N*(N-1)//2)
        i, j = sample_pairs(N, m, device=device)
        dots = (X[i] * X[j]).sum(dim=1)
        if loss_type == 'exp_penalty':
            loss = loss_exp_penalty(dots, **loss_kwargs)
        elif loss_type == 'hinge':
            loss = loss_hinge_margin(dots, **loss_kwargs)
        elif loss_type == 'rbf':
            loss = loss_rbf_repulsive(dots, **loss_kwargs)
        elif loss_type == 'logbar':
            loss = loss_log_barrier(dots, **loss_kwargs)
        elif loss_type == 'gram':
            # gram needs full X (but we'll compute on minibatch for speed if needed)
            # For small N it's fine to compute full
            if N <= 5000:
                loss = loss_gram_fro(normalize_rows(X))
            else:
                # approximate Gram penalty using sampled subset
                idx = torch.randperm(N, device=device)[:min(1000, N)]
                loss = loss_gram_fro(normalize_rows(X[idx]))
        else:
            raise ValueError("unknown loss type")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([X], 1.0)
        opt.step()
        # renormalize rows to keep on sphere
        with torch.no_grad():
            X.data = normalize_rows(X.data)

        # logging & eval
        if (step % max(1, iters // 20) == 0) or (step == iters - 1):
            with torch.no_grad():
                stats, _, _ = evaluate_pairwise_stats(X.detach(), eval_pairs=eval_pairs, device=device)
            history["loss"].append(float(loss.item()))
            history["eval_stats"].append(stats)
            if verbose:
                print(f"[step {step}/{iters}] loss={loss.item():.4e} angle_median={stats['angle_median']:.3f} angle_max={stats['angle_max']:.3f}")

    return X.detach(), history

# ------------------ Gaussian JL baseline (random projection) ------------------

def gaussian_jl_project(X_in: torch.Tensor, k: int, device='cpu'):
    """
    X_in: (N, D) original data in D dims (or identity vectors if you test directions).
    Returns projected X_proj (N, k) using Gaussian random matrix.
    """
    N, D = X_in.shape
    device = torch.device(device)
    G = torch.randn(D, k, device=device) / math.sqrt(k)  # scale to preserve norm
    X_proj = X_in @ G
    return X_proj, G

# ------------------ Hadamard / FJLT projection that uses uploaded file when possible ---------------

def load_hadamard_impl(hadamard_path=None):
    """
    Try to import user-provided Hadamard random projection implementation.
    If not available, return None and we use fallback.
    """
    if hadamard_path and os.path.exists(hadamard_path):
        mod = try_import_module_from_path(hadamard_path, "user_hadamard")
        if mod is not None:
            # we expect the module to expose something like 'hadamard_project' or similar
            # try common names; otherwise return module for user inspection
            if hasattr(mod, "hadamard_random_projection"):
                return getattr(mod, "hadamard_random_projection")
            if hasattr(mod, "hadamard_project"):
                return getattr(mod, "hadamard_project")
            # return module to call custom functions later
            return mod
    return None

def hadamard_project_fallback(X_in: torch.Tensor, k: int, device='cpu'):
    """
    Simple FJLT-like projection using randomly sign-flip, FWHT, and subsample k coordinates.
    This is not a production FJLT but is a functional fast-ish structured projection.
    """
    N, D = X_in.shape
    # pad D to next power of two
    p = 1 << (D - 1).bit_length()
    if p != D:
        # pad zeros
        pad = p - D
        X = torch.cat([X_in, torch.zeros(N, pad, device=X_in.device)], dim=1)
    else:
        X = X_in
    # random sign flips
    signs = (torch.randint(0, 2, (p,), device=X.device).float() * 2.0 - 1.0)
    X = X * signs.unsqueeze(0)
    # FWHT per sample
    X = fwht_batch(X)
    # normalize
    X = X / math.sqrt(p)
    # now randomly sample k coordinates out of p with scaling
    idx = torch.randperm(p, device=X.device)[:k]
    X_proj = X[:, idx] * (math.sqrt(p / k))
    return X_proj

# ------------------ Learned linear projection (optimize W) for experiment B ------------------

def learn_projection_matrix(X_in: torch.Tensor, k: int, device='cuda', iters=2000, lr=1e-3,
                            pairs_per_step=20000, eval_pairs=100000, seed=0, verbose=True):
    """
    Learn a linear projection matrix W (D -> k) to minimize worst-case distortion
    on random pairs sampled from X_in. We'll optimize to reduce | ||x_i - x_j||^2 - ||Px_i - Px_j||^2 |
    or equivalently preserve dot products.
    X_in is (N, D). We will normalize input vectors first.
    Returns W (D,k) and history.
    """
    set_seed(seed)
    device = torch.device(device)
    N, D = X_in.shape
    X = X_in.to(device)
    X = normalize_rows(X)

    W = torch.randn(D, k, device=device) * (1.0 / math.sqrt(k))
    W = W.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    history = {"loss": [], "eval": []}

    for step in range(iters):
        m = min(pairs_per_step, N*(N-1)//2)
        i, j = sample_pairs(N, m, device=device)
        xi = X[i]
        xj = X[j]
        # original squared distances
        dij2 = (xi - xj).pow(2).sum(dim=1)
        # projected
        pi = xi @ W
        pj = xj @ W
        pdij2 = (pi - pj).pow(2).sum(dim=1)
        # loss: squared error of distance preservation
        loss = ((pdij2 - dij2) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step % max(1, iters // 20) == 0) or (step == iters - 1):
            # evaluate on eval_pairs
            stats = {}
            with torch.no_grad():
                # sample eval pairs
                i_e, j_e = sample_pairs(N, min(eval_pairs, N*(N-1)//2), device=device)
                xi_e = X[i_e]; xj_e = X[j_e]
                dij2_e = (xi_e - xj_e).pow(2).sum(dim=1)
                pi_e = xi_e @ W; pj_e = xj_e @ W
                pdij2_e = (pi_e - pj_e).pow(2).sum(dim=1)
                rel_err = (pdij2_e - dij2_e) / (dij2_e + 1e-9)
                stats["rel_err_mean"] = float(rel_err.abs().mean().cpu().numpy())
                stats["rel_err_max"] = float(rel_err.abs().max().cpu().numpy())
            history["loss"].append(float(loss.item()))
            history["eval"].append(stats)
            if verbose:
                print(f"[learn_proj step {step}/{iters}] loss={loss.item():.4e} rel_err_mean={stats['rel_err_mean']:.4e} rel_err_max={stats['rel_err_max']:.4e}")
    return W.detach(), history

# ------------------ Mode B driver: find smallest k meeting distortion epsilon ---------------

def find_k_for_epsilon(X_data: torch.Tensor, method='gaussian', eps=0.1,
                       kmin=16, kmax=2048, device='cuda', hadamard_impl=None,
                       learn_iters=1500, verbose=True):
    """
    For given method ('gaussian','hadamard','learned') try k values and test if
    projection of data preserves pairwise distances within eps on sample of pairs.
    We'll test k on geometric progression or linear grid between kmin and kmax.
    Returns dict with best_k, and per-k metrics.
    """
    device = torch.device(device)
    N, D = X_data.shape
    metrics = []
    ks = list(range(kmin, kmax + 1, max(1, (kmax - kmin) // 32)))
    if kmax - kmin < 64:
        ks = list(range(kmin, kmax + 1, 1))
    for k in ks:
        if method == 'gaussian':
            Xp, _ = gaussian_jl_project(X_data.to(device), k, device=device)
        elif method == 'hadamard':
            if hadamard_impl is not None:
                try:
                    Xp = hadamard_impl(X_data.to(device), k)
                except Exception as e:
                    print("[warn] hadamard_impl failed, falling back:", e)
                    Xp = hadamard_project_fallback(X_data.to(device), k, device=device)
            else:
                Xp = hadamard_project_fallback(X_data.to(device), k, device=device)
        elif method == 'learned':
            W, _ = learn_projection_matrix(X_data, k, device=device, iters=learn_iters, verbose=verbose)
            Xp = (X_data.to(device) @ W)
        else:
            raise ValueError("unknown method")
        # evaluate worst-case relative distortion on sample pairs
        stats = {}
        with torch.no_grad():
            i_e, j_e = sample_pairs(N, min(50000, N*(N-1)//2), device=device)
            xi_e = X_data[i_e].to(device); xj_e = X_data[j_e].to(device)
            dij2 = (xi_e - xj_e).pow(2).sum(dim=1)
            pi_e = Xp[i_e]; pj_e = Xp[j_e]
            pdij2 = (pi_e - pj_e).pow(2).sum(dim=1)
            rel_err = (pdij2 - dij2) / (dij2 + 1e-9)
            # distortion defined as max |rel_err|; we check <= eps
            stats["rel_err_mean"] = float(rel_err.abs().mean().cpu().numpy())
            stats["rel_err_max"] = float(rel_err.abs().max().cpu().numpy())
        stats["k"] = k
        metrics.append(stats)
        if verbose:
            print(f"[k={k}] rel_err_max={stats['rel_err_max']:.4e} rel_err_mean={stats['rel_err_mean']:.4e}")
        if stats["rel_err_max"] <= eps:
            return k, metrics
    # not found
    return None, metrics

# ------------------ Mode drivers ------------------

def mode_A(args, hadamard_impl=None, jlopt_mod=None):
    # reproduce vector optimization experiments (pathology and fixes)
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    print(f"[mode A] device={device}")
    results = {}
    seeds = args.seeds if args.seeds else [args.seed]
    losses = [args.loss] if args.loss else ['exp_penalty']

    for seed in seeds:
        for loss in losses:
            print(f"-> seed={seed} loss={loss}")
            loss_kwargs = {}
            if loss == 'exp_penalty':
                loss_kwargs = {'alpha': args.alpha}
            elif loss == 'hinge':
                loss_kwargs = {'tau': args.tau}
            elif loss == 'rbf':
                loss_kwargs = {'beta': args.beta}
            X_final, history = optimize_vectors(N=args.N, k=args.k, device=device, iters=args.iters,
                                                pairs_per_step=args.pairs_per_step, eval_pairs=args.eval_pairs,
                                                lr=args.lr, loss_type=loss, loss_kwargs=loss_kwargs, seed=seed, verbose=True)
            key = f"seed{seed}_{loss}"
            results[key] = {
                "history": history,
                "N": args.N,
                "k": args.k,
            }
            # save vectors optionally
            if args.save_vectors:
                fname = f"vectors_{key}.npy"
                np.save(fname, X_final.cpu().numpy())
                results[key]["vectors_file"] = fname
    # dump results
    out = args.out or "results_A.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[mode A] saved results to {out}")
    return results

def mode_B(args, hadamard_impl=None, jlopt_mod=None):
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    device = torch.device(device)
    print(f"[mode B] device={device}")
    # create random data in D dims (or use small synthetic set)
    D = args.D
    N = args.N
    set_seed(args.seed)
    X = torch.randn(N, D, device=device)
    X = normalize_rows(X)
    methods = ['gaussian', 'hadamard', 'learned']
    hadamard_impl_fn = hadamard_impl
    results = {}
    for method in methods:
        print(f"== method {method} ==")
        k_found, metrics = find_k_for_epsilon(X, method=method, eps=args.eps, kmin=args.kmin, kmax=args.kmax,
                                             device=device, hadamard_impl=hadamard_impl_fn, learn_iters=args.learn_iters,
                                             verbose=not args.quiet)
        results[method] = {"best_k": k_found, "metrics": metrics}
    # compute empirical C using k = (C / eps^2) * log(N) => C = k * eps^2 / log(N)
    for method in results:
        k = results[method]["best_k"]
        if k is not None:
            C = k * (args.eps ** 2) / math.log(N)
            results[method]["empirical_C"] = float(C)
        else:
            results[method]["empirical_C"] = None
    out = args.out or "results_B.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[mode B] saved results to {out}")
    return results

def mode_C(args, hadamard_impl=None, jlopt_mod=None):
    # run multiple loss variants and compare
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    print(f"[mode C] device={device}")
    results = {}
    losses = args.losses.split(",") if args.losses else ['exp_penalty', 'hinge', 'gram', 'rbf', 'logbar']
    for loss in losses:
        print(f"-> loss {loss}")
        kw = {}
        if loss == 'exp_penalty':
            kw = {'alpha': args.alpha}
        elif loss == 'hinge':
            kw = {'tau': args.tau}
        elif loss == 'rbf':
            kw = {'beta': args.beta}
        X_final, history = optimize_vectors(N=args.N, k=args.k, device=device, iters=args.iters,
                                            pairs_per_step=args.pairs_per_step, eval_pairs=args.eval_pairs,
                                            lr=args.lr, loss_type=loss, loss_kwargs=kw, seed=args.seed, verbose=True)
        results[loss] = {"history": history}
        if args.save_vectors:
            fname = f"vectors_C_{loss}.npy"
            np.save(fname, X_final.cpu().numpy())
            results[loss]["vectors_file"] = fname
    out = args.out or "results_C.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[mode C] saved results to {out}")
    return results

def mode_D(args, hadamard_impl=None, jlopt_mod=None):
    # scaling grid: run optimized vector experiment for grid of Ns and ks
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    print(f"[mode D] device={device}")
    Ns = [int(x) for x in args.grid_Ns.split(",")]
    ks = [int(x) for x in args.grid_ks.split(",")]
    results = {}
    for N in Ns:
        for k in ks:
            print(f"Grid point N={N} k={k}")
            X_final, history = optimize_vectors(N=N, k=k, device=device, iters=args.iters,
                                                pairs_per_step=args.pairs_per_step, eval_pairs=args.eval_pairs,
                                                lr=args.lr, loss_type=args.loss, loss_kwargs={'alpha': args.alpha}, seed=args.seed, verbose=False)
            stats, dots, angles = evaluate_pairwise_stats(X_final, eval_pairs=args.eval_pairs, device=device)
            key = f"N{N}_k{k}"
            results[key] = {"stats": stats, "history": history}
            if args.save_vectors:
                fname = f"vectors_D_{key}.npy"
                np.save(fname, X_final.cpu().numpy())
                results[key]["vectors_file"] = fname
            print(f" -> angle_median={stats['angle_median']:.3f} angle_max={stats['angle_max']:.3f}")
    out = args.out or "results_D.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[mode D] saved results to {out}")
    return results

# ------------------ CLI & main ------------------

def build_parser():
    p = argparse.ArgumentParser(description="JL experiments driver")
    p.add_argument("--mode", type=str, choices=['A', 'B', 'C', 'D'], required=True)
    # general
    p.add_argument("--device", type=str, default='auto', help="cuda|cpu|auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs='*', help="list of seeds (for mode A)", default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save_vectors", action='store_true')
    # A/C/D params
    p.add_argument("--N", type=int, default=5000, help="number of vectors")
    p.add_argument("--k", type=int, default=200, help="embedding dimension")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--pairs_per_step", type=int, default=20000)
    p.add_argument("--eval_pairs", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    # losses params
    p.add_argument("--loss", type=str, default='exp_penalty', help="loss type for A/D")
    p.add_argument("--losses", type=str, default="exp_penalty,hinge,gram,rbf,logbar", help="comma separated for mode C")
    p.add_argument("--alpha", type=float, default=20.0, help="alpha for exp_penalty")
    p.add_argument("--tau", type=float, default=0.01, help="tau for hinge")
    p.add_argument("--beta", type=float, default=10.0, help="beta for rbf")
    # mode B params
    p.add_argument("--D", type=int, default=1024, help="input data dimension for mode B")
    p.add_argument("--kmin", type=int, default=16)
    p.add_argument("--kmax", type=int, default=2048)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--learn_iters", type=int, default=1500)
    p.add_argument("--kstep", type=int, default=32)
    p.add_argument("--quiet", action='store_true')
    # grid for D
    p.add_argument("--grid_Ns", type=str, default="1000,5000,10000")
    p.add_argument("--grid_ks", type=str, default="128,256,512")
    # external file paths
    p.add_argument("--hadamard-path", type=str, default="/mnt/data/Hadamard random projection.py")
    p.add_argument("--jlopt-path", type=str, default="/mnt/data/JL optimizer.py")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # attempt to import uploaded files
    hadamard_mod = try_import_module_from_path(args.hadamard_path, "user_hadamard")
    jlopt_mod = try_import_module_from_path(args.jlopt_path, "user_jlopt")
    hadamard_impl = None
    if hadamard_mod:
        # try to detect expected function names
        if hasattr(hadamard_mod, "hadamard_random_projection"):
            hadamard_impl = getattr(hadamard_mod, "hadamard_random_projection")
            print("[info] using hadamard_random_projection from uploaded file")
        elif hasattr(hadamard_mod, "hadamard_project"):
            hadamard_impl = getattr(hadamard_mod, "hadamard_project")
            print("[info] using hadamard_project from uploaded file")
        else:
            print("[info] hadamard module loaded but no known function found; will fallback to internal")
            hadamard_impl = None
    else:
        print("[info] no hadamard uploaded file found; will use internal fallback")

    if jlopt_mod:
        print("[info] loaded JL optimizer uploaded file (module available)")
        # we will not call functions from it directly unless there are clear exports
    else:
        print("[info] no JL optimizer uploaded file found; using internal implementations")

    # dispatch modes
    if args.mode == 'A':
        mode_A(args, hadamard_impl=hadamard_impl, jlopt_mod=jlopt_mod)
    elif args.mode == 'B':
        mode_B(args, hadamard_impl=hadamard_impl, jlopt_mod=jlopt_mod)
    elif args.mode == 'C':
        mode_C(args, hadamard_impl=hadamard_impl, jlopt_mod=jlopt_mod)
    elif args.mode == 'D':
        mode_D(args, hadamard_impl=hadamard_impl, jlopt_mod=jlopt_mod)
    else:
        print("unknown mode")

if __name__ == "__main__":
    main()
