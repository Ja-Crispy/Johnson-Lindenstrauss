# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements and experiments with the Johnson-Lindenstrauss (JL) lemma, exploring dimension reduction via Gaussian random projections, Hadamard-based projections (FJLT), and learned linear projections. Based on Nick Yoder's work (nickyoder.com/johnson-lindenstrauss/).

## Directory Structure

```
cuda/               # PyTorch/CUDA implementations (cleaned up)
  experiments.py    # Main 4-mode framework (A/B/C/D)
  optimizer.py      # Fixed optimizer (while step_now < num_steps)
  hadamard.py       # FJLT with Sylvester Hadamard + sparse matrices
  comparison.py     # Side-by-side original vs fixed validation

mlx_jl/             # MLX (Apple Silicon) ports (named mlx_jl to avoid shadowing mlx package)
  experiments.py    # Same 4 modes, uses nn.Module + nn.value_and_grad
  optimizer.py      # Fixed optimizer, VectorParams(nn.Module) wrapper
  hadamard.py       # FJLT with numpy construction + MLX analysis

archive/            # Preserved originals for reference
  jl_optimizer_original.py   # Buggy: while vector_len < num_vectors
  jl_optimizer_fixed.py      # Fixed: while step_now < num_steps
  JL optimizer.py            # Nick Yoder's original from GitHub

docs/
  analysis.md               # Mathematical writeup
  blog_style_guide.md
  chatgpt_conversation.md   # Original ChatGPT session that created the experiments
  img/                       # Result plots
```

Root still has the original flat files (`jl_experiments.py`, `jl_full_comparison.py`, `Hadamard random projection.py`) for backward compatibility.

## Running Experiments

### CUDA (PyTorch)
```bash
python cuda/experiments.py --mode A --N 5000 --k 200 --loss exp_penalty
python cuda/experiments.py --mode B --D 1024 --eps 0.1 --kmin 16 --kmax 2048
python cuda/experiments.py --mode C --losses "exp_penalty,hinge,gram,rbf,logbar"
python cuda/experiments.py --mode D --grid_Ns "1000,5000,10000" --grid_ks "128,256,512"
python cuda/comparison.py           # original vs fixed side-by-side
python cuda/comparison.py --parallel  # with multiprocessing
```

### MLX (Apple Silicon)
```bash
python mlx_jl/experiments.py --mode A --N 5000 --k 200 --loss exp_penalty
python mlx_jl/experiments.py --mode B --D 1024 --eps 0.1 --kmin 16 --kmax 2048
python mlx_jl/experiments.py --mode C --losses "exp_penalty,hinge,gram,rbf,logbar"
python mlx_jl/experiments.py --mode D --grid_Ns "1000,5000,10000" --grid_ks "128,256,512"
python mlx_jl/optimizer.py          # standalone grid optimizer
```

## Dependencies

**CUDA**: `pip install torch numpy scipy tqdm matplotlib` (optional: `seaborn`)

**MLX**: `pip install mlx numpy scipy tqdm matplotlib`

## Architecture

### Experiment Modes
- **A** (reproduce): Optimize N unit vectors in k dims with different losses, observe pathologies
- **B** (compare): Find smallest k meeting distortion epsilon for Gaussian/Hadamard/learned projections, compute empirical C
- **C** (losses): Battery test of all 5 loss functions
- **D** (scaling): Grid over (N,k) to collect angle statistics

### Loss Functions
- `exp_penalty`: exp(alpha * dot^2) - Yoder's fix for gradient traps
- `hinge`: max(0, |dot| - tau) margin-based
- `gram`: ||XX^T - I||_F^2 Frobenius norm
- `rbf`: exp(beta * (dot - 1)) repulsive potential
- `logbar`: -log(1 - |dot|) log barrier

### Key Bug Fix
Original `JL optimizer.py` had `while vector_len < num_vectors` (skipped k >= N configs). Fixed to `while step_now < num_steps`. Also removed `int()` truncation on loss exponent for smooth gradients.

### MLX Porting Notes
- Optimizable arrays wrapped in `nn.Module` (VectorModel, ProjectionModel)
- Gradients via `nn.value_and_grad(model, loss_fn)` instead of `.backward()`
- FWHT uses numpy internally (MLX arrays are immutable during grad tracing)
- Pair sampling uses numpy for rejection loop, converts to `mx.array`
- `mx.eval()` calls required after optimizer updates for lazy evaluation

## Known Issues
- OOM on large configs (N=30000) with full Gram matrix (O(N^2) memory)
- T4 (15GB) handles up to ~N=20000; L4 (22GB) crashed at N=30000 in sequential mode
- MLX unified memory should handle larger N but slower than dedicated VRAM
