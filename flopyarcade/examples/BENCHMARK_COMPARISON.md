# Neuroevolution Benchmark Comparison — FloPyArcade

> **A11: External neuroevolution examples**
>
> This document compares three external neuroevolution approaches for
> evolving FloPyArcade neural-network policy weights. All scripts are
> **code-only**; run them on GPU resources for full training.

## Methods

| Method | Package | Backend | Key Feature |
|--------|---------|---------|-------------|
| **CMA-ES** | `cma` | NumPy + Keras | Covariance Matrix Adaptation; sample-efficient, no gradient |
| **RLlib ES** | `ray[rllib]` | TF / PyTorch | Distributed ES via Ray; parallel rollout workers |
| **EvoTorch** | `evotorch` | PyTorch | Modern ES (CMA, SAES) with adaptive noise; PyTorch-native |

## Quick-Start

```bash
# 1. CMA-ES (lightweight, single-process)
pip install cma
python -m flopyarcade.examples.train_cmaes \
    --envtype 3s-d --nrow 50 --ncol 50 \
    --popsize 60 --maxiter 5 --seed 1

# 2. Ray RLlib ES (distributed)
pip install ray[rllib]
python -m flopyarcade.examples.train_rllib_es \
    --envtype 3s-d --nrow 50 --ncol 50 \
    --popsize 64 --maxiters 3 --num-threads 4

# 3. EvoTorch / manual ES (PyTorch)
pip install evotorch torch
python -m flopyarcade.examples.train_evotorch \
    --envtype 3s-d --nrow 50 --ncol 50 \
    --popsize 64 --maxiter 5 --seed 1
```

## Comparison Criteria

| Criterion | CMA-ES | RLlib ES | EvoTorch |
|-----------|--------|----------|----------|
| Ease of setup | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| GPU acceleration | ❌ | ✅ | ✅ |
| Distributed rollouts | ❌ | ✅ (Ray) | ❌ |
| Adaptive step-size | ✅ (CMA) | ✅ (GaussianExplorer) | ✅ (SAES) |
| Multi-objective | ❌ | ❌ | Partial |
| Minimal deps | `cma` only | `ray` + `rllib` | `torch` (+`evotorch`) |
| Best for | Small nets, quick prototyping | Large-scale, multi-worker | PyTorch ecosystems, SAES |

## Notes

- **CMA-ES** is the best starting point for small networks (< 10k params) and
  single-GPU experiments. It requires no gradient and converges quickly.
- **RLlib ES** shines when you need parallel rollout workers across multiple
  CPUs/GPUs. The `ES` trainer disables SGD; only the stochastic policy
  exploration drives weight updates.
- **EvoTorch** provides a PyTorch-native interface with SAES (Self-Adaptive
  ES). The fallback manual ES loop in `train_evotorch.py` works without the
  `evotorch` package if only `torch` is installed.

## Running Full Benchmarks

For a fair comparison, use identical environment settings:

```bash
ENV="3s-d"; NROW=50; NCOL=50; SEED=42; POP=64; ITERS=50

python -m flopyarcade.examples.train_cmaes \
    --envtype $ENV --nrow $NROW --ncol $NCOL \
    --popsize $POP --maxiter $ITERS --seedagent $SEED \
    --output runs/bench_cmaes

python -m flopyarcade.examples.train_rllib_es \
    --envtype $ENV --nrow $NROW --ncol $NCOL \
    --popsize $POP --maxiters $ITERS --num-threads 4 \
    --output runs/bench_rllib

python -m flopyarcade.examples.train_evotorch \
    --envtype $ENV --nrow $NROW --ncol $NCOL \
    --popsize $POP --maxiter $ITERS --seedagent $SEED \
    --output runs/bench_evotorch
```

Collect `results.json` from each `runs/bench_*` directory and compare
`best_reward`.

## References

- CMA-ES: [cma package](https://github.com/hansen/cma), Hansen 2016
- Ray RLlib ES: [RLlib docs](https://docs.ray.io/en/latest/rllib-algorithms-es.html)
- EvoTorch: [GitHub](https://github.com/EvoTorch/EvoTorch), Glasmaster et al. 2023

---
*Generated for FloPyArcade A11 roadmap item.*
