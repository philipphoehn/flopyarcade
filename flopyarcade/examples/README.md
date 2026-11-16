# FloPyArcade Training Examples

## Stable-Baselines3 Branch (`feat/sb3-examples`)

### Scripts

| Script | Algorithm | Actions | Notes |
|--------|-----------|---------|-------|
| `train_sb3_sac.py` | SAC | continuous | Entropy-regularized, multi-critic, N action noise |
| `train_sb3_ddpg.py` | DDPG | continuous | Off-policy, deterministic policy + noise |
| `train_sb3_ppo.py` | PPO | discrete/cont. | On-policy, clip-based objective |

### Common Module

`common.py` provides shared utilities used by all scripts:
- `add_common_args()` — standard argparse flags
- `make_flopy_env()` — environment factory
- `run_evaluation()` — standard eval loop (gymnasium 5-tuple)
- `get_outdir()` — resolve output path
- `parse_net_arch()` — parse "512,256" → [512, 256]
- `setup_logging()` — verbosity control

### Algorithm-Specific Flags

| Flag | SAC | DDPG | PPO |
|------|-----|------|-----|
| `--tau` | ✅ | ✅ | — |
| `--n-critics` | ✅ | — | — |
| `--train-freq` | ✅ | — | — |
| `--noise-sigma` | ✅ | ✅ | — |
| `--buffer-size` | ✅ | ✅ | — |
| `--learning-starts` | — | ✅ | — |
| `--n-epochs` | — | — | ✅ |
| `--clip-range` | — | — | ✅ |
| `--ent-coef` | — | — | ✅ |

### Quick Start

```bash
# Smoke test
python train_sb3_sac.py --quick-test

# Full training with custom net
python train_sb3_sac.py --envtype 3s-c --timesteps 5000000 --net-arch "512,256" --tau 0.01

# Resume
python train_sb3_sac.py --resume ./results/3s-c-SAC/sac_3s-c_1000000_steps

# Eval only
python train_sb3_sac.py --eval-only ./results/3s-c-SAC/sac_final --eval-episodes 20
```

### Requirements

```
stable-baselines3>=2.0
flopy>=3.9.2
gymnasium
numpy
```
