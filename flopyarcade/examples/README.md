# FloPyArcade Training Examples

## Ray RLlib Branch (`feat/rllib-examples`)

### New Scripts (not on master)

| Script | Algorithm | Actions | Notes |
|--------|-----------|---------|-------|
| `train_rllib_ddpg.py` | DDPG | continuous | Off-policy, Gaussian noise exploration |
| `train_rllib_a3c.py` | A3C | discrete | On-policy, multi-worker async |
| `train_rllib_dqn.py` | DQN | discrete | Off-policy, ε-greedy |
| `train_rllib_es.py` | ES | discrete/cont. | No critic, population-based |
| `train_rllib_ppo_icm.py` | PPO+ICM | discrete/cont. | Intrinsic curiosity exploration |

### Already on Master (unchanged)

| Script | Algorithm |
|--------|-----------|
| `train_rllib_apexdqn.py` | APEX-DQN |
| `train_rllib_appo.py` | APPO |
| `train_rllib_impala.py` | IMPALA |
| `train_rllib_ppo.py` | PPO |

### Removed (broken/obsolete)

| Script | Reason |
|--------|--------|
| ~~`train_rllib_td3.py`~~ | TD3 removed from Ray RLlib core after v2.8; no longer exists |
| ~~`train_rllib_cql.py`~~ | CQL is not a native Ray algorithm; script used fake config keys |

### Common Module

`common.py` provides shared utilities used by all scripts:
- `add_common_args()` — standard argparse flags
- `make_flopy_env()` — environment factory
- `run_evaluation()` — standard eval loop (gymnasium 5-tuple)
- `get_outdir()` — resolve output path
- `parse_net_arch()` — parse "512,256" → [512, 256]
- `setup_logging()` — verbosity control

### Common CLI Flags

All scripts share these flags (see `--help` for full list):

| Flag | Purpose |
|------|---------|
| `--envtype` | Environment config (e.g. `1s-d`, `3s-c`, `6r-c`) |
| `--timesteps` | Total training timesteps |
| `--lr` | Learning rate |
| `--gamma` | Discount factor |
| `--net-arch` | Hidden layer sizes, e.g. `"512,256"` |
| `--cpus` / `--gpus` | Parallelism |
| `--seed` | Algorithm seed |
| `--seed-env` | Environment seed (optional, defaults to `--seed`) |
| `--resume PATH` | Continue from checkpoint |
| `--eval-only PATH` | Evaluate without training |
| `--quick-test` | 200-step smoke test |
| `--render` | Enable visualization |
| `--outdir` | Output directory |
| `--reward-threshold` | Stop on mean reward ≥ N |
| `--eval-episodes` | Number of eval episodes |
| `--verbose` | 0-3 logging level |

### Quick Start

```bash
# Smoke test (200 steps, 1 eval episode)
python train_rllib_ddpg.py --quick-test

# Full training
python train_rllib_ddpg.py --envtype 3s-c --timesteps 10000000 --cpus 8

# Resume from checkpoint
python train_rllib_ddpg.py --resume ./results/3s-c-DDPG/checkpoint_100

# Eval only
python train_rllib_ddpg.py --eval-only ./results/3s-c-DDPG/checkpoint_200 --eval-episodes 10
```

### Requirements

```
ray[default]>=2.9
flopy>=3.9.2
gymnasium
numpy
tensorflow>=2.10  # or torch
```
