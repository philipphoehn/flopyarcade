# FloPyArcade Training Examples

## Ray RLlib Branch (`feat/rllib-examples`)

### New Scripts (not on master)

| Script | Algorithm | Actions | Notes |
|--------|-----------|---------|-------|
| `train_rllib_dqn.py` | DQN | discrete | Off-policy, ε-greedy, resume support |
| `train_rllib_ppo_icm.py` | PPO+ICM | discrete/cont. | Intrinsic curiosity exploration, resume support |

### Already on Master (unchanged)

| Script | Algorithm |
|--------|-----------|
| `train_rllib_apexdqn.py` | APEX-DQN |
| `train_rllib_appo.py` | APPO |
| `train_rllib_impala.py` | IMPALA |
| `train_rllib_ppo.py` | PPO |

### Removed (not available in Ray 2.56)

| Script | Reason |
|--------|--------|
| ~~`train_rllib_td3.py`~~ | TD3 never existed in Ray core (was rllib_contrib only) |
| ~~`train_rllib_cql.py`~~ | Script used fake config keys; real CQL needs custom policy |
| ~~`train_rllib_ddpg.py`~~ | DDPG removed from Ray after v2.8 (use SB3 instead) |
| ~~`train_rllib_a3c.py`~~ | A3C removed from Ray after v2.8 (use IMPALA/APPO instead) |
| ~~`train_rllib_es.py`~~ | ES removed from Ray after v2.8 (use SB3 or neuroevolution) |

### Common Module

`common_rllib.py` provides shared utilities used by all scripts:
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
| `--framework` | `tf` or `torch` |

### Quick Start

```bash
# Smoke test (200 steps, 1 eval episode)
python train_rllib_dqn.py --quick-test
python train_rllib_ppo_icm.py --quick-test

# Full training
python train_rllib_dqn.py --envtype 1s-d --timesteps 5000000 --cpus 8
python train_rllib_ppo_icm.py --envtype 6r-c --timesteps 10000000 --cpus 8

# Resume from checkpoint
python train_rllib_dqn.py --resume ./results/1s-d-DQN/checkpoint_100

# Eval only (use common_rllib.run_evaluation with a loaded model)
python -c "from common_rllib import *; ..."
```

### Requirements

```
ray[default]>=2.49
flopy>=3.9.2
gymnasium
numpy
tensorflow>=2.10  # or torch
```

### Ray 2.56 Algorithm Availability

| Available | Removed (post-v2.8) |
|-----------|---------------------|
| DQN, PPO, SAC, CQL | DDPG, TD3, A3C, ES, APEX-DQN* |
| APPO, IMPALA, MARWIL, BC | PPO (old API) |

*APEX-DQN still works via `tune.run("DQN", config={"algorithm": "APEX_DQN"})` in some versions.

For continuous-action off-policy (DDPG/SAC/TD3), use the **SB3 branch** (`feat/sb3-examples`) instead.
