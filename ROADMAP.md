# FloPyArcade — Roadmap

> Canonical prioritised backlog and development plan.
> Last updated: 2026-11-01 (rebuilt after module-split audit)
> Supersedes `ideas.dev` (dev branch) and previous ROADMAP revision.

---

## Architecture (current)

The monolithic `flopyarcade.py` has been **split into focused modules**:

| Module | Responsibility |
|--------|----------------|
| `env_base.py` | Environment setup, MODFLOW/MODPATH coupling, step loop |
| `actions.py` | `ActionsMixin` — action initialisation, mutation, application |
| `agent.py` | Agent logic, novelty search, DQN, neuroevolution, RLlib |
| `reward.py` | `RewardMixin` — reward computation, fail/boundary checks |
| `observations.py` | `ObservationsMixin` — head sampling, normalisation, field access |
| `modflow_runner.py` | MODFLOW 6 simulation wrapper |
| `modpath_runner.py` | MODPATH particle tracking wrapper |
| `arcade.py` | Core game loop orchestration |
| `render.py` | 2D matplotlib renderer |
| `render3d.py` | 3D cyberpunk renderer (mplot3d) |
| `play.py` / `play3d.py` | Interactive play + demo/animation |
| `train_dqn.py` | DQN training entry point |
| `train_neuroevolution.py` | Neuroevolution training entry point |
| `train_rllib_*.py` | RLlib (APEX-DQN, APEX, IMPALA, PPO) entry points |

---

## ✅ Already Implemented (verified in codebase)

| Original # | Feature | Status |
|------------|---------|--------|
| 1 (partial) | ConvNet architecture (3× Conv2D: 32→64→64) | ✅ `agent.py` NNTYPE='convolution' |
| 3 (partial) | Cost/money reward objective | ✅ `reward.py` `5s-c-cost` envtype |
| 3 (partial) | Don't zero reward on loss (cost mode) | ✅ `update_reward_on_fail` keeps running sum |
| 7 (partial) | DQN training pipeline | ✅ `train_dqn.py` + `agent.runDQN()` |
| 17 (partial) | FPS display | ✅ `play3d.py` calculates + prints FPS |
| 19 | Functionalize `run()` into modules | ✅ Full module split |
| 22 (partial) | Resume function (neuroevolution) | ✅ `train_neuroevolution.py --resume` |
| 36 (partial) | Save models with ID | ✅ Novelty archive in `agent.py` |
| — | RLlib trainers (APEX-DQN, APEX, IMPALA, PPO) | ✅ 4 `train_rllib_*.py` files |
| — | Mac standalone build + CI | ✅ `build-standalone.yml`, `build-mac-prototype.yml` |
| — | 3D cyberpunk renderer | ✅ `render3d.py` + `play3d.py` |
| — | Code coverage (78%) + CodeCov | ✅ `coverage.xml`, `test_coverage_gaps.py` |
| — | MODFLOW convergence backoff | ✅ `refactor-module-split` branch |

---

## 🔴 P1 — Remaining Core Work (High Impact)

| # | Task | Impact | Status |
|---|------|--------|--------|
| A1 | **Double-DQN**: target network for Q-predictions in `agent.py` | Very High | ❌ Not implemented — `get_qs()` uses main model |
| A2 | **Novelty metric fix**: length-invariant score, stop below threshold, separate novelty-children vs elites | Very High | ❌ Bugs #26/#27/#28 still open |
| A3 | **ConvNet → full field**: feed head + velocity + pressure 2D arrays (not just sampled points) into Conv2D layers | High | ⚠️ Architecture exists, input shape may be 1D |
| A4 | **Multi-particle / particle swarm** per episode | High | ❌ Not implemented |
| A5 | **K-regions** (multiple capture zones / helper wells) | High | ⚠️ Helper wells exist (ENVTYPE 4/5), but no formal K-region abstraction |
| A6 | **Avoid keep-winning**: cap max distance, random boundary noise | High | ❌ Not implemented |
| A7 | **Hyperparameter optimisation** (Bayesian / Optuna / grid) | Med-High | ❌ Manual only |
| A8 | **Reward scaling 0→1** across all env types (unify with cost mode) | Med-High | ⚠️ Partial (cost mode yes, others no) |
| A9 | **Combine multiple initial runs** → high-variability seed set | Med-High | ❌ Not implemented |
| A10 | **Prevent protection-zone overlap** for multi-well envs | Med-High | ❌ Not implemented |

---

## 🟡 P2 — Performance, UX & Visualisation

| # | Task | Impact | Status |
|---|------|--------|--------|
| B1 | **Smooth trajectory** (interpolate in-between MODPATH steps) | High UX | ❌ Not implemented |
| B2 | **Split-screen GUI**: 4 panels (random / inexperienced / experienced / agent) | High UX | ❌ Not implemented |
| B3 | **Actions bar at top**, colormap scaled by reward | Med UX | ❌ Not implemented |
| B4 | **Store & plot all well trajectories** (ensemble) | Med UX | ❌ Only last game stored |
| B5 | **Generational trajectory ensemble plot** | Med UX | ❌ Not implemented |
| B6 | **Matplotlib blitting** for 60+ FPS in `play.py` | Med Perf | ❌ Not implemented |
| B7 | **FPS counter in 2D HUD** (`play.py`) | Med Perf | ❌ Only in 3D |
| B8 | **HDF5 novelty archive** (lazy load, append-only) | Med Perf | ❌ Currently pickle/JSON |
| B9 | **Auto-profile runtimes** (cProfile / line_profiler integration) | Med Perf | ❌ Not implemented |
| B10 | **RAM-disk / in-memory filesystem** for MODFLOW work files | Med Perf | ❌ Not implemented |
| B11 | **Resume function for DQN** (checkpoint save/load) | Med UX | ❌ Only neuroevolution has it |
| B12 | **Website: auto-update benchmark figures** | Med UX | ❌ Manual |
| B13 | **Website: animated GIF logo** | Low UX | ❌ Static |
| B14 | **Scale linewidth around well** with pumping rate | Low UX | ❌ Not implemented |
| B15 | **Edit white line** (highest reward) + transparent initial | Low UX | ❌ Not implemented |
| B16 | **Generate GIFs of progress** during evolution | Low UX | ❌ Not implemented |

---

## 🟢 P3 — Research, Debugging & Polish

| # | Task | Impact | Status |
|---|------|--------|--------|
| C1 | **Surrogate model**: test sim-boost factor, recall test | Med | ❌ Research |
| C2 | **Why does novelty increase with action length?** | High | ❌ Open bug |
| C3 | **Why is best-reward agent inconsistent?** | High | ❌ Open bug |
| C4 | **Duplicate agents after restart?** | Med | ❌ Open bug |
| C5 | **Check convergence / success flag** in MODFLOW | Med | ⚠️ Backoff added, flag check TBD |
| C6 | **ENVTYPE 3 reward not fully reset on loss?** | Med | ❌ Investigate |
| C7 | **Is initialisation triggering limited action set?** | Med | ❌ Investigate |
| C8 | **Debug: flexibility when changing mesh resolution** | Med | ❌ Not tested |
| C9 | **Pruning + ensemble validation** with archived models | Med | ❌ Not implemented |
| C10 | **GA: include network architecture in search** | Low | ❌ Not implemented |
| C11 | **Test averaging behaviour** with same model | Low | ❌ Not implemented |
| C12 | **DOCS**: complete docstrings + Sphinx auto-doc | Low | ⚠️ Partial (modules have docstrings) |
| C13 | **DOCS**: GIF/video for CLI installation | Low | ❌ Not done |
| C14 | **Prognostic auxiliary particle** (longer MODPATH) | Low | ❌ Research |
| C15 | **Remove prints** from FloPy / TF logging | Low | ⚠️ Partial |
| C16 | **BinderHub / Littlest JupyterHub** | Low | ❌ Not set up |
| C17 | **Upload function for autograder** | Low | ❌ Not implemented |
| C18 | **Jupyter notebook with pre-prepared solutions** | Low | ❌ Not done |
| C19 | **Save test game dataset** for evaluation | Low | ❌ Not done |
| C20 | **Same model set for training + cross-validation** | Low | ❌ Not enforced |

---

## 🌿 Development Branch Plan (Revised)

### Dependency graph

```
Phase 1 (parallel):
  feat/double-dqn        feat/novelty-fix        feat/env-hardening
        │                      │                      │
        │                      │                      │
        ▼                      ▼                      ▼
Phase 2 (parallel):
  feat/convnet-field     feat/perf-render       feat/rl-rlib-tune
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
Phase 3:
  feat/gui-split-screen
```

---

### Phase 1 — Independent, unblock everything

#### `feat/double-dqn`
| Task | Detail |
|------|--------|
| A1 | Add target network to `agent.py` DQN path; `get_qs()` uses target for bootstrapping |
| A8 | Unify reward scaling 0→1 across all env types |
| B11 | Add DQN checkpoint save/load (resume) |

**Deliverable:** Double-DQN matches or beats vanilla DQN on ENVTYPE 1 within N episodes. Resume works.

#### `feat/novelty-fix`
| Task | Detail |
|------|--------|
| A2 | Weighted novelty score, threshold stop, separate children vs elites |
| C2 | Debug: novelty increases with action length |
| C3 | Debug: best-reward agent overwritten |
| C4 | Debug: duplicate agents after restart |
| C7 | Check if init triggers limited action set |
| C9 | Pruning + ensemble validation |

**Deliverable:** Neuroevolution stable. Novelty metric length-invariant. Best-agent tracking correct.

#### `feat/env-hardening`
| Task | Detail |
|------|--------|
| A4 | Multi-particle / particle swarm per episode |
| A5 | K-region abstraction (generalise helper wells) |
| A6 | Cap max distance, random boundary noise |
| A10 | Prevent protection-zone overlap |
| C5 | Verify MODFLOW convergence flag |
| C6 | Fix ENVTYPE 3 reward reset |
| C8 | Test mesh resolution flexibility |

**Deliverable:** New env types (multi-particle, K-region). All existing types still pass tests.

---

### Phase 2 — Parallel, build on Phase 1

#### `feat/convnet-field`
| Task | Detail |
|------|--------|
| A3 | Feed 2D head + velocity + pressure arrays into Conv2D (reshape observations) |
| A7 | Hyperparameter optimisation (Optuna) for ConvNet + DQN |
| A9 | Multiple initial runs → high-variability seed set |
| C1 | Surrogate model speedup test |

**Deliverable:** ConvNet ingesting full 2D fields. Beats 1D-MLP on ENVTYPE 2/3. HP search config.

#### `feat/perf-render`
| Task | Detail |
|------|--------|
| B1 | Smooth trajectory (interpolate MODPATH steps) |
| B6 | Matplotlib blitting in `play.py` |
| B7 | FPS counter in 2D HUD |
| B8 | HDF5 novelty archive |
| B9 | Auto-profile runtimes |
| B10 | In-memory filesystem for MODFLOW |
| B16 | Progress GIFs during evolution |

**Deliverable:** 2–5× faster play loop. FPS visible. Archive scales. 30+ FPS for 40×40.

#### `feat/rl-rlib-tune`
| Task | Detail |
|------|--------|
| A7 | Tune RLlib (APEX-DQN, IMPALA, PPO) hyperparameters |
| A9 | Seed diversity for RLlib |
| C11 | Test averaging / reproducibility |
| C20 | Enforce same model set for train + CV |
| C19 | Save test game dataset |

**Deliverable:** Benchmark table: DQN vs APEX-DQN vs IMPALA vs PPO vs Neuroevolution per env type.

---

### Phase 3 — Presentation (needs Phase 1+2 stable)

#### `feat/gui-split-screen`
| Task | Detail |
|------|--------|
| B2 | 4-panel split-screen (random / inexperienced / experienced / agent) |
| B3 | Actions bar at top, reward-colourmap |
| B4 | Store & plot all well trajectories |
| B5 | Generational ensemble plot |
| B12 | Auto-update benchmark figures |
| B13 | Animated GIF logo |
| B14 | Scale well linewidth with Q |
| B15 | Edit white/transparent reward line |
| C12 | Sphinx docs + auto-doc |
| C13 | CLI install GIF/video |
| C16 | BinderHub |
| C17 | Autograder upload |
| C18 | Jupyter solutions notebook |

**Deliverable:** `play_gui.py` — 4-panel, live FPS, action bar, trajectory history. Auto benchmarks. Binder-ready.

---

## Top 5 — Recommended Immediate Actions

| Rank | Task | Branch | Effort | Gain |
|------|------|--------|--------|------|
| 1 | Double-DQN + reward unification | `feat/double-dqn` | S | Immediate agent quality lift |
| 2 | Novelty metric fix | `feat/novelty-fix` | M | Unlocks reliable neuroevolution |
| 3 | Env hardening (multi-particle, K-region) | `feat/env-hardening` | M | Richer game, more realistic |
| 4 | ConvNet full-field input | `feat/convnet-field` | M | State-of-the-art RL path |
| 5 | Matplotlib blitting + smooth trajectory | `feat/perf-render` | S | 2–5× interactive speed |

---

## References

### Novelty / Similarity
- Weighted novelty / fitness: http://people.idsia.ch/~tino/papers/cuccu.evostar11.pdf
- Similarity measures: https://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
- Approximate novelty: https://books.google.de/books?id=wJiRDwAAQBAJ

### RL / Hyperparameters
- Hyperparameter example: https://arxiv.org/pdf/1207.6682.pdf

### Infrastructure
- Littlest JupyterHub: https://the-littlest-jupyterhub.readthedocs.io/
- BinderHub: https://binderhub.readthedocs.io/en/latest/
- RAM-disk benchmarks: https://www.raymond.cc/blog/12-ram-disk-software-benchmarked-for-fastest-read-and-write-speed/
- pyfilesystem: https://docs.pyfilesystem.org/en/latest/concepts.html

---

## Changelog

| Date | Change |
|------|--------|
| 2026-11-01 | Rebuilt: module-split audit, marked 13 items as implemented, re-prioritised remaining work, revised branch plan to 7 branches in 3 phases |
| 2026-11-01 | Added 6-branch development plan + dependency graph |
| 2026-11-01 | Initial: 47 items from `ideas.dev` |
