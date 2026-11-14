# FloPyArcade — Roadmap

> Canonical prioritised backlog and development plan.
> Last updated: 2026-11-14 (PR batch merged, test suite stabilised, MF6+PRT reverted to experimental)
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
| A1 | **Double-DQN**: target network for Q-predictions in `agent.py` | Very High | 🔄 PR [#40](https://github.com/philipphoehn/flopyarcade/pull/40) open |
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
| B6 | **Matplotlib blitting** for 60+ FPS in `play.py` | Med Perf | 🔄 PR [#42](https://github.com/philipphoehn/flopyarcade/pull/42) open |
| B7 | **FPS counter in 2D HUD** (`play.py`) | Med Perf | 🔄 PR [#43](https://github.com/philipphoehn/flopyarcade/pull/43) open |
| B8 | **HDF5 novelty archive** (lazy load, append-only) | Med Perf | 🔄 PR [#44](https://github.com/philipphoehn/flopyarcade/pull/44) open |
| B9 | **Auto-profile runtimes** (cProfile / line_profiler integration) | Med Perf | ❌ Not implemented |
| B10 | **RAM-disk / in-memory filesystem** for MODFLOW work files | Med Perf | ❌ Not implemented |
| B11 | **Resume function for DQN** (checkpoint save/load) | Med UX | 🔄 PR [#41](https://github.com/philipphoehn/flopyarcade/pull/41) open |
| B12 | **Website: auto-update benchmark figures** | Med UX | ❌ Manual |
| B13 | **Website: animated GIF logo** | Low UX | ✅ PR #53 merged (2026-11-14) |
| B14 | **Scale linewidth around well** with pumping rate | Low UX | ❌ Not implemented |
| B15 | **Edit white line** (highest reward) + transparent initial | Low UX | ✅ PR #48 merged (2026-11-14) |
| B16 | **Generate GIFs of progress** during evolution | Low UX | ❌ Not implemented |
| B17 | **Benchmark suite orchestrator** (`examples/benchmark_suite.py`): runs random / human / trained-agent on each env type, outputs reproducible score table. **Requires locally stored human control data (operator session recordings) to be uploaded to the benchmark environment to reproduce the human baseline score.** Without the uploaded control data the human column is unavailable and the table is incomplete. | High (validation) | ❌ Not implemented |

---

## 🟢 P3 — Research, Debugging & Polish

| # | Task | Impact | Status |
|---|------|--------|--------|
| C1 | **Surrogate model**: test sim-boost factor, recall test | Med | ❌ Research |
| C2 | **Novelty length-invariance fix**: normalize per-pair score by episode count | High | 🔄 PR [#45](https://github.com/philipphoehn/flopyarcade/pull/45) open |
| C3 | **Why is best-reward agent inconsistent?** | High | ❌ Open bug |
| C4 | **Duplicate agents after restart?** | Med | ❌ Open bug |
| C5 | **Check convergence / success flag** in MODFLOW | Med | ⚠️ Backoff added, flag check TBD |
| C6 | **Reward reset on fail**: reset rewardCurrent to prevent accumulation leak | Med | 🔄 PR [#46](https://github.com/philipphoehn/flopyarcade/pull/46) open |
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

## 🔬 P0 — 50-Idea Expansion (2026-11-14)

> Curated from 500 candidates. All items below are **new** (not tracked above).
> Distribution: Research 12 · Benchmarking 10 · Reporting 10 · Visualisation 8 · Tooling 4 · Testing 3 · Science 2 · Community 1

### Research & RL Methods (12)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| R1 | **Surrogate-assisted Bayesian optimisation** (GP surrogate on 20 cheap runs → 5 expensive full runs) for HP search per env type | Very High | ❌ Not implemented |
| R2 | **Curriculum learning**: 1s-d → 2s-d → 3s-d → cost modes; auto-promote when reward > threshold | Very High | ❌ Not implemented |
| R3 | **Physics-informed loss**: penalty <latex>\lambda \| \nabla \cdot (K\nabla h) - S_s \partial_t h \|^2</latex> in reward; policy respects governing PDE | High | ❌ Not implemented |
| R4 | **Multi-objective NSGA-II** on (reward, cost, wall-clock); archive Pareto front to HDF5 | High | ❌ Not implemented |
| R5 | **Differentiable MODFLOW surrogate**: CNN predicts <latex>h_{t+1}</latex> from <latex>h_t, q_{wells}</latex>; gradient for MPC step | Very High | ❌ Not implemented |
| R6 | **Equivariant network on grid**: E(2)-equivariant layer; mesh-invariant policy generalises across env sizes | High | ❌ Not implemented |
| R7 | **Reward hacking detector**: track <latex>\nabla_\theta \mathbb{E}[r]</latex> vs measured reward; alert on divergence | Med-High | ❌ Not implemented |
| R8 | **Constrained MDP / safe RL**: hard constraint <latex>h_{capture} \geq h_{safety}</latex> via Lagrangian dual; report violation rate | High | ❌ Not implemented |
| R9 | **Neuroevolution island model**: 4 islands, migrate top-5 every 50 gens; diversity metric | Med-High | ❌ Not implemented |
| R10 | **Meta-learning (MAML) on reward weights**: outer loop adapts per env type in < 10 episodes | Med-High | ❌ Not implemented |
| R11 | **Uncertainty quantification**: MC-dropout ensemble of 5 DQN heads → mean±std Q; use for exploration bonus | Med-High | ❌ Not implemented |
| R12 | **Graph neural network on MODFLOW mesh**: nodes=cells, edges=connectivity; GNN predicts heads; agent reads features | High | ❌ Not implemented |

### Benchmarking & Evaluation (10)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| B18 | **Bootstrap CI on benchmark scores**: 1000 resamples → 95% CI + p-value vs baseline | Very High | ❌ Not implemented |
| B19 | **Seed-sensitivity table**: 10 seeds × 6 envs × 3 algorithms → std, IQR, CV per cell | High | ❌ Not implemented |
| B20 | **Ablation matrix**: remove one reward component × 6 envs → Δscore table | High | ❌ Not implemented |
| B21 | **Cross-env transfer heatmap**: train X, eval Y → 6×6 matrix + t-test per pair | High | ❌ Not implemented |
| B22 | **Cost-reward Pareto front**: scatter (cost, reward) per algorithm × env; 95% CI envelope | Med-High | ❌ Not implemented |
| B23 | **Wall-clock vs sim-steps dual axis**: ms/step + steps/s; flag >10% drift per commit | Med-High | ❌ Not implemented |
| B24 | **CI benchmark gate**: fail if any env score drops >5% from main artifact (JSON in Actions cache) | High | ❌ Not implemented |
| B25 | **Agent-vs-human paired test**: same 10 seeds, paired t-test + Cohen's d | High | ❌ Not implemented |
| B26 | **Longitudinal benchmark archive**: append-only JSONL (`benchmarks/history.jsonl`); every CI run appends | Med-High | ❌ Not implemented |
| B27 | **Sample efficiency metric**: episodes-to-80%-of-best per algorithm × env | Med-High | ❌ Not implemented |

### Reporting & Data Export (10)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| P1 | **Per-step trajectory JSON**: `[{step, action, reward, obs_sha, pos_2d, head_mean}]` in HDF5 | Very High | ❌ Not implemented |
| P2 | **MODFLOW convergence trace**: parse lst → iterations, condition #, residual per step | High | ❌ Not implemented |
| P3 | **Seed manifest**: `{seed, git_sha, config_hash, env_version, score, ci_run_id}` JSON per run | High | ❌ Not implemented |
| P4 | **Particle trajectory export**: MODPATH paths as `.npz` (T × P × 3) per episode | Med-High | ❌ Not implemented |
| P5 | **Head/velocity field snapshots**: grid arrays every N steps → `.npz` | Med-High | ❌ Not implemented |
| P6 | **Failure taxonomy auto-classifier**: label `boundary_hit | convergence_fail | no_progress | NaN | timeout` | High | ❌ Not implemented |
| P7 | **Reward decomposition stacked bar**: per-component per episode in HDF5 | Med-High | ❌ Not implemented |
| P8 | **Checkpoint comparison tool**: side-by-side policy table + KL divergence between two Q-networks | Med-High | ❌ Not implemented |
| P9 | **Observation coverage heatmap**: fraction of action space visited per generation | Med-High | ❌ Not implemented |
| P10 | **Training curve 3-panel JSON**: loss, reward, ε → TensorBoard-compatible single file | Med-High | ❌ Not implemented |

### Visualisation & UX (8)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| V1 | **Plume front tracking plot**: distance-from-source over time; box-plot across episodes | High | ❌ Not implemented |
| V2 | **Hydraulic head change heatmap**: Δh per well, annotated with flow arrows | Med-High | ❌ Not implemented |
| V3 | **Interactive Plotly dashboard**: HDF5 → 3D scatter + time slider + well selector + head contour | High | ❌ Not implemented |
| V4 | **Video export (mp4)**: 3D trajectory at 30 fps via matplotlib.animation → ffmpeg | Med-High | ❌ Not implemented |
| V5 | **Ablation tornado plot**: rank HPs by \|Δscore\| across ±20% perturbation | Med-High | ❌ Not implemented |
| V6 | **Cross-env transfer 6×6 heatmap**: colour = score; p-value significance stars | Med-High | ❌ Not implemented |
| V7 | **Well-activation timeline Gantt**: per-step pumping rate per well; stacked bar | Med | ❌ Not implemented |
| V8 | **Benchmark drift line chart**: score vs date; 95% CI band; alert marker on >5% drop | Med-High | ❌ Not implemented |

### Tooling & Workflows (4)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| T1 | **`flopyarcade bench` CLI**: `python -m flopyarcade bench --env 3s-d --seed 42 --episodes 50` | High | ❌ Not implemented |
| T2 | **Hydra config system**: YAML hierarchies for env/agent/reward/trainer; CLI overrides | Med-High | ❌ Not implemented |
| T3 | **MLflow / TensorBoard integration**: log metrics, params, artifacts; `mlflow ui` | Med-High | ❌ Not implemented |
| T4 | **ONNX export + inference bench**: DQN → ONNX; ms/inference on CPU & GPU vs PyTorch | Med | ❌ Not implemented |

### Testing & Quality (3)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| Q1 | **Property-based testing (Hypothesis)**: random valid configs → assert obs space, reward bounds, step count | High | ❌ Not implemented |
| Q2 | **Golden-file regression**: reference HDF5 for 3 episodes; CI diffs < 0.1% tolerance | High | ❌ Not implemented |
| Q3 | **Performance regression gate**: assert ms/step < 2× baseline; fail CI on >50% slowdown | Med-High | ❌ Not implemented |

### Science & Domain (2)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| S1 | **Multi-component reactive transport env**: 2-species degradation (parent→daughter), first-order kinetics; env `7r-d` | High | ❌ Not implemented |
| S2 | **Heterogeneous aquifer env**: layered K + lens geometry; env `8s-d`; capture-zone efficiency vs homogeneous | Med-High | ❌ Not implemented |

### Community & Adoption (1)

| # | Idea | Impact | Status |
|---|------|--------|--------|
| C21 | **Zenodo DOI + reproducible benchmark package**: archive results + config + seeds → DOI; cite in paper | Med-High | ❌ Not implemented |

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
| 2026-11-14 | Merged 9 PRs (dep bumps: imageio/tf/matplotlib/numpy/tqdm; B13 animated banner; B15 reward line; README fix). Reverted MF6+PRT (PR #60) from master → experimental on `feat/mf6-prt-bmi`. Fixed test suite: skipped reverted BMI test, robust num_workers assertion. All 1538 tests pass (2-pass CI style). |
| 2026-11-03 | Layer separation: moved 6 train scripts to `examples/`, split deps into `[solvers]`/`[ray]` extras, made `FloPyAgent` import optional, bumped to 0.7.0, added B17 benchmark suite (requires human control data) |
| 2026-11-01 | Rebuilt: module-split audit, marked 13 items as implemented, re-prioritised remaining work, revised branch plan to 7 branches in 3 phases |
| 2026-11-01 | Added 6-branch development plan + dependency graph |
| 2026-11-01 | Initial: 47 items from `ideas.dev` |
