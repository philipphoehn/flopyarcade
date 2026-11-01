# FloPyArcade — Roadmap

> Canonical prioritised backlog. Supersedes `ideas.dev` (dev branch).
> Last updated: 2026-11-01

---

## P1 — Core Gameplay & Agent Quality

| # | Idea | Impact | Notes |
|---|------|--------|-------|
| 1 | Couple ConvNet to **full flow-field observations** (all stresses, not just head) | Very High | Current agents see a tiny slice; full-field conv could match Atari-grade agents to groundwater |
| 2 | **Novelty search overhaul**: weighted novelty score, stop below threshold, separate novelty-children vs novelty-elites | Very High | Current novelty metric is noisy (increases with action length); fixing it stabilises evolution |
| 3 | **Reward redesign**: scale 0→1, don't zero all rewards on loss, add cost/money objective | High | Reward shape drives everything; current reset-to-0 loses signal |
| 4 | **Multiple particles / particle swarm** per episode | High | More realistic, harder problem, richer training signal |
| 5 | **K-regions** (multiple capture zones) | High | Turns single-well into multi-objective game |
| 6 | **Avoid keep-winning**: cap max distance from well Y, add random boundary noise | High | Prevents degenerate trivial solutions |
| 7 | **Target model for Q-predictions** (Double-DQN fix) | High | Current `get_qs()` uses main model → overestimation bias |
| 8 | **Hyperparameter optimisation** (Bayesian / grid) + discount rationale | Med-High | Manual tuning is the bottleneck for reproducibility |
| 9 | **Combine multiple initial runs** → high-variability seed set | Med-High | Better population diversity from gen 0 |
| 10 | **Avoid protection-zone overlap** for multi-well envs | Med-High | Edge-case correctness in realistic scenarios |

---

## P2 — Performance, UX & Visualisation

| # | Idea | Impact | Notes |
|---|------|--------|-------|
| 11 | **Smooth trajectory** (interpolate in-between particle-tracking steps) | High UX | Jittery animation → fluid motion |
| 12 | **Split-screen GUI**: 4 panels (random / inexperienced / experienced / agent) | High UX | Instant comparison, great for demos & papers |
| 13 | **Actions bar at top**, colormap scaled by reward | Med UX | Makes policy behaviour legible at a glance |
| 14 | **Store & plot all well trajectories** (not just last game) | Med UX | Enables ensemble visualisation |
| 15 | **Generational trajectory ensemble plot** | Med UX | Shows evolution converging over generations |
| 16 | **Matplotlib blitting** for 60+ FPS display | Med Perf | Eliminates full redraw per frame |
| 17 | **Calculate / display FPS** in HUD | Med Perf | User sees performance live |
| 18 | **HDF5 novelty archive** (lazy load, append-only) | Med Perf | Avoids full reload as archive grows |
| 19 | **Functionalize `run()`** + convert models to JSON | Med Perf | Cleaner API, faster serialisation |
| 20 | **Auto-profile runtimes** | Med Perf | Identify bottlenecks automatically |
| 21 | **RAM-disk / in-memory filesystem** for MODFLOW work files | Med Perf | Avoids disk I/O on every step |
| 22 | **Resume function** (DQN + neuroevolution) | Med UX | Long training runs crash → lose progress |
| 23 | **Website: auto-update benchmark figures** from saved policy models | Med UX | Keeps README / docs current without manual regen |
| 24 | **Website: animated GIF logo** (particle moving on load) | Low UX | Brand polish |

---

## P3 — Research, Debugging & Polish

| # | Idea | Impact | Notes |
|---|------|--------|-------|
| 25 | **Surrogate model**: test sim-boost factor, genetic HP sampling, recall test | Med | Research — could 10× training speed |
| 26 | **Why does novelty increase with action length?** (investigate) | High | Debugging — may reveal metric bug |
| 27 | **Why is best-reward agent inconsistent?** (overwritten by novelty child?) | High | Bug-hunt |
| 28 | **Duplicate agents after restart?** | Med | Bug-hunt |
| 29 | **Check convergence / success flag** in MODFLOW | Med | Correctness |
| 30 | **ENVTYPE 3 reward not fully reset on loss?** | Med | Bug-hunt |
| 31 | **Is initialisation triggering limited action set?** (shuffle within layer) | Med | Agent quality |
| 32 | **Debug: flexibility when changing mesh resolution** | Med | Robustness |
| 33 | **Pruning + ensemble validation** with archived models | Med | Agent quality |
| 34 | **Genetic algorithm**: HP search, syntax consistency, include arch in search | Low | Polish |
| 35 | **Test averaging behaviour** with same model (reproducibility) | Low | QA |
| 36 | **Save all models with ID** for visuals | Low | Dev convenience |
| 37 | **Scale linewidth around well** with pumping rate magnitude | Low UX | Visual cue |
| 38 | **Edit white line** (highest reward) + keep initial line transparent | Low UX | Visual polish |
| 39 | **Generate GIFs of progress** during evolution | Low UX | Progress visibility |
| 40 | **DOCS**: GIF/video for CLI installation | Low | Onboarding |
| 41 | **DOCS**: complete docstrings + auto-doc | Low | Maintainability |
| 42 | **Prognostic auxiliary particle** (longer MODPATH run) | Low | Research |
| 43 | **Remove prints** from FloPy / TensorFlow (workspace cleanup) | Low | Cleanliness |
| 44 | **BinderHub / Littlest JupyterHub** for reproducible demos | Low | Infra |
| 45 | **Upload function** for an autograder | Low | Education / competition |
| 46 | **Jupyter notebook** with pre-prepared solutions | Low | Onboarding |
| 47 | **Save test game dataset** / check prediction dataset | Low | QA |

---

## Top 5 — Recommended Next Actions

| Rank | Idea(s) | Effort | Expected gain |
|------|---------|--------|---------------|
| 1 | Novelty metric fix (#2, #26, #27) | M | Stabilises evolution, unlocks reliable neuroevolution |
| 2 | Reward redesign (#3) | S | Immediate agent-quality improvement across all algorithms |
| 3 | Full-field ConvNet (#1) | L | Opens path to state-of-the-art RL agent |
| 4 | Smooth trajectory + split-screen (#11, #12) | S | Dramatic UX improvement for demos & paper figures |
| 5 | Matplotlib blitting + FPS (#16, #17) | S | 2–5× interactive speed, professional feel |

---

## Environment & Simulator Notes

- Improved discretisation
- Retrieved head in circles looking ahead a few metres, not directly at well
- Runtime offset genetic current to all in print
- Check how actions differ with given `mutationPower`
- Add fully independent test dataset (i.e. seeds)
- Is it not better to give reward and not reset all rewards to 0?

## Infrastructure

- BinderHub / Littlest JupyterHub for reproducible demos
  - The Littlest JupyterHub: https://the-littlest-jupyterhub.readthedocs.io/
  - BinderHub: https://binderhub.readthedocs.io/en/latest/
- RAM-disk options: `pyfilesystem` in-memory, Windows RAMDISK, OSForensics
- Speed benchmarks: https://www.raymond.cc/blog/12-ram-disk-software-benchmarked-for-fastest-read-and-write-speed/

## Novelty / Similarity References

- Weighted novelty / fitness: http://people.idsia.ch/~tino/papers/cuccu.evostar11.pdf
- Popular similarity measures: https://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
- Approximate novelty & replace: https://books.google.de/books?id=wJiRDwAAQBAJ

## DQN / RL Specific

- Scale rewards between 0 and 1?
- ReplayMemory stored that is then only updated during run?
- Able to reproduce memory increase that slows everything down?
- Add resume function
- GS: include network architecture in genetic search?
- NOVELTY: example of hyperparameters: https://arxiv.org/pdf/1207.6682.pdf

## GUI / Play

- Start button
- Split screen with 4 motions: random, inexperienced, experienced, agent
- Actions at top progress bar, colormap scaled with reward
- Calculate prognostic/auxiliary particle trajectory with longer MODPATH run

## Misc

- Make sure to use same set of models for training all the time and for cross-validation
- Hyperparameters for genetic algorithm
- Syntax consistency genetic algorithm
- Remove prints from FloPy and TensorFlow (changing model workspace → flopy → mbase; using tensorflow → keras → init)
