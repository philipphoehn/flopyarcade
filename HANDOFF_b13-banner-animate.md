# B13 Banner — Agent Build & Handoff

> Self-contained status + next steps for a following agent building/continuing `feat/b13-banner-animate`. All facts were verified read-only against the live repo.

## 1. Identity

| Item | Value |
|------|-------|
| Branch | `feat/b13-banner-animate` |
| Tip | `01cf3e3` (local == remote, in sync) |
| Fork base (merge-base with master) | `e3bf092` (Merge PR #49 layer-separation) |
| Current master | `e83dce8` — **ahead of this branch's base** |
| PR | #53 → base `master`, state **open** |
| ROADMAP item | **B13** (P2 — UX / branding) |
| Scope | **Asset-only** (one regenerated GIF). No code or test changes. |

## 2. Commit chain (newest → oldest)

| SHA | Subject | Note |
|-----|---------|------|
| `01cf3e3` | B13 — game-clipped FloPy letters + red particle | **current deliverable (v3)** |
| `9348566` | feat(ci): TestPyPI + PyPI publish workflow | shared CI commit (also on other branches) |
| `6bcac34` | B13 — animated banner (v1, teal dot) | **superseded** by `01cf3e3` |
| `e3bf092` | Merge PR #49 | fork base |

## 3. What changed (the only functional diff)

| File | Before (v1 `6bcac34`) | After (v3 `01cf3e3`) |
|------|----------------------|---------------------|
| `flopyarcade/examples/banner_animated.gif` | 1,843,834 bytes (teal dot, full-height sine) | **825,124 bytes** (red dot, game-clipped letters) |

## 4. What the v3 banner does

- The **FloPy letter bodies** (F l o P y) are filled with a clipped **head-field game render** (Blues `contourf` + concentric contours + a well marker placed at the banner's existing red ring) — the letters act as a *window into the game*.
- Letter **counters (o / P) are filled** (`binary_fill_holes`) so the game reads across whole glyph bodies, not just thin strokes.
- The banner's **existing red particle** (core + fading red trail) animates **left→right→left (ping-pong, seamless loop)** *through* the letter bodies, clipped to the glyphs, with a soft red glow bridging the counters.
- **"Arcade" + tagline are untouched** (effect limited to the top ~62% band).
- Specs: **1400×315 px, 48 frames, 70 ms/frame, infinite loop**.

## 5. How to verify (no GPU needed)

1. `git log --oneline -3 feat/b13-banner-animate` → tip must be `01cf3e3`.
2. `git cat-file -s 01cf3e3:flopyarcade/examples/banner_animated.gif` → **825124**.
3. `git diff master...feat/b13-banner-animate --stat` → single file `banner_animated.gif`.
4. Open `flopyarcade/examples/banner_animated.gif` in a browser → red dot moves *inside* the letters, game render visible behind them, lower text intact.
5. `gh pr view 53 --json state,headRefOid,baseRefName` → open, head `01cf3e3`, base `master`.

## 6. How to regenerate the GIF (if it must be rebuilt)

Input: `flopyarcade/examples/banner.png` (14000×3152 source). Pipeline (Pillow + scipy + matplotlib `Agg`):

1. Downscale banner to width **1400**.
2. Glyph mask = bright pixels in the **top ~62%** band → `binary_fill_holes` (fills o/P counters).
3. Locate the well from the **largest red-pixel cluster** (≈ column 909, row 54 after scaling).
4. Render head field `exp(-(r/(0.11·w))²) + 0.18·(x/w)` with `contourf` + white contours + well marker, using a **2-D `np.meshgrid`**, and **reshape** `buffer_rgba()` to `(h, w, 4)`.
5. Composite: `base = np.where(mask, game_render, original_banner)`.
6. Animate red core + 11-dot fading trail along a ping-pong path; clip to `mask`; add unclipped soft glow; `img.save(..., save_all=True, loop=0, duration=70)`.

> **Pitfalls already hit (avoid):** use **2-D `meshgrid`** (not 1-D) for `contourf`; **reshape** `buffer_rgba()` to `(h, w, 4)`; bound the mask to the **top band** so it does not grab "Arcade"/tagline; do NOT commit binary assets onto an unrelated WIP branch.

## 7. Caveats / open actions

| # | Note |
|---|------|
| 1 | Branch base `e3bf092` is **behind current master `e83dce8`**. Before merge, prefer **`git rebase master`** (or a merge commit); only content diff is the GIF, so conflict risk is minimal. |
| 2 | PR #53 is **open, not merged** — waiting on human review. Do **not** self-merge. |
| 3 | The branch also carries CI commit `9348566`; shared with other branches, **not** part of the B13 functional change. |
| 4 | An earlier mis-landed commit `4e439b8` (same B13 message) was accidentally created on `feat/mf6-mp7-backend`, then **detached** (recoverable via `git reflog`) — not on this branch, not on master. No action needed. |
| 5 | Asset-only change; regression/DQN/render3d suites are unaffected. |

**Bottom line:** branch is clean, pushed, in sync, PR #53 open into master. Likely next steps for the following agent: (a) rebase onto current master, (b) push, (c) hand PR #53 to the human for review/merge. Nothing is blocked.
