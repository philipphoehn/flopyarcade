"""Numba JIT PCG groundwater flow solver.

Replaces MODFLOW subprocess with in-process Numba-compiled PCG iterative solver.
Zero external dependencies (no C binaries, no SciPy sparse). Valid for:
  single-layer, uniform-K, CHD-boundary environments (1s-d, 1s-c, 1r-d, 1r-c).

Physics: Backward-Euler finite-difference discretization of
    Sy/dt * (h_new - h_old) = -div(K * grad(h_new)) + q

The linear system is:
    A @ h_new_int = b
where:
    A has diagonal = 4*beta + Sy/dt (constant for uniform grid)
    off-diagonals = -beta (4-neighbor stencil)
    b = (Sy/dt)*h_old + CHD_contributions + well_source

PCG with Jacobi preconditioner (diagonal is constant → trivial M^{-1}).
Using h_old as initial guess gives 1-3 iterations for similar RHS.
"""

import numpy as np
from numba import njit
from time import time


@njit(cache=True)
def _pcg_matvec(x, nrow, ncol, beta_x, beta_y, alpha, chd_w, chd_e, chd_n, chd_s):
    """Compute A @ x for interior cells using inline 5-point stencil.

    x: (nrow-2, ncol-2) interior heads
    Returns: (nrow-2, ncol-2) result
    """
    nr = nrow - 2  # interior rows
    nc = ncol - 2  # interior cols
    res = np.empty((nr, nc), dtype=np.float64)
    diag = 2.0 * beta_x + 2.0 * beta_y + alpha

    for i in range(nr):
        for j in range(nc):
            val = diag * x[i, j]

            # West
            if j > 0:
                val -= beta_x * x[i, j - 1]
            else:
                val -= beta_x * chd_w

            # East
            if j < nc - 1:
                val -= beta_x * x[i, j + 1]
            else:
                val -= beta_x * chd_e

            # North
            if i > 0:
                val -= beta_y * x[i - 1, j]
            else:
                val -= beta_y * chd_n

            # South
            if i < nr - 1:
                val -= beta_y * x[i + 1, j]
            else:
                val -= beta_y * chd_s

            res[i, j] = val

    return res


@njit(cache=True)
def _pcg_solve(h_old_int, nrow, ncol, beta_x, beta_y, alpha,
               chd_w, chd_e, chd_n, chd_s,
               well_i, well_j, well_q, tol, maxiter):
    """Solve A @ h_new = b using PCG with Jacobi preconditioner.

    Args:
        h_old_int: (nrow-2, ncol-2) previous step interior heads (used as initial guess)
        nrow, ncol: full grid dimensions
        beta_x: K / dCol^2
        beta_y: K / dRow^2
        alpha: Sy / dt
        chd_w, chd_e, chd_n, chd_s: boundary head values
        well_i, well_j: interior indices of well (-1 if none)
        well_q: well flow rate
        tol: relative tolerance
        maxiter: max PCG iterations

    Returns:
        h_new_int: (nrow-2, ncol-2) new interior heads
    """
    nr = nrow - 2
    nc = ncol - 2
    diag = 2.0 * beta_x + 2.0 * beta_y + alpha

    # Build RHS: b = alpha * h_old + CHD contributions + well
    b = np.empty((nr, nc), dtype=np.float64)
    for i in range(nr):
        for j in range(nc):
            b[i, j] = alpha * h_old_int[i, j]

    # CHD contributions (boundary-adjacent cells)
    for j in range(nc):
        b[j, 0] += beta_x * chd_w      # west boundary (col 0 of full grid)
        b[j, nc - 1] += beta_x * chd_e  # east boundary
    for i in range(nr):
        b[0, i] += beta_y * chd_n      # north boundary
        b[nr - 1, i] += beta_y * chd_s  # south boundary

    # Well source
    if well_i >= 0 and well_j >= 0:
        b[well_i, well_j] += well_q

    # Initial guess: h_old (good for similar RHS)
    x = h_old_int.copy()

    # r = b - A @ x
    Ax = _pcg_matvec(x, nrow, ncol, beta_x, beta_y, alpha, chd_w, chd_e, chd_n, chd_s)
    r = b - Ax

    # Jacobi precondition: z = r / diag (constant)
    inv_diag = 1.0 / diag
    z = r * inv_diag

    # rz = dot(r, z)
    rz = 0.0
    for i in range(nr):
        for j in range(nc):
            rz += r[i, j] * z[i, j]

    # Check initial residual
    b_norm = 0.0
    for i in range(nr):
        for j in range(nc):
            b_norm += b[i, j] * b[i, j]
    b_norm = np.sqrt(b_norm)
    r_norm = np.sqrt(rz * inv_diag)  # ||r|| = sqrt(rz/diag) since z=r/diag

    if r_norm < tol * b_norm:
        return x

    # p = z
    p = z.copy()

    for k in range(maxiter):
        # Ap = A @ p
        Ap = _pcg_matvec(p, nrow, ncol, beta_x, beta_y, alpha, chd_w, chd_e, chd_n, chd_s)

        # pAp = dot(p, Ap)
        pAp = 0.0
        for i in range(nr):
            for j in range(nc):
                pAp += p[i, j] * Ap[i, j]

        if pAp == 0.0:
            break

        # alpha_k = rz / pAp
        ak = rz / pAp

        # x += alpha_k * p
        for i in range(nr):
            for j in range(nc):
                x[i, j] += ak * p[i, j]

        # r -= alpha_k * Ap
        for i in range(nr):
            for j in range(nc):
                r[i, j] -= ak * Ap[i, j]

        # Check convergence
        r_norm = 0.0
        for i in range(nr):
            for j in range(nc):
                r_norm += r[i, j] * r[i, j]
        r_norm = np.sqrt(r_norm)

        if r_norm < tol * b_norm:
            break

        # z = r / diag
        z = r * inv_diag

        # rz_new = dot(r, z)
        rz_new = 0.0
        for i in range(nr):
            for j in range(nc):
                rz_new += r[i, j] * z[i, j]

        # beta_k = rz_new / rz
        bk = rz_new / rz
        rz = rz_new

        # p = z + beta_k * p
        for i in range(nr):
            for j in range(nc):
                p[i, j] = z[i, j] + bk * p[i, j]

    return x


@njit(cache=True)
def _advect_particle(h_int, px, py, nrow, ncol, K, dRow, dCol, dt, extentX, extentY):
    """Euler advection: v = -K * grad(h), x += v*dt.

    h_int: (nrow-2, ncol-2) interior heads
    px, py: particle position in model coords
    Returns: (px_new, py_new)
    """
    nr = nrow - 2
    nc = ncol - 2

    # Convert to cell indices (full grid)
    col_f = px / dCol
    row_f = py / dRow

    col = int(col_f)  # 0-based full grid col
    row = int(row_f)  # 0-based full grid row

    # Clamp to interior (1..nrow-2, 1..ncol-2 in full grid)
    col = max(1, min(col, ncol - 2))
    row = max(1, min(row, nrow - 2))

    # Interior indices
    ii = row - 1  # 0-based interior row
    jj = col - 1  # 0-based interior col

    # Get 4 neighbor heads for central difference
    def _get_h(i, j):
        # i, j are full grid indices
        ii2 = i - 1
        jj2 = j - 1
        if ii2 < 0:
            ii2 = 0
        if ii2 >= nr:
            ii2 = nr - 1
        if jj2 < 0:
            jj2 = 0
        if jj2 >= nc:
            jj2 = nc - 1
        return h_int[ii2, jj2]

    h_c = h_int[ii, jj]
    h_e = _get_h(row, col + 1)
    h_w = _get_h(row, col - 1)
    h_n = _get_h(row - 1, col)
    h_s = _get_h(row + 1, col)

    dh_dx = (h_e - h_w) / (2.0 * dCol)
    dh_dy = (h_n - h_s) / (2.0 * dRow)

    vx = -K * dh_dx
    vy = -K * dh_dy

    x_new = px + vx * dt
    y_new = py + vy * dt

    # Clip to domain
    if x_new < 0.0:
        x_new = 0.0
    if x_new > extentX:
        x_new = extentX
    if y_new < 0.0:
        y_new = 0.0
    if y_new > extentY:
        y_new = extentY

    return x_new, y_new


class NumbaSolverMixin:
    """Mixin providing `run_numba_solver()` as a drop-in for `runMODFLOW()`.

    Uses Numba JIT PCG with inline 5-point stencil.
    Zero external dependencies. No fill-in. O(N x iterations) per solve.
    """

    def _numba_setup(self):
        """Compute solver constants (called once at first step)."""
        nrow = self.nRow
        ncol = self.nCol
        K = self.hk
        Sy = 0.15
        dt = self.dt if hasattr(self, 'dt') and self.dt else 1.0
        dRow = self.dRow
        dCol = self.dCol

        self._nb_K = K
        self._nb_Sy = Sy
        self._nb_dt = dt
        self._nb_dRow = dRow
        self._nb_dCol = dCol
        self._nb_beta_x = K / (dCol * dCol)  # x-direction (col)
        self._nb_beta_y = K / (dRow * dRow)  # y-direction (row)
        self._nb_alpha = Sy / dt
        self._nb_nrow = nrow
        self._nb_ncol = ncol
        self._nb_h_prev = None  # (nrow-2, ncol-2) interior heads
        self._nb_ready = True

        # Pre-allocate working arrays to avoid re-allocation
        nr = nrow - 2
        nc = ncol - 2
        self._nb_nr = nr
        self._nb_nc = nc

    def run_numba_solver(self):
        """Drop-in replacement for runMODFLOW() + runMODPATH() + evaluateParticleTracking()."""
        t_start = time()

        if not hasattr(self, '_nb_ready') or not self._nb_ready:
            self._numba_setup()

        nr = self._nb_nr
        nc = self._nb_nc
        beta_x = self._nb_beta_x
        beta_y = self._nb_beta_y
        alpha = self._nb_alpha
        K = self._nb_K
        dRow = self._nb_dRow
        dCol = self._nb_dCol
        dt = self._nb_dt

        # CHD values
        chd_w = self.headSpecWest
        chd_e = self.headSpecEast
        chd_n = self.actionValueNorth
        chd_s = self.actionValueSouth

        # Well position (full grid 0-based)
        wl = self.wellCellLayer
        wr = self.wellCellRow  # 0-based row
        wc = self.wellCellColumn  # 0-based col
        # Interior indices
        well_i = wr - 1 if (wr >= 1 and wr <= self.nRow - 2) else -1
        well_j = wc - 1 if (wc >= 1 and wc <= self.nCol - 2) else -1
        well_q = self.wellQ

        # Previous heads (interior)
        if self._nb_h_prev is None:
            # First step: use starting head
            if hasattr(self, 'strt'):
                h_full = self.strt[0]  # (nrow, ncol)
            else:
                h_full = np.ones((self.nRow, self.nCol), dtype=np.float64)
            h_prev_int = h_full[1:-1, 1:-1].copy()
        else:
            h_prev_int = self._nb_h_prev

        # Solve with Numba PCG
        h_new_int = _pcg_solve(
            h_prev_int, self.nRow, self.nCol, beta_x, beta_y, alpha,
            chd_w, chd_e, chd_n, chd_s,
            well_i, well_j, well_q,
            1e-8, 200
        )

        # Reconstruct full head field
        h_full = np.zeros((self.nRow, self.nCol), dtype=np.float64)
        h_full[1:-1, 1:-1] = h_new_int

        # Fill boundaries
        h_full[:, 0] = chd_w
        h_full[:, -1] = chd_e
        h_full[0, :] = chd_n
        h_full[-1, :] = chd_s

        h_3d = h_full.reshape(1, self.nRow, self.nCol)

        # Advect particle
        px, py = self.particleX, self.particleY
        px_new, py_new = _advect_particle(
            h_new_int, px, py,
            self.nRow, self.nCol, K, dRow, dCol, dt,
            self.extentX, self.extentY
        )

        # Update state
        self.particleX = px_new
        self.particleY = py_new
        self.particleCoords = [px_new, py_new, self.particleZ]

        self.trajectories['x'].append(np.array([px_new]))
        self.trajectories['y'].append(np.array([py_new]))
        self.trajectories['z'].append(np.array([self.particleZ]))

        self.heads = h_3d
        self.times = [1.0]
        self.success_modflow = True
        self.buff = ''
        self.t_simulate = time() - t_start

        # Store for next step
        self._nb_h_prev = h_new_int.copy()

    def numba_reset(self):
        """Reset solver state for new episode."""
        self._nb_h_prev = None
        self._nb_ready = False
