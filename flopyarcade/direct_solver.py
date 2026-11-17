"""Direct pre-factorized groundwater flow solver.

Replaces MODFLOW subprocess with in-process SciPy sparse LU solve.
Valid for: single-layer, uniform-K, CHD-boundary environments (1s-d, 1s-c, 1r-d, 1r-c).

Physics: Backward-Euler finite-difference discretization of
    Sy/dt * (h_new - h_old) = -div(K * grad(h_new)) + q
which gives the linear system:
    A @ h_new = (Sy/dt) * h_old + b_chd + b_well

A is constant (depends only on grid + K + Sy/dt), so we factorize once.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from time import time
from copy import copy


class DirectSolverMixin:
    """Mixin providing `run_direct_solver()` as a drop-in for `runMODFLOW()`.

    Expected attributes on self (set by env_base / modflow_runner):
        nLay, nRow, nCol  – grid dimensions
        dRow, dCol        – cell sizes (delr, delc)
        hk                – hydraulic conductivity (uniform)
        Sy                – specific yield (0.15)
        dt                – timestep (periodLength, 1.0)
        periodSteps       – timesteps per episode (11)
        headSpecWest, headSpecEast  – CHD values
        actionValueSouth, actionValueNorth – CHD values
        wellCellLayer, wellCellColumn, wellCellRow – 0-based well cell
        wellQ             – well flow rate (negative = pumping)
        extentX, extentY  – domain extent
        minX, minY        – domain origin
        zBot, zTop        – vertical extent
        particleX, particleY, particleZ – particle position
        trajectories      – dict of position lists
        timeStep          – current step index (0-based)
        ENVTYPE           – environment type string
    """

    def _direct_build_matrix(self):
        """Build and factorize the constant coefficient matrix A.

        For 1s-d: all 4 boundaries are CHD (ibound=-1).
        Interior cells: rows 1..nRow-2, cols 1..nCol-2.
        """
        nrow, ncol = self.nRow, self.nCol
        K = self.hk
        Sy = 0.15  # specific yield (from LPF package)
        dt = self.dt if hasattr(self, 'dt') and self.dt else 1.0
        dRow = self.dRow  # delr
        dCol = self.dCol  # delc

        # Transmissivities (uniform grid, single layer)
        # T for x-direction flow: K * dRow (perpendicular distance)
        # T for y-direction flow: K * dCol
        Tx = K * dRow  # flow in x (col) direction, thickness = dRow
        Ty = K * dCol  # flow in y (row) direction, thickness = dCol

        # Interior mask: exclude all boundary cells
        N = nrow * ncol
        int_mask = np.ones(N, dtype=bool)
        # Row boundaries (top and bottom)
        int_mask[0 * ncol:(0 + 1) * ncol] = False       # row 0
        int_mask[(nrow - 1) * ncol:(nrow) * ncol] = False  # row nrow-1
        # Col boundaries (left and right)
        for r in range(nrow):
            int_mask[r * ncol + 0] = False
            int_mask[r * ncol + (ncol - 1)] = False

        int_idx = np.where(int_mask)[0]
        Nint = len(int_idx)

        # Mapping: flat index → interior index
        flat_to_int = np.full(N, -1, dtype=np.int32)
        for i, f in enumerate(int_idx):
            flat_to_int[f] = i

        # Build A (LIL for easy construction)
        A = lil_matrix((Nint, Nint))
        self._b_chd = np.zeros(Nint)  # constant CHD contribution (values change per episode)

        # FIXED: Diagonal is ALWAYS 4*K/dx² + Sy/dt for 2D (4-neighbor stencil)
        # CHD neighbors do NOT reduce the diagonal; their contribution goes to RHS.
        diag_val = 4.0 * (K / (dRow * dCol) if dRow != dCol else K / (dRow ** 2)) + Sy / dt
        # For uniform grid (dRow == dCol == dx): diag = 4K/dx² + Sy/dt
        if dRow == dCol:
            beta_x = K / (dRow ** 2)
            beta_y = beta_x
        else:
            beta_x = K / (dRow ** 2)  # x-direction (col)
            beta_y = K / (dCol ** 2)  # y-direction (row)
            diag_val = 2.0 * beta_x + 2.0 * beta_y + Sy / dt

        for ii, flat in enumerate(int_idx):
            r = flat // ncol
            c = flat % ncol

            # West neighbor (c-1)
            if c - 1 == 0:
                pass  # CHD: no off-diagonal, contribution in RHS
            else:
                j = flat_to_int[r * ncol + (c - 1)]
                if j >= 0:
                    A[ii, j] = -beta_x

            # East neighbor (c+1)
            if c + 1 == ncol - 1:
                pass  # CHD
            else:
                j = flat_to_int[r * ncol + (c + 1)]
                if j >= 0:
                    A[ii, j] = -beta_x

            # North neighbor (r-1)
            if r - 1 == 0:
                pass  # CHD
            else:
                j = flat_to_int[(r - 1) * ncol + c]
                if j >= 0:
                    A[ii, j] = -beta_y

            # South neighbor (r+1)
            if r + 1 == nrow - 1:
                pass  # CHD
            else:
                j = flat_to_int[(r + 1) * ncol + c]
                if j >= 0:
                    A[ii, j] = -beta_y

            # Diagonal: ALWAYS 4*beta + alpha (unchanged by CHD neighbors)
            A[ii, ii] = diag_val

        A = A.tocsr()

        # Factorize (one-time cost ~20ms for 9604x9604)
        self._direct_lu = splu(A.tocsc())

        # Store metadata
        self._direct_Nint = Nint
        self._direct_flat_to_int = flat_to_int
        self._direct_int_idx = int_idx
        self._direct_int_mask = int_mask
        self._direct_K = K
        self._direct_Sy = Sy
        self._direct_dt = dt
        self._direct_Tx = Tx
        self._direct_Ty = Ty
        self._direct_nrow = nrow
        self._direct_ncol = ncol
        self._direct_head_field = None  # will be set after first solve

        # Store which interior cells are adjacent to each CHD boundary
        # for efficient RHS computation
        self._chd_west_cells = []   # (ii, head_value_slot)
        self._chd_east_cells = []
        self._chd_north_cells = []
        self._chd_south_cells = []

        for ii, flat in enumerate(int_idx):
            r = flat // ncol
            c = flat % ncol
            if c == 1:      # adjacent to west boundary (col 0)
                self._chd_west_cells.append(ii)
            if c == ncol - 2:  # adjacent to east boundary (col ncol-1)
                self._chd_east_cells.append(ii)
            if r == 1:      # adjacent to north boundary (row 0)
                self._chd_north_cells.append(ii)
            if r == nrow - 2:  # adjacent to south boundary (row nrow-1)
                self._chd_south_cells.append(ii)

        # Convert to arrays for vectorized RHS
        self._chd_west_cells = np.array(self._chd_west_cells, dtype=np.int32)
        self._chd_east_cells = np.array(self._chd_east_cells, dtype=np.int32)
        self._chd_north_cells = np.array(self._chd_north_cells, dtype=np.int32)
        self._chd_south_cells = np.array(self._chd_south_cells, dtype=np.int32)

        self._direct_built = True

    def _direct_solve_step(self, h_old_flat, well_flat, well_q, 
                           chd_west, chd_east, chd_north, chd_south):
        """Solve one timestep: A @ h_new = Sy/dt * h_old + b_chd + b_well.

        Args:
            h_old_flat: (Nint,) previous heads (interior cells only)
            well_flat: flat index of well cell (full grid), or -1 if none
            well_q: well flow rate (negative = pumping)
            chd_west, chd_east, chd_north, chd_south: boundary head values

        Returns:
            h_new: (Nint,) new heads (interior cells only)
        """
        Sy = self._direct_Sy
        dt = self._direct_dt
        Tx = self._direct_Tx
        Ty = self._direct_Ty

        # RHS = Sy/dt * h_old
        b = (Sy / dt) * h_old_flat

        # Add CHD contributions (vectorized)
        b[self._chd_west_cells] += Tx * chd_west
        b[self._chd_east_cells] += Tx * chd_east
        b[self._chd_north_cells] += Ty * chd_north
        b[self._chd_south_cells] += Ty * chd_south

        # Add well source (if well is interior)
        if well_flat >= 0:
            wi = self._direct_flat_to_int[well_flat]
            if wi >= 0:
                b[wi] += well_q  # Q is already in the right units (L^3/T)

        # Solve
        h_new = self._direct_lu.solve(b)
        return h_new

    def _direct_advect_particle(self, h_int, particle_x, particle_y, dt):
        """Euler advection of particle using head gradient.

        v = -K * grad(h)
        x_new = x + vx * dt
        y_new = y + vy * dt

        Args:
            h_int: (Nint,) interior heads
            particle_x, particle_y: current position (model coordinates)
            dt: timestep

        Returns:
            (x_new, y_new)
        """
        nrow = self._direct_nrow
        ncol = self._direct_ncol
        K = self._direct_K
        dRow = self.dRow  # delr (row size in y-direction)
        dCol = self.dCol  # delc (col size in x-direction)

        # Convert particle position to cell indices
        # Model coords: x in [0, extentX), y in [0, extentY)
        # Grid: col = x/dCol, row = y/dRow (0-based)
        col_f = particle_x / dCol
        row_f = particle_y / dRow

        col = int(col_f)
        row = int(row_f)

        # Clamp to interior
        col = max(1, min(col, ncol - 2))
        row = max(1, min(row, nrow - 2))

        # Get head at 4 neighbors for central difference
        def get_head(r, c):
            flat = r * ncol + c
            idx = self._direct_flat_to_int[flat]
            if idx >= 0:
                return h_int[idx]
            # Boundary cell: use CHD value
            # For our purposes, clamp to nearest interior
            r_c = max(1, min(r, nrow - 2))
            c_c = max(1, min(c, ncol - 2))
            return h_int[self._direct_flat_to_int[r_c * ncol + c_c]]

        # Central difference gradient
        h_c = get_head(row, col)
        h_e = get_head(row, col + 1)
        h_w = get_head(row, col - 1)
        h_n = get_head(row - 1, col)
        h_s = get_head(row + 1, col)

        dh_dx = (h_e - h_w) / (2.0 * dCol)
        dh_dy = (h_n - h_s) / (2.0 * dRow)

        # Darcy velocity
        vx = -K * dh_dx
        vy = -K * dh_dy

        # Euler step
        x_new = particle_x + vx * dt
        y_new = particle_y + vy * dt

        return x_new, y_new

    def run_direct_solver(self):
        """Drop-in replacement for runMODFLOW() + runMODPATH() + evaluateParticleTracking().

        Solves the flow equation using pre-factorized matrix and advects
        the particle using Euler method.
        """
        t_start = time()

        # Build matrix on first call
        if not hasattr(self, '_direct_built') or not self._direct_built:
            self._direct_build_matrix()

        # Get CHD boundary values
        chd_west = self.headSpecWest
        chd_east = self.headSpecEast
        chd_north = self.actionValueNorth
        chd_south = self.actionValueSouth

        # Get well position (flat index)
        wl = self.wellCellLayer
        wr = self.wellCellRow
        wc = self.wellCellColumn
        well_flat = wl * self.nRow * self.nCol + wr * self.nCol + wc
        well_q = self.wellQ

        # Get previous heads (interior)
        if not hasattr(self, '_direct_h_prev') or self._direct_h_prev is None:
            # First step: use starting head (strt)
            h_full = self.strt if hasattr(self, 'strt') else np.ones((1, self.nRow, self.nCol), dtype=np.float64)
            h_flat = h_full.flatten()
            h_prev_int = h_flat[self._direct_int_idx]
        else:
            h_prev_int = self._direct_h_prev

        # Solve
        h_new_int = self._direct_solve_step(
            h_prev_int, well_flat, well_q,
            chd_west, chd_east, chd_north, chd_south
        )

        # Reconstruct full head field (1, nRow, nCol)
        h_full = np.zeros(1 * self.nRow * self.nCol, dtype=np.float64)
        h_full[self._direct_int_idx] = h_new_int

        # Fill boundary cells with CHD values
        for r in range(self.nRow):
            # West (col 0)
            h_full[r * self.nCol + 0] = chd_west
            # East (col nCol-1)
            h_full[r * self.nCol + (self.nCol - 1)] = chd_east
        for c in range(self.nCol):
            # North (row 0)
            h_full[0 * self.nCol + c] = chd_north
            # South (row nRow-1)
            h_full[(self.nRow - 1) * self.nCol + c] = chd_south

        h_3d = h_full.reshape(1, self.nRow, self.nCol)

        # Advect particle
        px, py = self.particleX, self.particleY
        dt = self._direct_dt
        px_new, py_new = self._direct_advect_particle(h_new_int, px, py, dt)

        # Clip to domain
        px_new = max(0.0, min(self.extentX, px_new))
        py_new = max(0.0, min(self.extentY, py_new))

        # Update state (match what modpath_runner sets)
        self.particleX = px_new
        self.particleY = py_new
        self.particleZ = self.particleZ  # unchanged (single layer)
        self.particleCoords = [px_new, py_new, self.particleZ]

        # Update trajectories
        self.trajectories['x'].append(np.array([px_new]))
        self.trajectories['y'].append(np.array([py_new]))
        self.trajectories['z'].append(np.array([self.particleZ]))

        # Set outputs expected by the rest of the code
        self.heads = h_3d
        self.times = [1.0]  # dummy (MF2005 reads from .hds)
        self.success_modflow = True
        self.buff = ''
        self.t_simulate = time() - t_start

        # Store for next step
        self._direct_h_prev = h_new_int.copy()

    def direct_reset(self):
        """Reset solver state for new episode."""
        self._direct_h_prev = None
        self._direct_built = False
        if hasattr(self, '_direct_lu'):
            del self._direct_lu
