#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
FloPyArcade - Ensemble Hydraulic Conductivity Field Generation

Generates N spatially-correlated log-normal hydraulic conductivity (K) fields
for ensemble uncertainty quantification.

Each ensemble member is a realization of a random log-K field:
    log10(K) ~ N(mu, sigma^2) with exponential spatial covariance
    C(h) = sigma^2 * exp(-|h| / L)

where:
    mu  = log10(geometric mean K)
    sigma = standard deviation of log10(K)
    L   = correlation length (in grid cell units)
    h   = separation distance

The fields are generated via Cholesky decomposition of the covariance matrix,
which is exact for small-to-medium grids (<= ~200x200). For larger grids,
a truncated spectral method is used as fallback.

Author: Philipp Hoehn
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class EnsembleKField:
    """Generate and manage N spatially-correlated log-normal K fields.

    Parameters
    ----------
    nlay : int
        Number of layers.
    nrow : int
        Number of rows.
    ncol : int
        Number of columns.
    n_ensemble : int
        Number of ensemble members (K field realizations).
    k_geomean : float
        Geometric mean hydraulic conductivity (same units as model, e.g. m/d).
    k_log_std : float
        Standard deviation of log10(K). Typical range 0.3-1.0.
    corr_length : float
        Correlation length in grid-cell units. Typical: 2-5 cells.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        nlay: int,
        nrow: int,
        ncol: int,
        n_ensemble: int = 10,
        k_geomean: float = 10.0,
        k_log_std: float = 0.5,
        corr_length: float = 3.0,
        seed: Optional[int] = None,
    ):
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.n_cells = nlay * nrow * ncol
        self.n_ensemble = n_ensemble
        self.k_geomean = k_geomean
        self.k_log_std = k_log_std
        self.corr_length = corr_length
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Precompute covariance matrix and Cholesky factor
        self._build_covariance()

        # Generate all ensemble members
        self.k_fields: np.ndarray = self._generate_fields()  # shape (n_ensemble, nlay, nrow, ncol)

    def _build_covariance(self):
        """Build the spatial covariance matrix for log10(K).

        Uses exponential covariance: C(i,j) = sigma^2 * exp(-dist(i,j) / L)
        where dist is the Euclidean distance in grid-cell units.
        """
        n = self.n_cells
        sigma2 = self.k_log_std ** 2

        # Build coordinate grid
        iz, iy, ix = np.indices((self.nlay, self.nrow, self.ncol))
        coords = np.column_stack([iz.ravel(), iy.ravel(), ix.ravel()])  # (n, 3)

        # Pairwise distances (for n <= 10000, full matrix is feasible)
        if n <= 10000:
            # Full covariance via broadcasting (memory: n*n*8 bytes)
            # Chunk to avoid OOM for very large grids
            C = np.empty((n, n), dtype=np.float64)
            chunk = max(1, min(n, 2000))
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                diff = coords[start:end, :][:, np.newaxis, :] - coords[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (chunk, n)
                C[start:end, :] = sigma2 * np.exp(-dist / self.corr_length)
        else:
            # For very large grids, use a diagonal approximation + local correlation
            C = np.eye(n) * sigma2
            # Add local neighborhood correlation
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        dist = np.sqrt(dz**2 + dy**2 + dx**2)
                        if dist <= 5 * self.corr_length:
                            w = sigma2 * np.exp(-dist / self.corr_length)
                            # Shift indices
                            i_idx = iz.ravel() + dz
                            j_idx = iy.ravel() + dy
                            k_idx = ix.ravel() + dx
                            valid = (i_idx >= 0) & (i_idx < self.nlay) & \
                                    (j_idx >= 0) & (j_idx < self.nrow) & \
                                    (k_idx >= 0) & (k_idx < self.ncol)
                            rows = (i_idx * self.nrow * self.ncol + j_idx * self.ncol + k_idx)[valid]
                            cols = np.arange(n)[valid]
                            C[rows, cols] += w

        # Ensure symmetry and positive semi-definiteness
        C = 0.5 * (C + C.T)
        # Add small jitter for numerical stability
        C += np.eye(n) * 1e-10

        # Cholesky decomposition
        try:
            self._L = np.linalg.cholesky(C)
            self._method = 'cholesky'
        except np.linalg.LinAlgError:
            # Fallback: eigenvalue decomposition, clip negative eigenvalues
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            self._L = eigvecs @ np.diag(np.sqrt(eigvals))
            self._method = 'eigen'

    def _generate_fields(self) -> np.ndarray:
        """Generate N ensemble K field realizations.

        Returns
        -------
        np.ndarray of shape (n_ensemble, nlay, nrow, ncol)
        """
        n = self.n_cells
        mu = np.log10(self.k_geomean)

        # Generate N standard normal vectors
        Z = self.rng.standard_normal((self.n_ensemble, n))  # (N, n)

        # Apply Cholesky factor to get correlated normals
        logK = Z @ self._L.T + mu  # (N, n)

        # Transform back: K = 10^logK
        K = 10.0 ** logK  # (N, n)

        # Reshape to 4D
        K_fields = K.reshape(self.n_ensemble, self.nlay, self.nrow, self.ncol)

        return K_fields

    def get_member(self, idx: int) -> np.ndarray:
        """Get a single K field realization.

        Parameters
        ----------
        idx : int
            Ensemble member index (0-based).

        Returns
        -------
        np.ndarray of shape (nlay, nrow, ncol)
        """
        return self.k_fields[idx]

    def get_uniform_k(self) -> np.ndarray:
        """Get uniform K for all cells (equivalent to original behavior).

        Returns
        -------
        np.ndarray of shape (nlay, nrow, ncol)
        """
        return np.full((self.nlay, self.nrow, self.ncol), self.k_geomean)

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary statistics of the ensemble."""
        flat = self.k_fields.reshape(self.n_ensemble, -1)
        return {
            'n_ensemble': self.n_ensemble,
            'k_geomean_target': self.k_geomean,
            'k_log_std_target': self.k_log_std,
            'corr_length': self.corr_length,
            'k_min': float(flat.min()),
            'k_max': float(flat.max()),
            'k_mean_over_members': float(flat.mean(axis=1).mean()),
            'k_std_over_members': float(flat.mean(axis=1).std()),
            'method': self._method,
            'seed': self.seed,
        }

    def __repr__(self):
        s = self.stats
        return (f"EnsembleKField(N={s['n_ensemble']}, "
                f"grid={self.nlay}x{self.nrow}x{self.ncol}, "
                f"K_geom={s['k_geomean_target']}, "
                f"log_std={s['k_log_std_target']}, "
                f"L={s['corr_length']}, "
                f"K_range=[{s['k_min']:.2f}, {s['k_max']:.2f}], "
                f"method={s['method']})")
