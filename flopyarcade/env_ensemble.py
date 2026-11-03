#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
FloPyArcade - Ensemble Uncertainty Environment

Wraps the base FloPyArcade environment to simulate the same scenario
with N stochastic hydraulic conductivity (K) field realizations.

The agent observes a single state (from member 0 or ensemble-mean)
but the reward is the ensemble-mean, making the learned policy robust
to K uncertainty.

Usage:
    import gymnasium as gym
    from flopyarcade import FloPyArcadeEnsemble
    
    env = FloPyArcadeEnsemble(
        ENVTYPE='1s-d',
        n_ensemble=10,
        k_geomean=10.0,
        k_log_std=0.5,
        corr_length=3.0,
        seed=42,
    )
    obs, _ = env.reset()
    obs, reward, done, truncated, info = env.step(action)
    # info['ensemble'] contains per-member rewards, heads, K stats

Author: Philipp Hoehn
"""

import numpy as np
from copy import deepcopy
from typing import Optional, Any, Tuple

import gymnasium as gym
from gymnasium import spaces

from .ensemble_k import EnsembleKField


class FloPyArcadeEnsemble(gym.Env):
    """Ensemble uncertainty wrapper around FloPyArcade.

    Runs N independent simulations with different K fields per step.
    The reward is the ensemble-mean head-change reward.

    Parameters
    ----------
    ENVTYPE : str
        FloPyArcade environment type (e.g. '1s-d', '3r-c', '6s-d').
    n_ensemble : int
        Number of K field realizations (default 10).
    k_geomean : float
        Geometric mean hydraulic conductivity (default 10.0 m/d).
    k_log_std : float
        Std dev of log10(K) (default 0.5).
    corr_length : float
        Correlation length in grid cells (default 3.0).
    seed : Optional[int]
        Random seed for K field generation.
    **kwargs
        Additional keyword arguments passed to the base FloPyArcade env.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        ENVTYPE: str = "1s-d",
        n_ensemble: int = 10,
        k_geomean: float = 10.0,
        k_log_std: float = 0.5,
        corr_length: float = 3.0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.k_geomean = k_geomean
        self.k_log_std = k_log_std
        self.corr_length = corr_length
        self.seed = seed
        self.ENVTYPE = ENVTYPE
        self._kwargs = kwargs

        # Lazy-import base env to avoid circular issues
        from .flopyarcade import FloPyArcade

        # Base env for member 0 (used to get grid dimensions and action space)
        self._base = FloPyArcade(ENVTYPE=ENVTYPE, **kwargs)

        # Action space from base
        self.action_space = self._base.action_space

        # Observation space: same as base (agent sees single state)
        self.observation_space = self._base.observation_space

        # Ensemble K field generator (created on reset with grid dims)
        self._k_field: Optional[EnsembleKField] = None

        # Per-member environments (created on reset)
        self._members: list = []

        # Track current step rewards
        self._member_rewards: np.ndarray = np.zeros(n_ensemble)

    def reset(self, seed=None, **kwargs):
        """Reset the ensemble environment.

        Generates N K field realizations and resets all member environments.

        Returns
        -------
        obs : np.ndarray
            Observation from member 0.
        info : dict
            Reset info including ensemble stats.
        """
        if seed is not None:
            self._seed = seed
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        else:
            self._seed = self.seed

        # Create K field generator using base env grid dimensions
        nlay = self._base.nlay
        nrow = self._base.nrow
        ncol = self._base.ncol

        k_seed = self._seed if self._seed is not None else None
        self._k_field = EnsembleKField(
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            n_ensemble=self.n_ensemble,
            k_geomean=self.k_geomean,
            k_log_std=self.k_log_std,
            corr_length=self.corr_length,
            seed=k_seed,
        )

        # Reset base env first to get dimensions and initial state
        obs_base, info_base = self._base.reset(seed=self._seed, **kwargs)

        # Create member environments with different K fields
        # For MF2005 path: each member gets a spatially-varying hk array
        # We patch the LPF hk after creation
        self._members = []
        for i in range(self.n_ensemble):
            member_env = FloPyArcade(ENVTYPE=self.ENVTYPE, **self._kwargs)
            # Set spatial K for this member
            hk_array = self._k_field.get_member(i)
            # Patch LPF with spatial K
            if hasattr(member_env, 'mf_lpf'):
                member_env.mf_lpf.hk = hk_array
            # Store reference
            member_env._ensemble_k = hk_array
            self._members.append(member_env)

        # Reset member 0 (same as base)
        obs, info = self._members[0].reset(seed=self._seed, **kwargs)

        # For remaining members, reset and set K
        for i in range(1, self.n_ensemble):
            self._members[i].reset(seed=self._seed + i, **kwargs)
            # Patch LPF after reset (model is rebuilt on reset)
            if hasattr(self._members[i], 'mf_lpf'):
                self._members[i].mf_lpf.hk = self._k_field.get_member(i)

        info['ensemble'] = {
            'n_ensemble': self.n_ensemble,
            'k_stats': self._k_field.stats,
            'k_fields_shape': self._k_field.k_fields.shape,
        }

        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step the ensemble environment.

        Applies the same action to all N members, runs simulation,
        and returns ensemble-mean reward.

        Parameters
        ----------
        action : int or np.ndarray
            Action to apply (same for all members).

        Returns
        -------
        obs : np.ndarray
            Observation from member 0.
        reward : float
            Ensemble-mean reward.
        done : bool
            Episode done flag.
        truncated : bool
            Truncation flag.
        info : dict
            Contains 'ensemble' key with per-member rewards, heads, K stats.
        """
        rewards = []
        done_flags = []
        infos = []
        obs = None

        for i, member in enumerate(self._members):
            o, r, d, t, inf = member.step(action)
            rewards.append(r)
            done_flags.append(d)
            infos.append(inf)
            if i == 0:
                obs = o

        rewards = np.array(rewards)
        ensemble_reward = float(rewards.mean())

        info = {
            'ensemble': {
                'rewards': rewards,
                'reward_mean': ensemble_reward,
                'reward_std': float(rewards.std()),
                'reward_min': float(rewards.min()),
                'reward_max': float(rewards.max()),
                'k_stats': self._k_field.stats if self._k_field else None,
            }
        }
        # Merge member 0 info
        if infos and isinstance(infos[0], dict):
            info.update(infos[0])
            info['ensemble']['member0'] = infos[0].get('timestep', None)

        done = any(done_flags)
        truncated = False  # Simplified

        return obs, ensemble_reward, done, truncated, info

    def render(self, mode='human'):
        """Render member 0 state."""
        return self._members[0].render(mode=mode)

    def close(self):
        """Close all member environments."""
        for m in self._members:
            m.close()
        self._members.clear()

    @property
    def k_field(self) -> Optional[EnsembleKField]:
        """Access the K field generator."""
        return self._k_field

    def __repr__(self):
        return (f"FloPyArcadeEnsemble(ENVTYPE={self.ENVTYPE}, "
                f"N={self.n_ensemble}, K_geom={self.k_geomean}, "
                f"log_std={self.k_log_std}, L={self.corr_length})")
