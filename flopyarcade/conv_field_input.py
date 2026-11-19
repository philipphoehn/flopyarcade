#!/usr/bin/python3
# -*- coding: utf-8 -*-

# FloPyArcade - conv_field_input
# A3: ConvNet full-field input
# Provides proper 2D spatial field reshaping and Conv2D model construction
# so that head/velocity/pressure 2D arrays feed Conv2D layers directly.

"""Full-field Conv2D input pipeline for FloPyArcade.

The existing ``agent.py`` builds a Conv2D model with
``input_shape = shape(self.observationsVector)`` which is 1-D (N,).
That works only because Keras implicitly treats the first axis as the
channel dimension for a 2-D input, which is not the intended spatial
layout.

This module provides:

* **FieldReshaper** – reshapes a flat observation vector into a
  ``(nrow, ncol, nchannels)`` spatial field ready for Conv2D.
* **build_conv_model** – constructs a Keras Sequential model whose
  first layer has the correct 4-D input shape ``(nrow, ncol, C)``.
* **ConvFieldPipeline** – convenience wrapper that chains reshape →
  model forward pass.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# FieldReshaper
# ---------------------------------------------------------------------------


class FieldReshaper:
    """Reshape a flat observation vector into a 2-D spatial field.

    Parameters
    ----------
    nrow : int
        Number of grid rows (height of the spatial field).
    ncol : int
        Number of grid columns (width of the spatial field).
    nchannels : int, optional
        Number of feature channels per cell.  Defaults to 1.
    """

    def __init__(self, nrow: int, ncol: int, nchannels: int = 1) -> None:
        if nrow <= 0:
            raise ValueError("nrow must be positive")
        if ncol <= 0:
            raise ValueError("ncol must be positive")
        if nchannels <= 0:
            raise ValueError("nchannels must be positive")
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.nchannels = int(nchannels)
        self.expected_flat = self.nrow * self.ncol * self.nchannels

    # -- properties ---------------------------------------------------------

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Return ``(nrow, ncol, nchannels)``."""
        return (self.nrow, self.ncol, self.nchannels)

    # -- core ---------------------------------------------------------------

    def reshape(self, flat: np.ndarray) -> np.ndarray:
        """Reshape a 1-D (or already-shaped) vector to ``(nrow, ncol, C)``.

        Parameters
        ----------
        flat : np.ndarray
            Observation vector of length ``nrow * ncol * nchannels``.

        Returns
        -------
        np.ndarray with shape ``(nrow, ncol, nchannels)``.
        """
        arr = np.asarray(flat, dtype=np.float64).ravel()
        if arr.size != self.expected_flat:
            raise ValueError(
                f"expected {self.expected_flat} elements, got {arr.size}"
            )
        return arr.reshape(self.nrow, self.ncol, self.nchannels)

    def batch(self, flat_batch: np.ndarray) -> np.ndarray:
        """Reshape a batch of flat vectors to ``(B, nrow, ncol, C)``."""
        arr = np.asarray(flat_batch, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 2 or arr.shape[1] != self.expected_flat:
            raise ValueError(
                f"expected (B, {self.expected_flat}), got {arr.shape}"
            )
        B = arr.shape[0]
        return arr.reshape(B, self.nrow, self.ncol, self.nchannels)

    def __repr__(self) -> str:
        return (
            f"FieldReshaper(nrow={self.nrow}, ncol={self.ncol}, "
            f"nchannels={self.nchannels})"
        )


# ---------------------------------------------------------------------------
# build_conv_model
# ---------------------------------------------------------------------------


def build_conv_model(
    nrow: int,
    ncol: int,
    nchannels: int,
    action_size: int,
    action_type: str = "discrete",
    seed: int | None = None,
) -> "keras.Model":
    """Build a Keras Conv2D model with correct 4-D input shape.

    Parameters
    ----------
    nrow, ncol, nchannels : int
        Spatial field dimensions.
    action_size : int
        Number of output actions.
    action_type : str
        ``'discrete'`` → linear output; ``'continuous'`` → sigmoid output.
    seed : int or None
        Random seed for reproducible initialisation.

    Returns
    -------
    keras.Sequential
    """
    from tensorflow.keras.layers import (
        Conv2D, Dense, Flatten, BatchNormalization, Activation,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.initializers import glorot_normal

    if seed is not None:
        import tensorflow as tf
        tf.random.set_seed(seed)
        np.random.seed(seed)
        initializer = glorot_normal(seed=seed)
    else:
        initializer = glorot_normal()

    input_shape = (nrow, ncol, nchannels)

    model = Sequential()
    model.add(Conv2D(
        32, kernel_size=(8, 8), strides=(4, 4),
        activation='relu',
        kernel_initializer=initializer,
        bias_initializer=initializer,
        input_shape=input_shape, padding='same',
    ))
    model.add(BatchNormalization())
    model.add(Conv2D(
        64, kernel_size=(4, 4), strides=(2, 2),
        kernel_initializer=initializer,
        bias_initializer=initializer,
        activation='relu', padding='same',
    ))
    model.add(Conv2D(
        64, kernel_size=(3, 3), strides=(1, 1),
        kernel_initializer=initializer,
        bias_initializer=initializer,
        activation='relu', padding='same',
    ))
    model.add(Flatten())
    model.add(Dense(
        512,
        kernel_initializer=initializer,
        bias_initializer=initializer,
        activation='relu',
    ))
    model.add(BatchNormalization())

    if action_type == 'discrete':
        model.add(Dense(action_size, activation='linear',
                        kernel_initializer=initializer))
    elif action_type == 'continuous':
        model.add(Dense(action_size, activation='sigmoid',
                        kernel_initializer=initializer))
    else:
        raise ValueError(f"unknown action_type: {action_type!r}")

    return model


# ---------------------------------------------------------------------------
# ConvFieldPipeline
# ---------------------------------------------------------------------------


class ConvFieldPipeline:
    """Chain a FieldReshaper with a Conv2D model.

    Parameters
    ----------
    nrow, ncol, nchannels : int
        Spatial dimensions.
    action_size : int
        Number of output actions.
    action_type : str
        ``'discrete'`` or ``'continuous'``.
    seed : int or None
        Reproducibility seed.
    model : keras.Model, optional
        Pre-built model.  If None, one is built via :func:`build_conv_model`.
    """

    def __init__(
        self,
        nrow: int,
        ncol: int,
        nchannels: int = 1,
        action_size: int = 5,
        action_type: str = "discrete",
        seed: int | None = None,
        model: "keras.Model | None" = None,
    ) -> None:
        self.reshaper = FieldReshaper(nrow, ncol, nchannels)
        if model is None:
            self.model = build_conv_model(
                nrow, ncol, nchannels, action_size, action_type, seed,
            )
        else:
            self.model = model

    def forward(self, flat_obs: np.ndarray) -> np.ndarray:
        """Reshape flat observation and run the model (single sample)."""
        field = self.reshaper.reshape(flat_obs)
        return self.model.predict(np.expand_dims(field, axis=0), verbose=0)

    def forward_batch(self, flat_batch: np.ndarray) -> np.ndarray:
        """Reshape a batch and run the model."""
        field = self.reshaper.batch(flat_batch)
        return self.model.predict(field, verbose=0)

    def __repr__(self) -> str:
        return (
            f"ConvFieldPipeline("
            f"nrow={self.reshaper.nrow}, ncol={self.reshaper.ncol}, "
            f"C={self.reshaper.nchannels}, "
            f"action_size={self.model.output_shape[-1]})"
        )
