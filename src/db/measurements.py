"""
Measurement model and synthetic-noise utilities for the Bd chemostat.

We work with the "measurement set A" used in the project report, where
the measured outputs are

    y = [B, N, W]

with
    B = Z + S1 + S2 + S3  : total biomass (cells or a.u.)
    N                     : nutrient concentration (mM)
    W                     : waste / inhibitory by-product (a.u.)

This module also provides a convenience routine to generate noisy
measurements and to build an approximate measurement-noise covariance
matrix R that is consistent with the noise model.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Names of the three measured variables
MEASUREMENT_NAMES = ["B", "N", "W"]


def h(x_vec, u_vec, return_measurement_names: bool = False):
    """
    Continuous-time measurement function y = h(x, u).

    Parameters
    ----------
    x_vec
        State vector [Z, S1, S2, S3, N, W].
    u_vec
        Input vector. It is not used here, but kept for compatibility.
    return_measurement_names
        If True, return the list of measurement names instead of y.

    Returns
    -------
    y : ndarray, shape (3,)
        Measurement vector [B, N, W].
    """
    if return_measurement_names:
        return MEASUREMENT_NAMES

    Z, S1, S2, S3, N, W = x_vec
    B = Z + S1 + S2 + S3
    return np.array([B, N, W], dtype=float)


def meas_setA(X: np.ndarray, rng_seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy measurements according to the Set A noise model.

    Parameters
    ----------
    X
        Array of true states with shape (T, 6).
    rng_seed
        Seed for NumPy's random number generator.

    Returns
    -------
    Y : ndarray, shape (T, 3)
        Noisy measurements [B_m, N_m, W_m].
    B_true : ndarray, shape (T,)
        True biomass time series B(t) = Z + S1 + S2 + S3.
    """
    rng = np.random.default_rng(rng_seed)
    Z, S1, S2, S3, N, W = X.T

    B_true = Z + S1 + S2 + S3

    # Standard deviations for each measurement channel
    sigmaB = 0.05 * np.maximum(B_true, 0.0)   # 5% relative noise on B
    sigmaN = 0.05 * np.ones_like(N)          # absolute 0.05 mM noise on N
    sigmaW = 0.05 * np.maximum(W, 0.0)       # 5% relative noise on W

    Bm = B_true + rng.normal(0.0, sigmaB)
    Nm = N + rng.normal(0.0, sigmaN)
    Wm = W + rng.normal(0.0, sigmaW)

    Y = np.column_stack([Bm, Nm, Wm])
    return Y, B_true


def build_noise_covariance_from_trajectory(Y_true: np.ndarray) -> np.ndarray:
    """
    Construct a simple diagonal measurement-noise covariance matrix R.

    The construction mirrors the noise model used in :func:`meas_setA`,
    using the *true* (noise-free) outputs ``Y_true`` to approximate
    typical variances for each channel.

    Parameters
    ----------
    Y_true
        Array of true (noise-free) outputs with shape (T, 3).

    Returns
    -------
    R : ndarray, shape (3, 3)
        Diagonal covariance matrix diag(var_B, var_N, var_W).
    """
    B_true = Y_true[:, 0]
    N_true = Y_true[:, 1]
    W_true = Y_true[:, 2]

    sigmaB = 0.05 * np.maximum(B_true, 0.0)
    sigmaN = 0.05 * np.ones_like(N_true)
    sigmaW = 0.05 * np.maximum(W_true, 0.0)

    var_B = float(np.mean(sigmaB**2))
    var_N = float(np.mean(sigmaN**2))
    var_W = float(np.mean(sigmaW**2))

    R = np.diag([var_B, var_N, var_W])
    return R
