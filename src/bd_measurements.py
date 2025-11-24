"""
Measurement functions for the Bd chemostat model.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def h(x_vec, u_vec):
    """
    Noise-free measurement function for measurement set A.

    Returns [B, N, W] where B is the total biomass:
        B = Z + S1 + S2 + S3.
    """
    Z, S1, S2, S3, N, W = x_vec
    B = Z + S1 + S2 + S3
    return [B, N, W]


def meas_setA(X: np.ndarray, rng_seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy measurements corresponding to measurement set A.

    Measurements:
        y1 = B + noise   (relative noise, 5% of B)
        y2 = N + noise   (absolute noise, std = 0.05 mM)
        y3 = W + noise   (relative noise, 5% of W)

    Parameters
    ----------
    X : ndarray, shape (T, 6)
        State trajectory, each row is [Z, S1, S2, S3, N, W].
    rng_seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Y : ndarray, shape (T, 3)
        Noisy measurements [B_m, N_m, W_m].
    B_true : ndarray, shape (T,)
        Noise-free total biomass B.
    """
    rng = np.random.default_rng(rng_seed)
    Z, S1, S2, S3, N, W = X.T
    B_true = Z + S1 + S2 + S3

    sigmaB = 0.05 * np.maximum(B_true, 0.0)
    sigmaN = 0.05 * np.ones_like(N)  # absolute 0.05 mM
    sigmaW = 0.05 * np.maximum(W, 0.0)

    Bm = B_true + rng.normal(0.0, sigmaB)
    Nm = N + rng.normal(0.0, sigmaN)
    Wm = W + rng.normal(0.0, sigmaW)

    Y = np.column_stack([Bm, Nm, Wm])
    return Y, B_true


__all__ = ["h", "meas_setA"]
