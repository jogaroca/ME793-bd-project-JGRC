"""Measurement models for the Bd chemostat.

Measurement set A:
    y = [B, N, W]

where B = Z + S1 + S2 + S3 is the total biomass.
"""

from __future__ import annotations

import numpy as np


def h(x_vec, u_vec=None):
    """Noise-free measurement function for Set A.

    Parameters
    ----------
    x_vec : array_like, shape (6,)
        State vector x = [Z, S1, S2, S3, N, W].
    u_vec : array_like or None
        Control vector (unused here, included only for a consistent signature).

    Returns
    -------
    np.ndarray, shape (3,)
        Measurement vector y = [B, N, W].
    """
    Z, S1, S2, S3, N, W = x_vec
    B = Z + S1 + S2 + S3
    return np.array([B, N, W])


def meas_setA(X: np.ndarray, rng_seed: int = 123):
    """Generate noisy measurements for Set A.

    This follows the noise model used in the phase-1 simulations:

    - Relative noise (5%) on B and W.
    - Absolute noise of 0.05 mM on N.

    Parameters
    ----------
    X : array_like, shape (T, 6)
        State trajectory over time.
    rng_seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Y : np.ndarray, shape (T, 3)
        Noisy measurements [B_m, N_m, W_m].
    B_true : np.ndarray, shape (T,)
        True (noise-free) biomass B.
    """
    rng = np.random.default_rng(rng_seed)
    Z, S1, S2, S3, N, W = np.asarray(X).T

    B_true = Z + S1 + S2 + S3

    sigmaB = 0.05 * np.maximum(B_true, 0.0)
    sigmaN = 0.05 * np.ones_like(N)  # absolute 0.05 mM
    sigmaW = 0.05 * np.maximum(W, 0.0)

    Bm = B_true + rng.normal(0.0, sigmaB)
    Nm = N + rng.normal(0.0, sigmaN)
    Wm = W + rng.normal(0.0, sigmaW)

    Y = np.column_stack([Bm, Nm, Wm])
    return Y, B_true
