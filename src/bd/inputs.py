"""Control input motifs for the Bd chemostat.

All functions follow the signature

    u_func(x_vec, t) -> [u_D]

so that they can be passed directly to :mod:`scipy.integrate.odeint`
via the wrapper :func:`bd.dynamics.f_ode`.
"""

from __future__ import annotations

import numpy as np

from .dynamics import D_min, D_max


def u_const(x_vec, t, D: float = 0.12):
    """Constant dilution rate.

    Parameters
    ----------
    x_vec : array_like
        Current state (unused, included for a consistent signature).
    t : float
        Current time (unused).
    D : float, optional
        Constant value of the dilution rate.

    Returns
    -------
    list[float]
        Single-element list ``[u_D]``.
    """
    return [float(np.clip(D, D_min, D_max))]


def u_const_012(x_vec, t):
    """Shortcut for the constant motif with u_D = 0.12 h^-1."""
    return u_const(x_vec, t, D=0.12)


def u_step_24h(x_vec, t):
    """Step motif at 24 h.

    u_D(t) = 0.12,  t < 24 h
    u_D(t) = 0.06,  t >= 24 h

    Parameters
    ----------
    x_vec : array_like
        Current state (unused).
    t : float
        Current time (h).

    Returns
    -------
    list[float]
        Single-element list ``[u_D]``.
    """
    if t < 24.0:
        D = 0.12
    else:
        D = 0.06
    return [float(np.clip(D, D_min, D_max))]


def u_sin_12h(x_vec, t):
    """Sinusoidal motif with 12 h period.

    u_D(t) = 0.15 + 0.05 sin(2π t / 12)

    Parameters
    ----------
    x_vec : array_like
        Current state (unused).
    t : float
        Current time (h).

    Returns
    -------
    list[float]
        Single-element list ``[u_D]``.
    """
    D = 0.15 + 0.05 * np.sin(2.0 * np.pi * t / 12.0)
    return [float(np.clip(D, D_min, D_max))]


def u_steps_piecewise(x_vec, t):
    """Original piecewise-constant motif used in the phase-1 simulations.

    This is kept for backward compatibility.

    Parameters
    ----------
    x_vec : array_like
        Current state (unused).
    t : float
        Current time (h).

    Returns
    -------
    list[float]
        Single-element list ``[u_D]``.
    """
    if t < 12.0:
        D = 0.05
    elif t < 24.0:
        D = 0.20
    elif t < 36.0:
        D = 0.08
    else:
        D = 0.12
    return [float(np.clip(D, D_min, D_max))]
