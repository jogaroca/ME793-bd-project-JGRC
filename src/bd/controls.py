"""
Control input motifs for the Bd chemostat.

The single manipulated input is the dilution rate u_D(t) [h^-1].  This
module collects several time-dependent motifs that will be used in the
nonlinear observability analysis and later optimal-control studies.

We also provide small helper functions to clip the dilution rate to a
reasonable range and to wrap scalar u_D(t) functions into the
(x, t) -> u_vec interface expected by :func:`scipy.integrate.odeint`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Reasonable bounds for the dilution rate (per hour)
D_MIN: float = 0.0
D_MAX: float = 0.5


def _clip_dilution(D: float) -> float:
    """Clip the dilution rate into the admissible range [D_MIN, D_MAX]."""
    return float(np.clip(D, D_MIN, D_MAX))


# ---------------------------------------------------------------------------
# Simple u_D(t) motifs to be used in NOA
# ---------------------------------------------------------------------------

def uD_const_012(t: float) -> float:
    """
    Constant dilution motif

        u_D(t) = 0.12  [h^-1]
    """
    return _clip_dilution(0.12)


def uD_step_24h(t: float) -> float:
    """
    Step-change motif with a switch at 24 h

        u_D(t) = 0.12,  t < 24 h
               = 0.06,  t >= 24 h
    """
    if t < 24.0:
        D = 0.12
    else:
        D = 0.06
    return _clip_dilution(D)


def uD_sin_12h(t: float) -> float:
    """
    Sinusoidal motif with 12 h period

        u_D(t) = 0.15 + 0.05 sin( 2π t / 12 )
    """
    D = 0.15 + 0.05 * np.sin(2.0 * np.pi * t / 12.0)
    return _clip_dilution(D)


def uD_demo_steps(t: float) -> float:
    """
    Piecewise-constant demo motif similar to the one used in the
    original simulation script.  This is handy for quick testing.

    It cycles through several dilution levels over 48 hours.
    """
    if t < 12.0:
        D = 0.05
    elif t < 24.0:
        D = 0.20
    elif t < 36.0:
        D = 0.08
    else:
        D = 0.12
    return _clip_dilution(D)


# ---------------------------------------------------------------------------
# Helpers to interface with odeint-style dynamics
# ---------------------------------------------------------------------------

def make_u_func(uD_fun: Callable[[float], float]) -> Callable:
    """
    Wrap a scalar-valued u_D(t) function into a state-dependent input
    function with signature ``u_func(x_vec, t) -> [u_D]``.

    This is the interface expected in the existing ME793 simulation
    script and in :func:`scipy.integrate.odeint`.
    """

    def u_func(x_vec, tsim):
        return [float(uD_fun(float(tsim)))]

    return u_func


def u_func_const(x_vec, tsim, D: float = 0.12):
    """
    Backwards-compatible helper used in the original script.

    It is equivalent to::

        make_u_func(lambda t: D)(x_vec, tsim)
    """
    return [float(_clip_dilution(D))]


def u_func_steps(x_vec, tsim):
    """
    Backwards-compatible helper implementing the original piecewise-
    constant steps used in the exploratory simulations.
    """
    return [float(_clip_dilution(uD_demo_steps(float(tsim))))]
