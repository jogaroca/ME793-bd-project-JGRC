"""
Input (control) motifs for the Bd chemostat.

All functions here have the signature

    u_func(x_vec, t) -> [u_D]

where u_D is the dilution rate (h^{-1}).
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence
import numpy as np

# Reasonable bounds for the dilution rate u_D (h^{-1})
D_MIN: float = 0.0
D_MAX: float = 0.5


def _clip_D(D: float) -> float:
    """Clip the dilution rate to the allowed interval [D_MIN, D_MAX]."""
    return float(np.clip(D, D_MIN, D_MAX))


def u_const_012(x_vec: Sequence[float], t: float):
    """
    Constant dilution rate: u_D(t) = 0.12.
    """
    D = 0.12
    return [_clip_D(D)]


def u_step_24h(x_vec: Sequence[float], t: float):
    """
    Step change in dilution rate at t = 24 h:

        u_D(t) = 0.12,  t < 24 h
               = 0.06,  t >= 24 h
    """
    if t < 24.0:
        D = 0.12
    else:
        D = 0.06
    return [_clip_D(D)]


def u_sin_12h(x_vec: Sequence[float], t: float):
    """
    Sinusoidal dilution rate with period 12 h:

        u_D(t) = 0.15 + 0.05 * sin(2*pi*t/12)
    """
    D = 0.15 + 0.05 * np.sin(2.0 * np.pi * t / 12.0)
    return [_clip_D(D)]


# Convenience dictionary to loop over motifs
MOTIFS: Dict[str, Callable[[Sequence[float], float], Sequence[float]]] = {
    "const_0.12": u_const_012,
    "step_24h": u_step_24h,
    "sin_12h": u_sin_12h,
}


# Backwards-compatible aliases (optional, for old scripts)
def u_func_const(x_vec: Sequence[float], t: float, D: float = 0.12):
    """Compatibility wrapper for the original constant-input function."""
    return [_clip_D(D)]


def u_func_steps(x_vec: Sequence[float], t: float):
    """Compatibility wrapper for the original step-input function."""
    return u_step_24h(x_vec, t)


__all__ = [
    "D_MIN",
    "D_MAX",
    "u_const_012",
    "u_step_24h",
    "u_sin_12h",
    "MOTIFS",
    "u_func_const",
    "u_func_steps",
]
