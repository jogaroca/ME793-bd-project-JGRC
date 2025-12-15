"""
Input (dilution) control motifs for the Bd chemostat.

Motifs:
    - const_0p12 :      u_D(t) = 0.12
    - step_24h :        u_D(t) = 0.12 for t < 24 h, 0.06 for t >= 24 h
    - sin_12h :         u_D(t) = 0.15 + 0.05 * sin(2π t / 12 h)
    - sin_12h_big :     u_D(t) = 0.15 + 0.10 * sin(2π t / 12 h)
"""

from typing import Callable, Dict
import numpy as np


def D_const_0p12(t: float) -> float:
    """u_D(t) = 0.12"""
    return 0.12


def D_step_24h(t: float) -> float:
    """u_D(t) = 0.12 for t < 24 h, 0.06 for t >= 24 h."""
    return 0.12 if t < 24.0 else 0.06


def D_sin_12h(t: float) -> float:
    """u_D(t) = 0.15 + 0.05 * sin(2π t / 12 h)."""
    return 0.15 + 0.05 * np.sin(2.0 * np.pi * t / 12.0)
    
def D_sin_12h_big(t: float) -> float:
    """u_D(t) = 0.15 + 0.10 * sin(2π t / 12 h)."""
    return 0.15 + 0.10 * np.sin(2.0 * np.pi * t / 12.0)


MOTIFS: Dict[str, Callable[[float], float]] = {
    "const_0p12": D_const_0p12,
    "step_24h": D_step_24h,
    "sin_12h": D_sin_12h,
    "sin_12h_big": D_sin_12h_big,
}
