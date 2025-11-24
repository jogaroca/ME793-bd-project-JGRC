"""
Bd chemostat model definitions.

State vector:
    x = [Z, S1, S2, S3, N, W]
where
    Z  : free zoospores
    S1 : early sporangia
    S2 : intermediate sporangia
    S3 : mature sporangia
    N  : nutrient (mM)
    W  : waste / inhibitor (a.u.)
"""

from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

# Default parameter set for the Bd chemostat model
theta: Mapping[str, float] = dict(
    f0=4.0,      # S3 -> Z (zoospore release)
    m0=0.10,     # Z -> S1
    muZ=0.15,    # Z mortality
    muS=0.05,    # S_i mortality (i = 1,2,3)
    kmax=0.60,   # maximum maturation rate
    KN=0.5,      # half-saturation constant for N (mM)
    KW=1.0,      # inhibition constant for W (a.u.)
    hW=1.0,      # inhibition exponent due to W
    Ym=5e3,      # yield (cells per mM)
    sm=0.02,     # contribution to W from maturation
    sZ=0.01,     # contribution to W from Z death
    sS=0.005,    # contribution to W from S death
    N0=5.0,      # nutrient concentration in the influent (mM)
)

STATE_NAMES = ["Z", "S1", "S2", "S3", "N", "W"]


def k_m(N: float, W: float, th: Mapping[str, float] = theta) -> float:
    """
    Maturation rate as a function of nutrient N and waste W.
    """
    Np = max(float(N), 0.0)
    Wp = max(float(W), 0.0)
    return th["kmax"] * (Np / (th["KN"] + Np)) * (1.0 / (1.0 + (Wp / th["KW"]) ** th["hW"]))


def f(x_vec: Sequence[float], u_vec: Sequence[float], th: Mapping[str, float] = theta) -> np.ndarray:
    """
    Continuous-time dynamics for the Bd chemostat.

    The system is control-affine:
        x_dot = f0(x) + g_D(x) * u_D,
    where u_D is the dilution rate (chemostat outflow).
    """
    Z, S1, S2, S3, N, W = x_vec
    uD = float(u_vec[0])

    km = k_m(N, W, th)
    Ssum = S1 + S2 + S3

    # f0(x): dynamics without dilution
    f0 = np.array(
        [
            th["f0"] * km * S3 - (th["m0"] + th["muZ"]) * Z,
            th["m0"] * Z - (km + th["muS"]) * S1,
            km * S1 - (km + th["muS"]) * S2,
            km * S2 - (km + th["muS"]) * S3,
            -(1.0 / th["Ym"]) * km * Ssum,
            th["sm"] * km * Ssum + th["sZ"] * th["muZ"] * Z + th["sS"] * th["muS"] * Ssum,
        ]
    )

    # g_D(x) * u_D term (dilution and inflow)
    gDu = uD * np.array([-Z, -S1, -S2, -S3, (th["N0"] - N), -W])
    return f0 + gDu


def default_initial_state(th: Mapping[str, float] = theta) -> np.ndarray:
    """
    Convenience function returning the default initial condition used in the project.
    """
    return np.array([1.0, 0.2, 0.2, 0.2, th["N0"], 0.0], dtype=float)


__all__ = ["theta", "STATE_NAMES", "k_m", "f", "default_initial_state"]
