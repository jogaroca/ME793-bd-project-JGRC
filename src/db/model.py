"""
Continuous-time chemostat model for Batrachochytrium dendrobatidis (Bd).

This module defines the nominal parameter set and the continuous-time
dynamics for the temperature-dependent growth model in a well-mixed
chemostat. The state vector is

    x = [Z, S1, S2, S3, N, W]

where
    Z  : density of motile zoospores
    S1 : early encysted thalli
    S2 : intermediate thalli
    S3 : mature zoosporangia
    N  : nutrient concentration (mM)
    W  : waste / inhibitory by-product (a.u.)

The single manipulated input is the dilution rate u_D (h^-1), which
enters the model in a control–affine way.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Names of the six Bd states, used e.g. by PyBounds
STATE_NAMES = ["Z", "S1", "S2", "S3", "N", "W"]

# Nominal parameter set (see project report for details and units)
theta = dict(
    f0=4.0,      # S3 -> Z (zoospore release)
    m0=0.10,     # Z -> S1 (encystment)
    muZ=0.15,    # Z mortality
    muS=0.05,    # S_i mortality
    kmax=0.60,   # maximum maturation rate
    KN=0.5,      # Monod constant for nutrient N [mM]
    KW=1.0,      # inhibition constant for W [a.u.]
    hW=1.0,      # inhibition exponent due to W
    Ym=5e3,      # yield (cells per mM)
    sm=0.02,     # contribution to W from maturation
    sZ=0.01,     # contribution to W from Z death
    sS=0.005,    # contribution to W from S death
    N0=5.0,      # nutrient concentration in the influent [mM]
)


def k_m(N: float, W: float, th: dict | None = None) -> float:
    """
    Effective maturation rate k_m(N, W; theta).

    This combines Monod-type nutrient dependence with inhibition by the
    waste variable W, using the parameterization from the project report.
    """
    if th is None:
        th = theta

    # Guard against negative values due to numerical noise
    Np = max(float(N), 0.0)
    Wp = max(float(W), 0.0)

    return th["kmax"] * (Np / (th["KN"] + Np)) * (
        1.0 / (1.0 + (Wp / th["KW"]) ** th["hW"])
    )


def f(
    x_vec: Sequence[float] | np.ndarray,
    u_vec: Sequence[float] | np.ndarray,
    th: dict | None = None,
    return_state_names: bool = False,
) -> np.ndarray | list[str]:
    """
    Continuous-time dynamics x_dot = f(x, u; theta) for the Bd chemostat.

    Parameters
    ----------
    x_vec
        State vector [Z, S1, S2, S3, N, W].
    u_vec
        Input vector, where u_vec[0] = u_D is the dilution rate (h^-1).
    th
        Parameter dictionary. If None, the global ``theta`` is used.
    return_state_names
        If True, return the list of state names instead of x_dot.
        This is convenient for tools such as :mod:`pybounds`.

    Returns
    -------
    x_dot : ndarray, shape (6,)
        Time derivative of the state vector.
    """
    if return_state_names:
        # Used by planar_drone + PyBounds pattern; keeps our API compatible.
        return STATE_NAMES

    if th is None:
        th = theta

    Z, S1, S2, S3, N, W = x_vec
    uD = float(u_vec[0])

    km = k_m(N, W, th)
    Ssum = S1 + S2 + S3

    # Intrinsic dynamics f0 (no dilution term)
    f0 = np.array(
        [
            th["f0"] * km * S3 - (th["m0"] + th["muZ"]) * Z,       # Z
            th["m0"] * Z - (km + th["muS"]) * S1,                  # S1
            km * S1 - (km + th["muS"]) * S2,                       # S2
            km * S2 - (km + th["muS"]) * S3,                       # S3
            -(1.0 / th["Ym"]) * km * Ssum,                         # N
            th["sm"] * km * Ssum                                   # W (from maturation)
            + th["sZ"] * th["muZ"] * Z
            + th["sS"] * th["muS"] * Ssum,
        ]
    )

    # Dilution contribution g_D(x) * u_D
    gDu = uD * np.array([-Z, -S1, -S2, -S3, (th["N0"] - N), -W])

    return f0 + gDu


def default_initial_state(th: dict | None = None) -> np.ndarray:
    """
    Return a convenient default initial condition for the Bd model.

    By default, this matches the initial condition used in the ME793
    simulation script: a small initial biomass and nutrient N equal to N0.
    """
    if th is None:
        th = theta

    x0 = np.array([1.0, 0.2, 0.2, 0.2, th["N0"], 0.0], dtype=float)
    return x0
