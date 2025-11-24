"""Bd chemostat dynamics module.

States: x = [Z, S1, S2, S3, N, W]
Control: u_vec = [u_D]  (dilution rate)
"""

from __future__ import annotations

import numpy as np

# ===========================
# Model parameters (Bd chemostat)
# ===========================
theta = dict(
    f0=4.0,      # S3 -> Z (zoospore release)
    m0=0.10,     # Z -> S1
    muZ=0.15,    # Z mortality
    muS=0.05,    # S_i mortality
    kmax=0.60,   # maximum maturation rate
    KN=0.5,      # mM
    KW=1.0,      # a.u.
    hW=1.0,      # inhibition exponent due to W
    Ym=5e3,      # yield (cells per mM)
    sm=0.02,     # contribution to W from maturation
    sZ=0.01,     # contribution to W from Z death
    sS=0.005,    # contribution to W from S death
    N0=5.0       # mM in the influent
)

# Reasonable bounds for the dilution rate u_D (h^-1)
D_min, D_max = 0.0, 0.5


def k_m(N: float, W: float, th: dict | None = None) -> float:
    """Maturation rate k_m(N, W).

    Parameters
    ----------
    N : float
        Nitrogen concentration (mM).
    W : float
        Waste / inhibitor variable (a.u.).
    th : dict, optional
        Parameter dictionary. If None, uses the global ``theta``.

    Returns
    -------
    float
        Maturation rate k_m (1/h).
    """
    if th is None:
        th = theta

    Np = max(float(N), 0.0)
    Wp = max(float(W), 0.0)
    return th['kmax'] * (Np / (th['KN'] + Np)) * (1.0 / (1.0 + (Wp / th['KW']) ** th['hW']))


def f(x_vec: np.ndarray, u_vec: np.ndarray, th: dict | None = None) -> np.ndarray:
    """Continuous-time Bd dynamics.

    Implements the control-affine model

        x_dot = f0(x) + g_D(x) * u_D,

    where the state is x = [Z, S1, S2, S3, N, W].

    Parameters
    ----------
    x_vec : array_like, shape (6,)
        State vector.
    u_vec : array_like, shape (1,)
        Control vector, here u_vec[0] = u_D (dilution rate, 1/h).
    th : dict, optional
        Parameter dictionary. If None, uses the global ``theta``.

    Returns
    -------
    np.ndarray, shape (6,)
        Time derivative x_dot.
    """
    if th is None:
        th = theta

    Z, S1, S2, S3, N, W = x_vec
    uD = float(u_vec[0])

    km = k_m(N, W, th)
    Ssum = S1 + S2 + S3

    # f0 term (dynamics without dilution)
    f0_vec = np.array([
        th['f0'] * km * S3 - (th['m0'] + th['muZ']) * Z,
        th['m0'] * Z - (km + th['muS']) * S1,
        km * S1 - (km + th['muS']) * S2,
        km * S2 - (km + th['muS']) * S3,
        -(1.0 / th['Ym']) * km * Ssum,
        th['sm'] * km * Ssum + th['sZ'] * th['muZ'] * Z + th['sS'] * th['muS'] * Ssum,
    ])

    # g_D(x) * u_D term (chemostat dilution)
    gDu = uD * np.array([-Z, -S1, -S2, -S3, (th['N0'] - N), -W])

    return f0_vec + gDu


def f_ode(x_vec: np.ndarray, tsim: float, u_func, f_handle) -> np.ndarray:
    """Wrapper for :func:`scipy.integrate.odeint`.

    The signature matches ``odeint`` expectations and forwards to
    the Bd dynamics function ``f``.

    Parameters
    ----------
    x_vec : array_like, shape (6,)
        Current state.
    tsim : float
        Current time (h).
    u_func : callable
        Function ``u_func(x_vec, tsim) -> [u_D]`` giving the control value.
    f_handle : callable
        Function implementing the dynamics, typically :func:`f`.

    Returns
    -------
    np.ndarray, shape (6,)
        State derivative x_dot.
    """
    u_vec = u_func(x_vec, tsim)
    return f_handle(x_vec, u_vec)


def default_initial_condition(th: dict | None = None) -> np.ndarray:
    """Return a reasonable default initial condition for Bd.

    Parameters
    ----------
    th : dict, optional
        Parameter dictionary. If None, uses the global ``theta``.

    Returns
    -------
    np.ndarray, shape (6,)
        Initial state x0 = [Z0, S10, S20, S30, N0, W0].
    """
    if th is None:
        th = theta

    return np.array([1.0, 0.2, 0.2, 0.2, th['N0'], 0.0])
