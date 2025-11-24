"""
Simulation utilities for the Bd chemostat model.

These functions provide a clean interface around ``scipy.integrate.odeint``
and are designed to be reused from notebooks and from empirical
observability analyses (e.g., with pybounds).
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple
import numpy as np
from scipy.integrate import odeint

from .bd_model import f
from .bd_measurements import h

ControlFunction = Callable[[Sequence[float], float], Sequence[float]]


def f_ode(x_vec: Sequence[float], t: float, u_func: ControlFunction, f_handle=f) -> np.ndarray:
    """
    Wrapper with the signature expected by :func:`scipy.integrate.odeint`.
    """
    u_vec = u_func(x_vec, t)
    return f_handle(x_vec, u_vec)


def simulate_bd(
    x0: Sequence[float],
    t_grid: np.ndarray,
    u_func: ControlFunction,
    f_handle=f,
    return_measurements: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the Bd chemostat model in open loop using ``odeint``.

    Parameters
    ----------
    x0 : sequence of float, shape (6,)
        Initial state.
    t_grid : ndarray, shape (T,)
        Time grid (hours) for the integration.
    u_func : callable
        Input function u_func(x, t) -> [u_D].
    f_handle : callable, optional
        Continuous-time dynamics function f(x, u, theta). Defaults to ``bd_model.f``.
    return_measurements : bool, optional
        If True, also compute noise-free measurements y(t) using ``bd_measurements.h``.

    Returns
    -------
    t_grid : ndarray, shape (T,)
        The same time grid passed in.
    X : ndarray, shape (T, 6)
        State trajectory.
    Y : ndarray, shape (T, 3)
        Measurement trajectory [B, N, W]. If ``return_measurements=False``,
        Y will be ``None``.
    """
    X = odeint(f_ode, x0, t_grid, args=(u_func, f_handle))

    if return_measurements:
        Y = np.array([h(x, u_func(x, t)) for x, t in zip(X, t_grid)])
    else:
        Y = None

    return t_grid, X, Y


def compute_effluent_productivity(
    t_grid: np.ndarray, X: np.ndarray, u_func: ControlFunction
) -> np.ndarray:
    """
    Compute the production rate in the effluent,

        P_out(t) = u_D(t) * Z(t),

    where Z is the first component of the state vector.
    """
    u_trace = np.array([float(u_func(x, t)[0]) for x, t in zip(X, t_grid)])
    Z = X[:, 0]
    return u_trace * Z


__all__ = ["ControlFunction", "f_ode", "simulate_bd", "compute_effluent_productivity"]
