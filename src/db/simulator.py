"""
Simulation utilities for the Bd chemostat model.

This module provides a lightweight wrapper around :func:`scipy.integrate.odeint`
to integrate the Bd dynamics in open loop, using one of the dilution-rate
motifs defined in :mod:`bd.controls`.

The goal is to mirror the structure of the ME793 class repository
(`planar_drone.simulate_drone`) while keeping the implementation as simple
and transparent as possible.  A separate helper is also provided to create
a PyBounds ``Simulator`` object for later use in empirical observability
analysis, but this helper is optional and only works if :mod:`pybounds`
is installed in the Python environment.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import odeint

from .model import f, theta, STATE_NAMES, default_initial_state
from .controls import make_u_func
from .measurements import h


def default_time_grid(T: float = 48.0, dt: float = 0.01) -> np.ndarray:
    """
    Construct a uniform time grid from 0 to ``T`` (inclusive) with step ``dt``.
    """
    return np.arange(0.0, T + dt, step=dt)


def f_ode(x_vec, tsim, u_func: Callable, f_handle: Callable):
    """
    Thin wrapper with the signature expected by :func:`odeint`.

    Parameters
    ----------
    x_vec
        Current state vector.
    tsim
        Current time (float).
    u_func
        Callable ``u_func(x_vec, tsim) -> sequence`` returning the current
        input vector.
    f_handle
        Dynamics function ``f(x_vec, u_vec, **kwargs)``.
    """
    u_vec = u_func(x_vec, tsim)
    return f_handle(x_vec, u_vec)


def simulate_bd_odeint(
    x0: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    uD_fun: Callable[[float], float] | None = None,
    theta_local: dict | None = None,
    noisy: bool = False,
    rng_seed: int | None = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the Bd chemostat dynamics in open loop using ``odeint``.

    Parameters
    ----------
    x0
        Initial state vector. If None, :func:`default_initial_state` is used.
    t_grid
        1D array of time points. If None, ``default_time_grid()`` is used.
    uD_fun
        Scalar function ``u_D(t)`` providing the dilution rate in h^-1.
        If None, a constant motif with u_D = 0.12 h^-1 is used.
    theta_local
        Optional parameter dictionary. If None, the global :data:`theta`
        from :mod:`bd.model` is used.
    noisy
        If True, synthetic measurement noise is added via
        :func:`bd.measurements.meas_setA`.
    rng_seed
        Random seed for the measurement noise (if ``noisy`` is True).

    Returns
    -------
    t_sim : ndarray, shape (T,)
        Time vector.
    X : ndarray, shape (T, 6)
        State trajectory.
    U : ndarray, shape (T, 1)
        Input trajectory u_D(t).
    Y : ndarray, shape (T, 3)
        Output trajectory [B, N, W].  If ``noisy`` is True this includes
        noise; otherwise it is noise-free.
    """
    from .measurements import meas_setA  # local import to avoid cycles

    if x0 is None:
        x0 = default_initial_state(theta_local)
    if t_grid is None:
        t_grid = default_time_grid()
    if uD_fun is None:
        # Default to the constant motif used in the report
        from .controls import uD_const_012

        uD_fun = uD_const_012
    if theta_local is None:
        theta_local = theta

    # Wrap u_D(t) into a function of (x, t) for odeint
    u_func = make_u_func(uD_fun)

    # Integrate the dynamics
    result = odeint(
        f_ode,
        x0,
        t_grid,
        args=(u_func, lambda x, u: f(x, u, th=theta_local)),
    )

    # Build input and output traces
    u_trace = np.array(
        [u_func(result[i, :], float(t_grid[i]))[0] for i in range(len(t_grid))]
    )
    X = result
    U = u_trace.reshape(-1, 1)

    Y_true = np.vstack([h(x, [u]) for x, u in zip(X, u_trace)])

    if noisy:
        Y, _ = meas_setA(X, rng_seed=rng_seed)
    else:
        Y = Y_true

    return t_grid, X, U, Y


# ---------------------------------------------------------------------------
# Optional helper to build a PyBounds simulator
# ---------------------------------------------------------------------------

def make_pybounds_simulator(
    dt: float = 0.1,
    measurement_names=None,
):
    """
    Create and return a :class:`pybounds.Simulator` object for the Bd model.

    This helper mirrors the pattern used in ``Utility/planar_drone.py`` of
    the class repository.  It is **optional** and will raise an informative
    :class:`ImportError` if :mod:`pybounds` is not installed.

    Parameters
    ----------
    dt
        Integration time step in hours.
    measurement_names
        Optional list of measurement names. If None, they are inferred from
        :func:`bd.measurements.h`.

    Returns
    -------
    simulator : pybounds.Simulator
        Configured simulator instance for the Bd dynamics.
    """
    try:
        import pybounds  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pybounds is not installed. Install it to use make_pybounds_simulator()."
        ) from exc

    if measurement_names is None:
        measurement_names = h(None, None, return_measurement_names=True)

    simulator = pybounds.Simulator(
        f,
        h,
        dt=dt,
        state_names=STATE_NAMES,
        input_names=["u_D"],
        measurement_names=measurement_names,
    )

    return simulator
