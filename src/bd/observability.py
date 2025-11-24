"""Simulation and empirical observability tools for the Bd chemostat."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from .dynamics import f, f_ode, theta
from .measurements import h


def simulate_bd(u_func, x0, tsim, params: dict | None = None, return_arrays: bool = False):
    """Simulate the Bd chemostat for a given control law.

    Parameters
    ----------
    u_func : callable
        Function ``u_func(x_vec, t) -> [u_D]`` describing the dilution rate.
    x0 : array_like, shape (6,)
        Initial condition for the state.
    tsim : array_like
        Time vector (hours) at which the state is sampled.
    params : dict, optional
        Parameter dictionary. If None, uses :data:`bd.dynamics.theta`.
    return_arrays : bool, optional
        If True, returns plain NumPy arrays; otherwise returns pandas DataFrames.

    Returns
    -------
    t_sim : np.ndarray, shape (T,)
        Time vector (copy of ``tsim``).
    X : np.ndarray or pandas.DataFrame, shape (T, 6)
        State trajectory.
    U : np.ndarray or pandas.DataFrame, shape (T, 1)
        Input trajectory u_D(t).
    Y : np.ndarray or pandas.DataFrame, shape (T, 3)
        Noise-free measurements y(t) = [B, N, W].
    """
    if params is None:
        params = theta

    tsim = np.asarray(tsim, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    # Integrate the ODE system using odeint and the wrapper f_ode
    result = odeint(f_ode, x0, tsim, args=(u_func, f))

    # Build the control and measurement traces along the trajectory
    u_trace = np.array([u_func(result[i, :], tsim[i])[0] for i in range(len(tsim))])
    Y = np.array([h(result[i, :], [u_trace[i]]) for i in range(len(tsim))])

    if return_arrays:
        return tsim.copy(), result, u_trace, Y

    import pandas as pd

    Xdf = pd.DataFrame(result, columns=['Z', 'S1', 'S2', 'S3', 'N', 'W'])
    Udf = pd.DataFrame(u_trace, columns=['u_D'])
    Ydf = pd.DataFrame(Y, columns=['B', 'N', 'W'])

    return tsim.copy(), Xdf, Udf, Ydf


def empirical_observability_gramian(
    u_func,
    x0,
    tsim,
    eps: float = 1e-3,
    lam: float = 1e-6,
    params: dict | None = None,
    meas_indices: list[int] | None = None,
):
    """Compute the empirical observability Gramian for a given input motif.

    For each state direction e_i, the initial condition is perturbed as
    x0 ± eps * e_i, the system is re-simulated, and a sensitivity matrix
    O is built from the differences in the output trajectories.

    Parameters
    ----------
    u_func : callable
        Control law ``u_func(x_vec, t) -> [u_D]``.
    x0 : array_like, shape (6,)
        Nominal initial condition.
    tsim : array_like
        Time vector (hours).
    eps : float, optional
        Magnitude of the symmetric perturbation in each state direction.
    lam : float, optional
        Tikhonov regularization parameter for the inverse of the Gramian.
    params : dict, optional
        Parameter dictionary. If None, uses :data:`bd.dynamics.theta`.
    meas_indices : list[int] or None, optional
        Indices of the measurement components to use in the Gramian
        (0 -> B, 1 -> N, 2 -> W). If None, all measurements are used.

    Returns
    -------
    W_o : np.ndarray, shape (n_states, n_states)
        Empirical observability Gramian.
    Finv : np.ndarray, shape (n_states, n_states)
        Regularized inverse, interpreted as a minimum error variance matrix.
    O : np.ndarray, shape (T * n_meas, n_states)
        Sensitivity matrix whose columns correspond to perturbed states.
    """
    if params is None:
        params = theta

    tsim = np.asarray(tsim, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    # 1) Nominal simulation (used mainly to determine dimensions)
    _, X_nom, _, Y_nom = simulate_bd(
        u_func, x0, tsim, params=params, return_arrays=True
    )
    n_steps, n_states = X_nom.shape

    if meas_indices is None:
        meas_indices = list(range(Y_nom.shape[1]))
    else:
        meas_indices = list(meas_indices)

    n_meas = len(meas_indices)

    # 2) Build sensitivity matrix O using only selected measurements
    O_cols = []
    for i in range(n_states):
        e = np.zeros(n_states)
        e[i] = eps

        x0_plus = x0 + e
        x0_minus = x0 - e

        _, _, _, Y_plus = simulate_bd(
            u_func, x0_plus, tsim, params=params, return_arrays=True
        )
        _, _, _, Y_minus = simulate_bd(
            u_func, x0_minus, tsim, params=params, return_arrays=True
        )

        Y_plus_sel = Y_plus[:, meas_indices]
        Y_minus_sel = Y_minus[:, meas_indices]

        dY = (Y_plus_sel - Y_minus_sel).ravel()  # (n_steps * n_meas,)
        O_cols.append(dY)

    O = np.stack(O_cols, axis=1)  # (n_steps * n_meas, n_states)

    # 3) Empirical observability Gramian and regularized inverse
    W_o = O.T @ O
    Finv = np.linalg.inv(W_o + lam * np.eye(n_states))

    return W_o, Finv, O

