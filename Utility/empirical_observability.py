"""
Utility functions to compute empirical observability matrices
for nonlinear dynamical systems.

The approach is based on finite differences with respect to the
initial state x0. For each state component x_i we perturb x0_i
by +/- delta_i, simulate the output trajectories, and approximate
the sensitivity dy/dx_i. These sensitivities are stacked into a
Jacobian matrix J, and the empirical observability Gramian is
W = dt * J^T J.
"""

from typing import Callable, Sequence, Tuple, Dict
import numpy as np


def empirical_observability_matrix(
    simulate_outputs_fn: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    dt: float = 1.0,
    eps_rel: float = 1e-4,
    eps_abs: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical observability matrix J and Gramian W.

    Parameters
    ----------
    simulate_outputs_fn : callable
        Function that receives x0 (1D ndarray, shape (n_states,))
        and returns the output trajectory y(t) as an array of shape (T, n_outputs).
        The time grid and inputs are assumed to be fixed inside this function.
    x0 : array_like, shape (n_states,)
        Nominal initial state about which observability is evaluated.
    dt : float, optional
        Sampling time step used in the simulation; used as a weight to
        approximate the integral in the observability Gramian.
    eps_rel : float, optional
        Relative perturbation size for each state.
    eps_abs : float, optional
        Absolute perturbation size (added to the relative one).

    Returns
    -------
    J : ndarray, shape (T * n_outputs, n_states)
        Empirical observability matrix (stacked sensitivities dy/dx0).
    W : ndarray, shape (n_states, n_states)
        Empirical observability Gramian, W = dt * J^T J.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    n_states = x0.size

    # Simulate once at nominal state to get output dimensions
    y_nominal = np.asarray(simulate_outputs_fn(x0), dtype=float)
    if y_nominal.ndim != 2:
        raise ValueError("simulate_outputs_fn must return a 2D array of shape (T, n_outputs).")

    n_time, n_outputs = y_nominal.shape
    n_rows = n_time * n_outputs

    J = np.zeros((n_rows, n_states), dtype=float)

    for i in range(n_states):
        x_i = x0[i]
        delta_i = eps_rel * max(abs(x_i), 1.0) + eps_abs

        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[i] = x_i + delta_i
        x_minus[i] = x_i - delta_i

        y_plus = np.asarray(simulate_outputs_fn(x_plus), dtype=float)
        y_minus = np.asarray(simulate_outputs_fn(x_minus), dtype=float)

        if y_plus.shape != y_nominal.shape or y_minus.shape != y_nominal.shape:
            raise ValueError(
                "simulate_outputs_fn must return arrays of consistent shape (T, n_outputs)."
            )

        # Finite-difference approximation of dy/dx_i
        dy = (y_plus - y_minus) / (2.0 * delta_i)  # shape (T, n_outputs)

        # Stack the sensitivities in time and outputs into a single column
        J[:, i] = dy.reshape(-1)

    # Empirical observability Gramian
    W = dt * (J.T @ J)

    return J, W


def empirical_observability_metrics(W: np.ndarray) -> Dict[str, object]:
    """
    Compute standard scalar metrics from an empirical observability Gramian.

    Parameters
    ----------
    W : ndarray, shape (n_states, n_states)
        Empirical observability Gramian (symmetric positive semi-definite).

    Returns
    -------
    metrics : dict
        Dictionary with:
            - 'eigenvalues': sorted eigenvalues (ascending).
            - 'lambda_min': smallest eigenvalue.
            - 'lambda_max': largest eigenvalue.
            - 'trace': trace of W.
            - 'determinant': determinant of W.
            - 'condition_number': lambda_max / lambda_min (np.inf if lambda_min <= 0).
    """
    W = np.asarray(W, dtype=float)
    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix.")

    # For symmetric matrices, eigvalsh is more stable
    eigvals = np.linalg.eigvalsh(W)
    eigvals_sorted = np.sort(eigvals)

    lambda_min = float(eigvals_sorted[0])
    lambda_max = float(eigvals_sorted[-1])
    trace_val = float(np.trace(W))
    det_val = float(np.linalg.det(W))

    if lambda_min > 0.0:
        cond = float(lambda_max / lambda_min)
    else:
        cond = float("inf")

    return {
        "eigenvalues": eigvals_sorted,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "trace": trace_val,
        "determinant": det_val,
        "condition_number": cond,
    }
