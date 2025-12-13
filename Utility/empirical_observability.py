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

from typing import Callable, Sequence, Tuple, Dict, Optional, List
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


# ----------------------------------------------------------------------
# Sliding-window Fisher information and minimum error variance
# ----------------------------------------------------------------------

def empirical_output_sensitivities(
    simulate_outputs_fn: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    eps_rel: float = 1e-4,
    eps_abs: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute output sensitivities S(t) = dy(t)/dx0 via finite differences.

    Parameters
    ----------
    simulate_outputs_fn : callable
        Receives x0 (shape (n_states,)) and returns y(t) (shape (T, n_outputs)).
        The time grid and inputs are assumed fixed inside this function.
    x0 : array_like, shape (n_states,)
        Nominal initial state.
    eps_rel, eps_abs : float
        Relative and absolute perturbation sizes.

    Returns
    -------
    y_nominal : ndarray, shape (T, n_outputs)
        Nominal output trajectory.
    S : ndarray, shape (T, n_outputs, n_states)
        Sensitivity tensor where S[k, :, i] is dy(t_k)/dx0_i.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    n_states = x0.size

    y_nominal = np.asarray(simulate_outputs_fn(x0), dtype=float)
    if y_nominal.ndim != 2:
        raise ValueError("simulate_outputs_fn must return a 2D array of shape (T, n_outputs).")

    T, n_outputs = y_nominal.shape
    S = np.zeros((T, n_outputs, n_states), dtype=float)

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

        dy = (y_plus - y_minus) / (2.0 * delta_i)  # (T, n_outputs)
        S[:, :, i] = dy

    return y_nominal, S


def _build_precision_weights(
    measurement_noise_vars: Optional[Sequence[float]],
    n_outputs: int,
) -> np.ndarray:
    """Return per-output precision weights (1/variance)."""
    if measurement_noise_vars is None:
        return np.ones(n_outputs, dtype=float)

    var = np.asarray(measurement_noise_vars, dtype=float).reshape(-1)
    if var.size == 1:
        if float(var[0]) <= 0.0:
            raise ValueError("measurement_noise_vars must be strictly positive.")
        return np.ones(n_outputs, dtype=float) / float(var[0])

    if var.size != n_outputs:
        raise ValueError(
            f"measurement_noise_vars must have length {n_outputs} (or be scalar); got {var.size}."
        )

    if np.any(var <= 0.0):
        raise ValueError("measurement_noise_vars must be strictly positive.")

    return 1.0 / var


def sliding_fisher_information(
    S: np.ndarray,
    dt: float,
    window_steps: int,
    step: int = 1,
    measurement_noise_vars: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute Fisher information matrices on sliding windows.

    The (empirical) Fisher information for a window is:

        F = dt * sum_{k in window} H_k^T R^{-1} H_k

    where H_k = dy(t_k)/dx0 (shape n_outputs x n_states) and
    R is the output noise covariance (assumed diagonal here).

    Parameters
    ----------
    S : ndarray, shape (T, n_outputs, n_states)
        Sensitivity tensor S[k, :, i] = dy(t_k)/dx0_i.
    dt : float
        Time step [same units as the simulation].
    window_steps : int
        Window length in samples (must be >= 1).
    step : int
        Sliding step in samples (must be >= 1).
    measurement_noise_vars : array_like or scalar, optional
        Output noise variances. If provided, used as R = diag(vars) and
        Fisher uses R^{-1}. If None, R = I is assumed.

    Returns
    -------
    t_mid : ndarray, shape (n_windows,)
        Time stamps aligned to the center of each window (in units of dt).
    F_list : list of ndarray
        Fisher information matrices, each shape (n_states, n_states).
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 3:
        raise ValueError("S must be a 3D array of shape (T, n_outputs, n_states).")

    T, n_outputs, n_states = S.shape
    if window_steps < 1 or window_steps > T:
        raise ValueError(f"window_steps must be in [1, {T}].")
    if step < 1:
        raise ValueError("step must be >= 1.")

    weights = _build_precision_weights(measurement_noise_vars, n_outputs)  # precision (1/var)
    sqrt_w = np.sqrt(weights).reshape(-1, 1)  # (n_outputs, 1)

    starts = np.arange(0, T - window_steps + 1, step, dtype=int)
    t_mid = (starts + 0.5 * (window_steps - 1)) * float(dt)

    F_list: List[np.ndarray] = []
    for s0 in starts:
        F = np.zeros((n_states, n_states), dtype=float)
        for k in range(s0, s0 + window_steps):
            Hk = S[k, :, :]  # (n_outputs, n_states)
            Hk_w = sqrt_w * Hk  # weight rows
            F += Hk_w.T @ Hk_w
        F_list.append(float(dt) * F)

    return t_mid, F_list


def chernoff_inverse_psd(
    A: np.ndarray,
    lam: float = 1e-8,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chernoff-style (regularized) inverse for symmetric PSD matrices.

    We form A_tilde = A + lam*I and compute its eigen-decomposition. Any
    eigenvalue below tol * max_eig is treated as numerically zero.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric PSD matrix.
    lam : float
        Ridge regularization parameter (>= 0).
    tol : float
        Relative eigenvalue threshold (>= 0).

    Returns
    -------
    A_inv : ndarray, shape (n, n)
        Regularized pseudo-inverse.
    eigvals : ndarray, shape (n,)
        Eigenvalues of A_tilde (ascending).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    n = A.shape[0]
    if lam < 0.0:
        raise ValueError("lam must be >= 0.")
    if tol < 0.0:
        raise ValueError("tol must be >= 0.")

    A_tilde = A + float(lam) * np.eye(n)
    eigvals, eigvecs = np.linalg.eigh(A_tilde)

    max_eig = float(np.max(eigvals)) if eigvals.size > 0 else 0.0
    thresh = float(tol) * max_eig

    inv_eigs = np.zeros_like(eigvals)
    mask = eigvals > thresh
    inv_eigs[mask] = 1.0 / eigvals[mask]

    A_inv = (eigvecs * inv_eigs) @ eigvecs.T
    return A_inv, np.sort(eigvals)


def minimum_error_variance_timeseries(
    S: np.ndarray,
    dt: float,
    window_steps: int,
    step: int = 1,
    measurement_noise_vars: Optional[Sequence[float]] = None,
    lam: float = 1e-8,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the time series of minimum error variance (MEV) for each state.

    For each sliding window we compute the Fisher information F and a
    regularized inverse \tilde{F}^{-1}. The per-state minimum error
    variance is diag(\tilde{F}^{-1}).

    Parameters
    ----------
    S : ndarray, shape (T, n_outputs, n_states)
        Output sensitivity tensor.
    dt : float
        Time step.
    window_steps : int
        Window length in samples.
    step : int
        Window stride in samples.
    measurement_noise_vars : array_like or scalar, optional
        Output noise variances (diagonal R). If None, R=I.
    lam : float
        Ridge parameter for the inverse.
    tol : float
        Relative eigenvalue threshold for pseudo-inversion.

    Returns
    -------
    t_mid : ndarray, shape (n_windows,)
        Time aligned to the center of each window.
    mev : ndarray, shape (n_windows, n_states)
        Minimum error variance per state over time (Cramer-Rao lower bound).
    """
    t_mid, F_list = sliding_fisher_information(
        S=S,
        dt=dt,
        window_steps=window_steps,
        step=step,
        measurement_noise_vars=measurement_noise_vars,
    )

    if not F_list:
        raise RuntimeError("No windows were produced. Check window_steps and step.")

    n_states = F_list[0].shape[0]
    mev = np.zeros((len(F_list), n_states), dtype=float)

    for i, F in enumerate(F_list):
        F_inv, _ = chernoff_inverse_psd(F, lam=lam, tol=tol)
        mev[i, :] = np.diag(F_inv)

    return t_mid, mev
