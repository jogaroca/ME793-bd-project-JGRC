"""
Empirical nonlinear observability tools for the Bd chemostat model.

This module provides thin, user-friendly wrappers around the core
:mod:`pybounds` classes used in the ME793 course repository, namely

    * :class:`pybounds.SlidingEmpiricalObservabilityMatrix`
    * :class:`pybounds.FisherObservability`
    * :class:`pybounds.SlidingFisherObservability`

The idea is that you first simulate an open-loop trajectory of the Bd
system (states X, inputs U, outputs Y) using :func:`bd.simulator.simulate_bd_odeint`
and then use the functions in this module to

1. Construct empirical observability matrices in sliding windows, and
2. Convert them into Fisher-information-based metrics of observability
   for each state, taking the measurement noise into account.

These wrappers are intentionally lightweight: they do not hide pybounds,
but simply make the typical workflow for the Bd project clearer and
less error-prone.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .model import STATE_NAMES
from .measurements import MEASUREMENT_NAMES


def estimate_measurement_noise_variances_from_outputs(
    Y_true: np.ndarray,
    rel_std_B: float = 0.05,
    abs_std_N: float = 0.05,
    rel_std_W: float = 0.05,
    measurement_names: Sequence[str] = MEASUREMENT_NAMES,
) -> Dict[str, float]:
    """
    Estimate measurement-noise variances for each output channel.

    This function mirrors the noise model used in :func:`bd.measurements.meas_setA`,
    but instead of returning noisy measurements it returns *variances*
    that are consistent with the assumed noise levels.  These variances
    are packaged into a dictionary ``{sensor_name: variance}``, which is
    the format expected by :mod:`pybounds`.

    Parameters
    ----------
    Y_true
        Array of noise-free outputs with shape (T, 3) and columns
        corresponding to [B, N, W].
    rel_std_B
        Relative standard deviation (fraction of signal) for B.
    abs_std_N
        Absolute standard deviation (mM) for N.
    rel_std_W
        Relative standard deviation (fraction of signal) for W.
    measurement_names
        Names of the measurement channels. By default this is
        :data:`bd.measurements.MEASUREMENT_NAMES`.

    Returns
    -------
    noise_vars : dict
        Dictionary mapping measurement name -> variance.
    """
    Y_true = np.asarray(Y_true)
    if Y_true.shape[1] != 3:
        raise ValueError(
            f"Y_true must have shape (T, 3) with columns [B, N, W], "
            f"but got shape {Y_true.shape!r}."
        )

    B_true = Y_true[:, 0]
    N_true = Y_true[:, 1]
    W_true = Y_true[:, 2]

    sigmaB = rel_std_B * np.maximum(B_true, 0.0)
    sigmaN = abs_std_N * np.ones_like(N_true)
    sigmaW = rel_std_W * np.maximum(W_true, 0.0)

    var_B = float(np.mean(sigmaB**2))
    var_N = float(np.mean(sigmaN**2))
    var_W = float(np.mean(sigmaW**2))

    name_to_var = dict(zip(measurement_names, [var_B, var_N, var_W]))
    return name_to_var


def _ensure_state_dataframe(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    """
    Convert a state trajectory into a DataFrame with proper column names.
    """
    if isinstance(X, pd.DataFrame):
        # Assume the user set appropriate column names already.
        return X.copy()

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(STATE_NAMES):
        raise ValueError(
            f"State trajectory X must have shape (T, {len(STATE_NAMES)}), "
            f"but got {X.shape!r}."
        )

    return pd.DataFrame(X, columns=STATE_NAMES)


def _ensure_input_dataframe(U: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    """
    Convert an input trajectory into a DataFrame with a single column 'u_D'.
    """
    if isinstance(U, pd.DataFrame):
        return U.copy()

    U = np.asarray(U).reshape(len(U), -1)
    if U.shape[1] != 1:
        raise ValueError(
            f"Input trajectory U must have shape (T, 1), but got {U.shape!r}."
        )

    return pd.DataFrame(U, columns=["u_D"])


def build_sliding_empirical_observability_matrix(
    simulator,
    t_sim: np.ndarray,
    X: np.ndarray | pd.DataFrame,
    U: np.ndarray | pd.DataFrame,
    window_size: Optional[int] = None,
    eps: float = 1e-4,
):
    """
    Construct the sliding-window empirical observability matrix for Bd.

    This is a convenience wrapper around
    :class:`pybounds.SlidingEmpiricalObservabilityMatrix`, matching the
    usage from ``Lesson_8/B_empirical_nonlinear_observability_pybounds.ipynb``:

    .. code-block:: python

        SEOM = pybounds.SlidingEmpiricalObservabilityMatrix(
            simulator, t_sim, x_sim, u_sim, w=window_size, eps=1e-4
        )
        O_sliding = SEOM.get_observability_matrix()

    Parameters
    ----------
    simulator
        A :class:`pybounds.Simulator` instance configured for the Bd
        dynamics and measurement model.
    t_sim
        1D array of time stamps (length T).
    X
        State trajectory, either as ndarray shape (T, 6) or as a
        DataFrame.  If an ndarray is provided, it is converted into a
        DataFrame with columns :data:`bd.model.STATE_NAMES`.
    U
        Input trajectory, either as ndarray shape (T, 1) or a DataFrame.
        If an ndarray is provided, it is converted into a DataFrame with
        column ``'u_D'``.
    window_size
        Sliding-window size ``w`` passed to pybounds.  If None, pybounds
        uses the entire time-series as a single window (no sliding).
    eps
        Small perturbation size used by pybounds for finite-difference
        approximations of the observability matrix.

    Returns
    -------
    SEOM
        The :class:`pybounds.SlidingEmpiricalObservabilityMatrix`
        instance (contains additional attributes such as ``O_df_sliding``
        and ``t_sim``).
    O_sliding
        The structure returned by ``SEOM.get_observability_matrix()``;
        typically a list of DataFrames, one per time window.
    """
    try:
        import pybounds  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pybounds is required to build empirical observability matrices.\n"
            "Install it with `pip install pybounds`."
        ) from exc

    t_sim = np.asarray(t_sim)
    x_df = _ensure_state_dataframe(X)
    u_df = _ensure_input_dataframe(U)

    SEOM = pybounds.SlidingEmpiricalObservabilityMatrix(
        simulator, t_sim, x_df, u_df, w=window_size, eps=eps
    )

    O_sliding = SEOM.get_observability_matrix()
    return SEOM, O_sliding


def build_single_window_fisher(
    O_window,
    measurement_noise_variances: Dict[str, float],
    lam: float = 1e-8,
):
    """
    Construct a Fisher-information object for a *single* observability matrix.

    This is a thin wrapper around :class:`pybounds.FisherObservability`,
    matching the usage in the ME793 Lesson 8 notebook:

    .. code-block:: python

        FO = pybounds.FisherObservability(O_single_window,
                                          measurement_noise_vars,
                                          lam=1e-8)

    Parameters
    ----------
    O_window
        One empirical observability matrix (typically a pandas DataFrame)
        for a single time window.
    measurement_noise_variances
        Dictionary mapping each measurement name to its variance.
    lam
        Small regularization constant added on the diagonal before
        matrix inversion.

    Returns
    -------
    FO
        The :class:`pybounds.FisherObservability` instance.  The minimum
        error variance per state is available via ``FO.error_variance``.
    """
    try:
        import pybounds  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pybounds is required to construct Fisher observability objects.\n"
            "Install it with `pip install pybounds`."
        ) from exc

    FO = pybounds.FisherObservability(O_window, measurement_noise_variances, lam=lam)
    return FO


def build_sliding_fisher_observability(
    SEOM,
    measurement_noise_variances: Dict[str, float],
    states: Optional[Sequence[str]] = None,
    sensors: Optional[Sequence[str]] = None,
    time_steps: Optional[Sequence[int]] = None,
    lam: float = 1e-8,
    window_size: Optional[int] = None,
):
    """
    Construct a sliding-window Fisher observability object and return
    the aligned minimum error variances for each state.

    This wrapper mirrors the pattern used in the ME793 Lesson 8
    notebook (planar drone example):

    .. code-block:: python

        SFO = pybounds.SlidingFisherObservability(
            SEOM.O_df_sliding,
            time=SEOM.t_sim,
            lam=1e-8,
            R=o_measurement_noise_vars,
            states=o_states,
            sensors=o_sensors,
            time_steps=o_time_steps,
            w=None
        )
        EV_aligned = SFO.get_minimum_error_variance()
        EV_no_nan = EV_aligned.fillna(method='bfill').fillna(method='ffill')

    Parameters
    ----------
    SEOM
        The :class:`pybounds.SlidingEmpiricalObservabilityMatrix`
        instance returned by :func:`build_sliding_empirical_observability_matrix`.
    measurement_noise_variances
        Dictionary mapping measurement name -> variance.
    states
        Optional subset of state names to include in the analysis.
        If None, pybounds includes all states.
    sensors
        Optional subset of sensor names to include.  If None, pybounds
        includes all sensors.
    time_steps
        Optional array of relative time-step indices to include within
        each window, e.g. ``np.arange(0, w, 1)``.
    lam
        Regularization constant for the Fisher information matrices.
    window_size
        Optional override of the window size ``w`` used in the Fisher
        analysis.  If None, pybounds uses the window size from ``SEOM``.

    Returns
    -------
    SFO
        The :class:`pybounds.SlidingFisherObservability` instance.
    EV_no_nan : pandas.DataFrame
        DataFrame with the aligned minimum error variance for each
        state, with NaNs forward/backward filled for convenience.  The
        time vector can be obtained from ``SFO.time``.
    """
    try:
        import pybounds  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pybounds is required to construct sliding Fisher observability objects.\n"
            "Install it with `pip install pybounds`."
        ) from exc

    kwargs = dict(
        time=SEOM.t_sim,
        lam=lam,
        R=measurement_noise_variances,
    )

    if states is not None:
        kwargs["states"] = list(states)
    if sensors is not None:
        kwargs["sensors"] = list(sensors)
    if time_steps is not None:
        kwargs["time_steps"] = np.asarray(time_steps)
    if window_size is not None:
        kwargs["w"] = int(window_size)

    SFO = pybounds.SlidingFisherObservability(SEOM.O_df_sliding, **kwargs)

    EV_aligned = SFO.get_minimum_error_variance()
    EV_no_nan = EV_aligned.fillna(method="bfill").fillna(method="ffill")

    return SFO, EV_no_nan
