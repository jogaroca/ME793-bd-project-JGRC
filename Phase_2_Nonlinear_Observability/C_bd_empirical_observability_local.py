# Phase_2_Nonlinear_Observability/C_bd_empirical_observability_local.py

"""
Local empirical observability index along the Bd chemostat trajectory.

This script:
    - Simulates the Bd model for a fixed initial condition and constant D
    - Computes the local empirical observability Gramian W_k at each time step
      from finite-difference sensitivities dy(t_k)/dx0
    - Defines a local index lambda_min(W_k)
    - Plots B(t) colored by log10(lambda_min(W_k))
"""

import os
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths and imports
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILITY_DIR = os.path.join(PROJECT_ROOT, "Utility")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")

if UTILITY_DIR not in sys.path:
    sys.path.append(UTILITY_DIR)

import bd_chemostat as bd  # noqa: E402


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
T_FINAL = 48.0       # total simulation time [h]
DT = 0.05            # time step [h]
D_CONST = 0.12       # constant dilution rate [h^-1]

# Nominal initial condition [Z, S1, S2, S3, N, W]
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)

# Finite-difference perturbation sizes
EPS_REL = 1e-4
EPS_ABS = 1e-6

# Measurement option used for the local index
MEASUREMENT_OPTION = "h_BNW"  # y = [B, N, W]


# ----------------------------------------------------------------------
# Sensitivity computation
# ----------------------------------------------------------------------
def simulate_outputs(
    f_handle,
    h_handle,
    x0_vec: np.ndarray,
    t_final: float,
    dt: float,
    D_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper around bd.simulate_bd for convenience.
    Returns (t_sim, x_sim, u_sim, y_sim).
    """
    t_sim, x_sim, u_sim, y_sim = bd.simulate_bd(
        f=f_handle,
        h=h_handle,
        tsim_length=t_final,
        dt=dt,
        x0=x0_vec,
        D=D_value,
    )
    return t_sim, x_sim, u_sim, y_sim


def compute_sensitivity_trajectory(
    f_handle,
    h_handle,
    x0_vec: np.ndarray,
    t_final: float,
    dt: float,
    D_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the trajectory y(t) and the sensitivity tensor S(t) = dy/dx0
    using central finite differences on the initial state x0.

    Parameters
    ----------
    f_handle, h_handle : callables
        Dynamics and measurement functions from bd_chemostat.
    x0_vec : ndarray, shape (n_states,)
        Nominal initial condition.
    t_final : float
        Final time of the simulation.
    dt : float
        Time step.
    D_value : float
        Constant dilution rate.

    Returns
    -------
    t_sim : ndarray, shape (T,)
        Time grid.
    y_nominal : ndarray, shape (T, n_outputs)
        Nominal output trajectory.
    S : ndarray, shape (T, n_outputs, n_states)
        Sensitivity tensor, S[k, j, i] = d y_j(t_k) / d x0_i.
    """
    x0_vec = np.asarray(x0_vec, dtype=float).reshape(-1)
    n_states = x0_vec.size

    # Nominal trajectory
    t_sim, _, _, y_nominal = simulate_outputs(
        f_handle=f_handle,
        h_handle=h_handle,
        x0_vec=x0_vec,
        t_final=t_final,
        dt=dt,
        D_value=D_value,
    )

    T, n_outputs = y_nominal.shape
    S = np.zeros((T, n_outputs, n_states), dtype=float)

    for i in range(n_states):
        x_i = x0_vec[i]
        delta_i = EPS_REL * max(abs(x_i), 1.0) + EPS_ABS

        x_plus = x0_vec.copy()
        x_minus = x0_vec.copy()
        x_plus[i] = x_i + delta_i
        x_minus[i] = x_i - delta_i

        _, _, _, y_plus = simulate_outputs(
            f_handle=f_handle,
            h_handle=h_handle,
            x0_vec=x_plus,
            t_final=t_final,
            dt=dt,
            D_value=D_value,
        )
        _, _, _, y_minus = simulate_outputs(
            f_handle=f_handle,
            h_handle=h_handle,
            x0_vec=x_minus,
            t_final=t_final,
            dt=dt,
            D_value=D_value,
        )

        dy = (y_plus - y_minus) / (2.0 * delta_i)  # shape (T, n_outputs)
        S[:, :, i] = dy

    return t_sim, y_nominal, S


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    # Build dynamics and measurement functions
    f_obj = bd.F()
    f = f_obj.f

    h_obj = bd.H(measurement_option=MEASUREMENT_OPTION)
    h = h_obj.h

    # Sensitivity trajectory
    t_sim, y_nominal, S = compute_sensitivity_trajectory(
        f_handle=f,
        h_handle=h,
        x0_vec=X0_NOMINAL,
        t_final=T_FINAL,
        dt=DT,
        D_value=D_CONST,
    )

    T, n_outputs = y_nominal.shape
    _, _, n_states = S.shape

    # Local observability Gramian and index at each time step
    lambda_min_traj = np.zeros(T, dtype=float)

    for k in range(T):
        # S_k: (n_outputs, n_states)
        S_k = S[k, :, :]  # dy/dx0 at time t_k
        W_k = DT * (S_k.T @ S_k)  # local Gramian

        # eigenvalues of W_k
        eigvals_k = np.linalg.eigvalsh(W_k)
        lambda_min_traj[k] = np.min(eigvals_k)

    # Save trajectory data
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"local_obs_{MEASUREMENT_OPTION}.npz")
    np.savez(
        out_path,
        t_sim=t_sim,
        y_nominal=y_nominal,
        lambda_min_traj=lambda_min_traj,
    )
    print(f"Saved local observability trajectory to: {out_path}")

    # Plot B(t) colored by log10(lambda_min)
    B_traj = y_nominal[:, 0]  # first output is B in h_BNW
    log_lambda_min = np.log10(np.maximum(lambda_min_traj, 1e-30))

    fig, ax = plt.subplots()
    sc = ax.scatter(t_sim, B_traj, c=log_lambda_min, s=15)
    ax.set_xlabel("t [h]")
    ax.set_ylabel("B (total biomass)")
    ax.set_title(
        f"Local empirical observability index\n"
        f"Measurement option: {MEASUREMENT_OPTION}"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(lambda_min(W_k))")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
