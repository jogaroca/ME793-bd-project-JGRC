"""
Local empirical observability index along Bd chemostat trajectories.

This script:
    - Simulates the Bd model for a fixed initial condition and several
      trajectory motifs u_D(t)
    - For each motif, computes the local empirical observability Gramian W_k
      at each time step from finite-difference sensitivities dy(t_k)/dx0
    - Defines a local index as the smallest positive eigenvalue of W_k
    - For each motif, saves the trajectory and plots B(t) colored by
      log10 of that local index
"""

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from Utility import bd_chemostat as bd
from Utility import bd_motifs as motifs

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
T_FINAL = 48.0       # total simulation time [h]
DT = 0.05            # time step [h]

# Nominal initial condition [Z, S1, S2, S3, N, W]
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)

# Measurement option used for the local index
MEASUREMENT_OPTION = "h_BNW"  # y = [B, N, W]

# Trajectory motifs to analyze (keys of motifs.MOTIFS)
MOTIF_NAMES = ["const_0p12", "step_24h", "sin_12h"]

# Finite-difference perturbations
EPS_REL = 1e-4
EPS_ABS = 1e-6

# Threshold to decide which eigenvalues are "effectively zero"
EIGEN_TOL = 1e-10


# ----------------------------------------------------------------------
# Sensitivity computation
# ----------------------------------------------------------------------
def simulate_outputs(
    f_handle,
    h_handle,
    x0_vec: np.ndarray,
    t_final: float,
    dt: float,
    D_fun,
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
        D=0.0,        # ignored because D_fun is provided
        D_fun=D_fun,
    )
    return t_sim, x_sim, u_sim, y_sim


def compute_sensitivity_trajectory(
    f_handle,
    h_handle,
    x0_vec: np.ndarray,
    t_final: float,
    dt: float,
    D_fun,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the trajectory y(t) and the sensitivity tensor S(t) = dy/dx0
    using central finite differences on the initial state x0.
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
        D_fun=D_fun,
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
            D_fun=D_fun,
        )
        _, _, _, y_minus = simulate_outputs(
            f_handle=f_handle,
            h_handle=h_handle,
            x0_vec=x_minus,
            t_final=t_final,
            dt=dt,
            D_fun=D_fun,
        )

        dy = (y_plus - y_minus) / (2.0 * delta_i)  # shape (T, n_outputs)
        S[:, :, i] = dy

    return t_sim, y_nominal, S


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    # Dynamics and measurements
    f_obj = bd.F()
    f = f_obj.f

    h_obj = bd.H(measurement_option=MEASUREMENT_OPTION)
    h = h_obj.h

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for motif_name in MOTIF_NAMES:
        print("=" * 80)
        print(f"Trajectory motif: {motif_name}")

        D_fun = motifs.MOTIFS[motif_name]

        # Sensitivity trajectory for this motif
        t_sim, y_nominal, S = compute_sensitivity_trajectory(
            f_handle=f,
            h_handle=h,
            x0_vec=X0_NOMINAL,
            t_final=T_FINAL,
            dt=DT,
            D_fun=D_fun,
        )

        T, n_outputs = y_nominal.shape
        _, _, n_states = S.shape

        # Local observability index: smallest *positive* eigenvalue of W_k
        lambda_min_pos_traj = np.zeros(T, dtype=float)

        for k in range(T):
            S_k = S[k, :, :]  # (n_outputs, n_states)
            W_k = DT * (S_k.T @ S_k)  # (n_states, n_states)

            eigvals_k = np.linalg.eigvalsh(W_k)
            pos_mask = eigvals_k > EIGEN_TOL
            if np.any(pos_mask):
                lambda_min_pos_traj[k] = np.min(eigvals_k[pos_mask])
            else:
                lambda_min_pos_traj[k] = 0.0

        # Save trajectory data
        out_path = os.path.join(
            RESULTS_DIR, f"local_obs_{motif_name}_{MEASUREMENT_OPTION}.npz"
        )
        np.savez(
            out_path,
            t_sim=t_sim,
            y_nominal=y_nominal,
            lambda_min_pos_traj=lambda_min_pos_traj,
        )
        print(f"Saved local observability trajectory to: {out_path}")

        # Plot B(t) colored by log10 of the local index
        B_traj = y_nominal[:, 0]  # first output is B in h_BNW
        log_index = np.log10(np.maximum(lambda_min_pos_traj, 1e-30))

        finite_idx = np.isfinite(log_index)
        if np.any(finite_idx):
            vmin, vmax = np.percentile(log_index[finite_idx], [5, 95])
        else:
            vmin, vmax = -10.0, 0.0

        fig, ax = plt.subplots()
        sc = ax.scatter(t_sim, B_traj, c=log_index, s=15, vmin=vmin, vmax=vmax)
        ax.set_xlabel("t [h]")
        ax.set_ylabel("B (total biomass)")
        ax.set_title(
            f"Local empirical observability index\n"
            f"Measurement: {MEASUREMENT_OPTION}, motif: {motif_name}"
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("log10(smallest positive eigenvalue of W_k)")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()