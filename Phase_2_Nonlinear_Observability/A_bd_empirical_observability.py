"""
Phase 2: Empirical nonlinear observability analysis for the Bd chemostat model.

This script:
    - Uses the Bd model defined in Utility/bd_chemostat.py
    - Evaluates several trajectory motifs (time-varying dilution inputs)
    - For each motif and measurement configuration, computes the empirical
      observability Gramian and scalar metrics.
    - Additionally computes a Chernoff-style inverse of W to obtain
      minimum error variances per state (in particular for Z).
    - Saves the results to results/phase2_empirical/obs_{motif}_{meas}.npz
"""

import os
from typing import Dict, List

import numpy as np

from Utility import bd_chemostat as bd
from Utility import empirical_observability as eo
from Utility import bd_motifs as motifs

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
T_FINAL = 48.0       # total simulation time [h]
DT = 0.05            # time step [h]

# Nominal initial condition [Z, S1, S2, S3, N, W]
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)

# Trajectory motifs to explore
TRAJECTORY_MOTIFS: List[Dict[str, object]] = [
    {"name": "const_0p12", "x0": X0_NOMINAL, "D_fun": motifs.D_const_0p12},
    {"name": "step_24h",   "x0": X0_NOMINAL, "D_fun": motifs.D_step_24h},
    {"name": "sin_12h",    "x0": X0_NOMINAL, "D_fun": motifs.D_sin_12h},
    {"name": "sin_12h_big", "x0": X0_NOMINAL, "D_fun": motifs.D_sin_12h_big},
]

# Finite-difference perturbation sizes
EPS_REL = 1e-4
EPS_ABS = 1e-6

# Threshold for Chernoff-style inverse eigenvalues
CHERNOFF_TOL = 1e-10

# Measurement options to compare (must exist as methods in bd.H)
MEASUREMENT_OPTIONS = [
    "h_BNW",  # y = [B, N, W]
    "h_BN",   # y = [B, N]
    "h_B",    # y = [B]
    "h_NW",   # y = [N, W]
    "h_N",    # y = [N]
    "h_W",    # y = [W]
]


# ----------------------------------------------------------------------
# Helper: closure that returns y(t) given x0, for a given motif
# ----------------------------------------------------------------------
def build_simulate_outputs_fn(
    f_handle,
    h_handle,
    t_final: float,
    dt: float,
    D_fun,
):
    """
    Returns a closure simulate_outputs_fn(x0) that simulates the Bd model
    with a fixed motif D_fun(t), final time and time step, and returns
    only the output trajectory y(t).
    """

    def simulate_outputs_fn(x0_vec: np.ndarray) -> np.ndarray:
        _, _, _, y_sim = bd.simulate_bd(
            f=f_handle,
            h=h_handle,
            tsim_length=t_final,
            dt=dt,
            x0=x0_vec,
            D=0.0,       # ignored because D_fun is provided
            D_fun=D_fun,
        )
        return y_sim

    return simulate_outputs_fn


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    # Dynamics
    f_obj = bd.F()
    f = f_obj.f

    results_root = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")
    os.makedirs(results_root, exist_ok=True)

    for motif in TRAJECTORY_MOTIFS:
        motif_name = motif["name"]
        x0_val = motif["x0"]
        D_fun = motif["D_fun"]

        print("=" * 80)
        print(f"Trajectory motif: {motif_name}")

        for meas_opt in MEASUREMENT_OPTIONS:
            print("-" * 80)
            print(f"Measurement option: {meas_opt}")

            # Measurement
            h_obj = bd.H(measurement_option=meas_opt)
            h = h_obj.h

            # Closure that simulates y(t) given x0
            simulate_outputs_fn = build_simulate_outputs_fn(
                f_handle=f,
                h_handle=h,
                t_final=T_FINAL,
                dt=DT,
                D_fun=D_fun,
            )

            # Empirical observability matrix and Gramian
            J, W = eo.empirical_observability_matrix(
                simulate_outputs_fn=simulate_outputs_fn,
                x0=x0_val,
                dt=DT,
                eps_rel=EPS_REL,
                eps_abs=EPS_ABS,
            )

            metrics = eo.empirical_observability_metrics(W)

            lambda_min = metrics["lambda_min"]
            lambda_max = metrics["lambda_max"]
            cond_number = metrics["condition_number"]
            trace_W = metrics["trace"]
            det_W = metrics["determinant"]
            eigvals_sorted = metrics["eigenvalues"]

            print(f"lambda_min        = {lambda_min:.3e}")
            print(f"lambda_max        = {lambda_max:.3e}")
            print(f"condition_number  = {cond_number:.3e}")
            print(f"trace(W)          = {trace_W:.3e}")
            print(f"det(W)            = {det_W:.3e}")

            # ------------------------------------------------------------------
            # Chernoff-style inverse: minimum error variance per state
            # ------------------------------------------------------------------
            # eigen-decomposition of W
            eigvals, eigvecs = np.linalg.eigh(W)

            inv_eigs = np.zeros_like(eigvals)
            mask = eigvals > CHERNOFF_TOL
            inv_eigs[mask] = 1.0 / eigvals[mask]

            # C_chernoff = V diag(inv_eigs) V^T
            C_chernoff = (eigvecs * inv_eigs) @ eigvecs.T
            chernoff_diag = np.diag(C_chernoff)

            # States are ordered [Z, S1, S2, S3, N, W] in bd.F
            var_Z = float(chernoff_diag[0])     # minimum error variance for Z
            I_Z = float(np.sqrt(W[0, 0]))       # sqrt of W_ZZ, information index

            print(f"var_Z (Chernoff)  = {var_Z:.3e}")
            print(f"I_Z = sqrt(W_ZZ)  = {I_Z:.3e}")

            # Save results
            out_path = os.path.join(
                results_root, f"obs_{motif_name}_{meas_opt}.npz"
            )
            np.savez(
                out_path,
                W=W,
                eigenvalues=eigvals_sorted,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                trace_val=trace_W,
                determinant=det_W,
                condition_number=cond_number,
                chernoff_diag=chernoff_diag,
                var_Z=var_Z,
                I_Z=I_Z,
            )
            print(
                f"Saved results for motif={motif_name}, "
                f"meas={meas_opt} to: {out_path}"
            )


if __name__ == "__main__":
    main()
