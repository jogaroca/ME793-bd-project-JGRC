# Phase_2_Nonlinear_Observability/A_bd_empirical_observability.py

"""
Phase 2: Empirical nonlinear observability analysis for the Bd chemostat model.

This script:
    - Uses the Bd model defined in Utility/bd_chemostat.py
    - Computes the empirical observability Gramian for a given trajectory
      (fixed initial condition and constant dilution rate D)
    - Compares several measurement configurations (different outputs)
    - Saves the Gramian and metrics to results/phase2_empirical
"""

import os
import sys
from typing import Dict

import numpy as np

# ----------------------------------------------------------------------
# Add Utility directory to the Python path
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILITY_DIR = os.path.join(PROJECT_ROOT, "Utility")

if UTILITY_DIR not in sys.path:
    sys.path.append(UTILITY_DIR)

import bd_chemostat as bd  # noqa: E402
import empirical_observability as eo  # noqa: E402


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
# Helper to build a simulate_outputs_fn for a given measurement option
# ----------------------------------------------------------------------
def build_simulate_outputs_fn(
    f_handle,
    h_handle,
    t_final: float,
    dt: float,
    D_value: float,
):
    """
    Returns a closure simulate_outputs_fn(x0) that simulates the Bd model
    from initial condition x0, with fixed D, t_final and dt, and returns
    only the output trajectory y(t).
    """

    def simulate_outputs_fn(x0_vec: np.ndarray) -> np.ndarray:
        _, _, _, y_sim = bd.simulate_bd(
            f=f_handle,
            h=h_handle,
            tsim_length=t_final,
            dt=dt,
            x0=x0_vec,
            D=D_value,
        )
        return y_sim

    return simulate_outputs_fn


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def main():
    # Build dynamics function f(x, u)
    f_obj = bd.F()
    f = f_obj.f

    results: Dict[str, Dict[str, object]] = {}

    for meas_opt in MEASUREMENT_OPTIONS:
        print("=" * 80)
        print(f"Measurement option: {meas_opt}")

        # Build measurement function h(x, u)
        h_obj = bd.H(measurement_option=meas_opt)
        h = h_obj.h

        # Closure that simulates y(t) given x0
        simulate_outputs_fn = build_simulate_outputs_fn(
            f_handle=f,
            h_handle=h,
            t_final=T_FINAL,
            dt=DT,
            D_value=D_CONST,
        )

        # Compute empirical observability matrix and Gramian
        J, W = eo.empirical_observability_matrix(
            simulate_outputs_fn=simulate_outputs_fn,
            x0=X0_NOMINAL,
            dt=DT,
            eps_rel=EPS_REL,
            eps_abs=EPS_ABS,
        )

        metrics = eo.empirical_observability_metrics(W)

        print(f"lambda_min        = {metrics['lambda_min']:.3e}")
        print(f"lambda_max        = {metrics['lambda_max']:.3e}")
        print(f"condition_number  = {metrics['condition_number']:.3e}")
        print(f"trace(W)          = {metrics['trace']:.3e}")
        print(f"det(W)            = {metrics['determinant']:.3e}")
        print(f"eigenvalues (sorted, log10): "
              f"{np.log10(np.maximum(metrics['eigenvalues'], 1e-30))}")

        results[meas_opt] = {
            "J": J,
            "W": W,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_dir = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")
    os.makedirs(results_dir, exist_ok=True)

    for meas_opt, data in results.items():
        W = data["W"]
        metrics = data["metrics"]

        out_path = os.path.join(results_dir, f"obs_constD_{meas_opt}.npz")
        np.savez(
            out_path,
            W=W,
            eigenvalues=metrics["eigenvalues"],
            lambda_min=metrics["lambda_min"],
            lambda_max=metrics["lambda_max"],
            trace_val=metrics["trace"],
            determinant=metrics["determinant"],
            condition_number=metrics["condition_number"],
        )
        print(f"Saved results for {meas_opt} to: {out_path}")


if __name__ == "__main__":
    main()
