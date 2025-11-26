# Phase_3_Nonlinear_Estimation/A_bd_generate_synthetic_data.py
"""
Phase 3: Synthetic data generation for nonlinear estimator design.

- Re-uses the Bd chemostat model in Utility/bd_chemostat.py
- Simulates the system for the three dilution motifs (const, step, sinusoidal)
- Saves the *true* state trajectories and D(t) to:
    results/phase3_estimation/truth_motif_<name>.npz

These files will be used later to:
    - generate noisy measurements for different measurement sets
    - run the nonlinear estimators (EKF, etc.)
"""

import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths and imports
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILITY_DIR = os.path.join(PROJECT_ROOT, "Utility")

if UTILITY_DIR not in sys.path:
    sys.path.append(UTILITY_DIR)

import bd_chemostat as bd        # noqa: E402
import bd_motifs as motifs       # noqa: E402


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
T_FINAL = 48.0       # total simulation time [h]
DT = 0.05            # time step [h]

# Nominal initial condition [Z, S1, S2, S3, N, W]
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)

# Trajectory motifs to explore (same as in Phase 2)
TRAJECTORY_MOTIFS: List[Dict[str, object]] = [
    {"name": "const_0p12", "x0": X0_NOMINAL, "D_fun": motifs.D_const_0p12},
    {"name": "step_24h",   "x0": X0_NOMINAL, "D_fun": motifs.D_step_24h},
    {"name": "sin_12h",    "x0": X0_NOMINAL, "D_fun": motifs.D_sin_12h},
]


# ----------------------------------------------------------------------
# Dummy measurement function (we don't need y(t) here)
# ----------------------------------------------------------------------
def dummy_h(*args, **kwargs):
    """
    Dummy measurement function for simulate_bd.

    It accepts any arguments (*args, **kwargs) and simply returns a
    fixed 3-element vector. This keeps simulate_bd happy, but we will
    ignore the output trajectory y(t) and compute B, N, W ourselves
    from the state.
    """
    return np.zeros(3, dtype=float)


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    # Dynamics object (used internally by simulate_bd)
    f_obj = bd.F()
    f = f_obj.f

    # Where to store synthetic "truth" data
    results_root = os.path.join(PROJECT_ROOT, "results", "phase3_estimation")
    os.makedirs(results_root, exist_ok=True)

    for motif in TRAJECTORY_MOTIFS:
        motif_name = motif["name"]
        x0_val = motif["x0"]
        D_fun = motif["D_fun"]

        print("=" * 80)
        print(f"Simulating motif: {motif_name}")

        # Simulate Bd chemostat with the chosen dilution motif
        t_vec, x_traj, u_traj, _ = bd.simulate_bd(
            f=f,
            h=dummy_h,           # we ignore y(t)
            tsim_length=T_FINAL,
            dt=DT,
            x0=x0_val,
            D=0.0,               # ignored because D_fun is provided
            D_fun=D_fun,
        )

        # Unpack states: x = [Z, S1, S2, S3, N, W]
        Z = x_traj[:, 0]
        S1 = x_traj[:, 1]
        S2 = x_traj[:, 2]
        S3 = x_traj[:, 3]
        N = x_traj[:, 4]
        W = x_traj[:, 5]

        # Total biomass proxy B = Z + S1 + S2 + S3
        B = Z + S1 + S2 + S3

        # Save "truth" data for later use in estimation scripts
        out_path = os.path.join(results_root, f"truth_motif_{motif_name}.npz")
        np.savez(
            out_path,
            t=t_vec,
            x_true=x_traj,
            u_D=u_traj,
            B=B,
            N=N,
            W=W,
            motif_name=motif_name,
            dt=DT,
            T_final=T_FINAL,
            x0=X0_NOMINAL,
        )
        print(f"Saved synthetic truth for motif={motif_name} to: {out_path}")

        # Optional quick sanity-check plots (can be commented out if desired)
        plt.figure(figsize=(6, 4))
        plt.plot(t_vec, B, label="B (total biomass)")
        plt.plot(t_vec, N, label="N (nutrient)")
        plt.plot(t_vec, W, label="W (waste)")
        plt.xlabel("t [h]")
        plt.ylabel("States")
        plt.title(f"Bd trajectories for motif: {motif_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\nAll motifs simulated and saved.")


if __name__ == "__main__":
    main()
