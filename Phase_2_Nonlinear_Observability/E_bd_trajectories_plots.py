"""
Simple trajectory plots for the Bd chemostat:
 - Dilution motifs D(t) used in the empirical observability analysis.
 - Output trajectories B, N, W vs time for each motif.
"""

import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from Utility import bd_chemostat as bd
from Utility import bd_motifs as motifs

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
T_FINAL = 48.0
DT = 0.05

# Motifs: name and corresponding D_fun(t)
TRAJECTORY_MOTIFS: List[Tuple[str, callable]] = [
    ("const_0p12", motifs.D_const_0p12),
    ("step_24h",   motifs.D_step_24h),
    ("sin_12h",    motifs.D_sin_12h),
    ("sin_12h_big", motifs.D_sin_12h_big),
]


# ----------------------------------------------------------------------
# Plot D(t) for all motifs in a single figure
# ----------------------------------------------------------------------
def plot_dilution_motifs():
    t_grid = np.arange(0.0, T_FINAL + DT, DT)

    plt.figure()
    for name, D_fun in TRAJECTORY_MOTIFS:
        D_vals = [D_fun(t) for t in t_grid]
        plt.plot(t_grid, D_vals, label=name)

    plt.xlabel("t [h]")
    plt.ylabel("D(t) [h$^{-1}$]")
    plt.title("Dilution motifs used for empirical observability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Plot B, N, W vs time for each motif (one figure per motif)
# ----------------------------------------------------------------------
def plot_BNW_for_all_motifs():
    # Bd dynamics and measurement (B, N, W)
    f_obj = bd.F()
    h_obj = bd.H(measurement_option="h_BNW")

    for motif_name, D_fun in TRAJECTORY_MOTIFS:
        t_sim, x_sim, u_sim, y_sim = bd.simulate_bd(
            f=f_obj.f,
            h=h_obj.h,
            tsim_length=T_FINAL,
            dt=DT,
            x0=None,       # use default initial condition inside simulate_bd
            D=0.0,         # ignored because D_fun is provided
            D_fun=D_fun,
        )

        # y_sim columns are [B, N, W]
        B = y_sim[:, 0]
        N = y_sim[:, 1]
        W = y_sim[:, 2]

        plt.figure()
        plt.plot(t_sim, B, label="B (total biomass)")
        plt.plot(t_sim, N, label="N (nutrient)")
        plt.plot(t_sim, W, label="W (waste)")

        plt.xlabel("t [h]")
        plt.ylabel("Concentration [mM]")
        plt.title(f"Outputs B, N, W for motif={motif_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Plotting dilution motifs D(t)...")
    plot_dilution_motifs()

    print("Plotting B, N, W trajectories for all motifs...")
    plot_BNW_for_all_motifs()


if __name__ == "__main__":
    main()
