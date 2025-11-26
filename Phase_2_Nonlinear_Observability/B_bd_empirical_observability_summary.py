"""
Summarize and plot empirical observability metrics for the Bd chemostat.

This script:
    - Loads the .npz files produced by A_bd_empirical_observability.py
      for several trajectory motifs.
    - For each motif, collects lambda_min and condition numbers for all
      measurement sets and generates bar plots.
"""

import os
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths and configuration
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")

MOTIF_NAMES: List[str] = ["const_0p12", "step_24h", "sin_12h"]

MEASUREMENT_OPTIONS: List[str] = [
    "h_BNW",
    "h_BN",
    "h_B",
    "h_NW",
    "h_N",
    "h_W",
]


def load_metrics(motif_name: str, meas_opt: str) -> Dict[str, float]:
    """
    Load observability metrics for a given motif and measurement option.
    """
    filename = f"obs_{motif_name}_{meas_opt}.npz"
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found for motif='{motif_name}', "
            f"measurement='{meas_opt}': {path}\n"
            "Run A_bd_empirical_observability.py first."
        )

    data = np.load(path)
    metrics = {
        "lambda_min": float(data["lambda_min"]),
        "lambda_max": float(data["lambda_max"]),
        "condition_number": float(data["condition_number"]),
        "trace": float(data["trace_val"]),
        "determinant": float(data["determinant"]),
    }
    return metrics


def main():
    print(f"Loading results from: {RESULTS_DIR}")

    for motif_name in MOTIF_NAMES:
        print("=" * 80)
        print(f"Trajectory motif: {motif_name}")

        lambda_min_list = []
        cond_list = []

        for meas_opt in MEASUREMENT_OPTIONS:
            metrics = load_metrics(motif_name, meas_opt)
            lambda_min_list.append(metrics["lambda_min"])
            cond_list.append(metrics["condition_number"])

            print("------------------------------------------------------------------")
            print(f"Measurement option: {meas_opt}")
            print(f"  lambda_min       = {metrics['lambda_min']:.3e}")
            print(f"  condition_number = {metrics['condition_number']:.3e}")
            print(f"  trace(W)         = {metrics['trace']:.3e}")
            print(f"  det(W)           = {metrics['determinant']:.3e}")

        lambda_min_arr = np.array(lambda_min_list)
        cond_arr = np.array(cond_list)

        log_lambda_min = np.log10(np.maximum(lambda_min_arr, 1e-30))
        log_cond = np.log10(np.maximum(cond_arr, 1.0))

        x = np.arange(len(MEASUREMENT_OPTIONS))

        # Plot: log10(lambda_min)
        fig1, ax1 = plt.subplots()
        ax1.bar(x, log_lambda_min)
        ax1.set_xticks(x)
        ax1.set_xticklabels(MEASUREMENT_OPTIONS, rotation=45, ha="right")
        ax1.set_ylabel("log10(lambda_min)")
        ax1.set_title(
            f"Empirical observability (motif={motif_name}): smallest eigenvalue"
        )
        fig1.tight_layout()

        # Plot: log10(condition number)
        fig2, ax2 = plt.subplots()
        ax2.bar(x, log_cond)
        ax2.set_xticks(x)
        ax2.set_xticklabels(MEASUREMENT_OPTIONS, rotation=45, ha="right")
        ax2.set_ylabel("log10(condition number)")
        ax2.set_title(
            f"Empirical observability (motif={motif_name}): condition number"
        )
        fig2.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
