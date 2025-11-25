# Phase_2_Nonlinear_Observability/B_bd_empirical_observability_summary.py

"""
Summarize and plot empirical observability metrics for the Bd chemostat.

This script:
    - Loads the .npz files produced by A_bd_empirical_observability.py
    - Collects lambda_min and condition numbers for each measurement set
    - Generates bar plots for comparison
"""

import os
import sys
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths and configuration
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")

MEASUREMENT_OPTIONS: List[str] = [
    "h_BNW",
    "h_BN",
    "h_B",
    "h_NW",
    "h_N",
    "h_W",
]


def load_metrics_for_option(meas_opt: str) -> Dict[str, float]:
    """
    Load observability metrics from the corresponding .npz file.
    """
    filename = f"obs_constD_{meas_opt}.npz"
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found for measurement option '{meas_opt}': {path}\n"
            "Make sure you have run A_bd_empirical_observability.py first."
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
    lambda_min_list = []
    cond_list = []

    print(f"Loading results from: {RESULTS_DIR}")
    for meas_opt in MEASUREMENT_OPTIONS:
        metrics = load_metrics_for_option(meas_opt)
        lambda_min_list.append(metrics["lambda_min"])
        cond_list.append(metrics["condition_number"])

        print("------------------------------------------------------------------")
        print(f"Measurement option: {meas_opt}")
        print(f"  lambda_min       = {metrics['lambda_min']:.3e}")
        print(f"  condition_number = {metrics['condition_number']:.3e}")
        print(f"  trace(W)         = {metrics['trace']:.3e}")
        print(f"  det(W)           = {metrics['determinant']:.3e}")

    # Convert to numpy arrays
    lambda_min_arr = np.array(lambda_min_list)
    cond_arr = np.array(cond_list)

    # Use log10 for better visualization (values span many orders of magnitude)
    log_lambda_min = np.log10(np.maximum(lambda_min_arr, 1e-30))
    log_cond = np.log10(np.maximum(cond_arr, 1.0))

    x = np.arange(len(MEASUREMENT_OPTIONS))

    # ------------------------------------------------------------------
    # Plot: log10(lambda_min)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    ax1.bar(x, log_lambda_min)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MEASUREMENT_OPTIONS, rotation=45, ha="right")
    ax1.set_ylabel("log10(lambda_min)")
    ax1.set_title("Empirical observability: smallest eigenvalue of W")
    fig1.tight_layout()

    # ------------------------------------------------------------------
    # Plot: log10(condition number)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.bar(x, log_cond)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MEASUREMENT_OPTIONS, rotation=45, ha="right")
    ax2.set_ylabel("log10(condition number)")
    ax2.set_title("Empirical observability: condition number of W")
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
