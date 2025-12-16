"""
Phase 3 - Milestone 5
F_crank_sinusoid_Bonly_EKF_vs_UKF.py

Goal:
  - Hard case: B-only measurements (meas_opt = h_B)
  - Compare motif "sin_12h" vs "sin_12h_big" (cranked sinusoid)
  - Compare EKF vs UKF

Inputs:
  results/phase3/truth/truth_<motif>.npz
  results/phase3/meas/meas_<motif>_h_B.npz

Outputs:
  results/phase3/figures/bonly_crank_ekf_vs_ukf.png
  results/phase3/metrics/bonly_crank_ekf_vs_ukf.csv
  results/phase3/filters/bonly_crank_ekf_vs_ukf.npz
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Robust imports (project root on path)
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase_3_Nonlinear_Estimation.config_phase3 import (
    default_config,
    meas_file,
    results_dir,
    truth_file,
)
from Utility.bd_ekf import BdEKF

# IMPORTANT:
# This assumes you have Utility/bd_ukf.py defining BdUKF with predict/update methods
# matching BdEKF's interface: predict(x,P,u,dt) and update(x,P,y).
from Utility.bd_ukf import BdUKF  # adjust name here if your class is different


STATE_NAMES = ("Z", "S1", "S2", "S3", "N", "W")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rms(x: np.ndarray, axis=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x * x, axis=axis))


def _B_from_x(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]


def _case_seed(base_seed: int, motif: str, filt_name: str) -> int:
    s = f"{motif}|{filt_name}".encode("utf-8")
    return int(base_seed + (zlib.crc32(s) % 2_000_000_000))


def _build_Q(dt: float, q_scale: float) -> np.ndarray:
    # Default continuous-ish Qc (per hour), discretized as Q = dt*Qc and scaled.
    Qc_diag = np.array([1e-3, 1e-3, 1e-3, 1e-3, 5e-4, 5e-4], dtype=float)
    return (q_scale * dt) * np.diag(Qc_diag)


def _run_filter(
    filter_obj,
    t: np.ndarray,
    dt: float,
    x_true: np.ndarray,
    u_D: np.ndarray,
    y_meas: np.ndarray,
    seed_init: int,
    init_perturb_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic EKF/UKF runner with the common interface:
      predict(x,P,u,dt) -> (x_pred,P_pred)
      update(x,P,y)     -> (x_upd,P_upd)

    Update is performed at time k+1 using measurement y_meas[k+1].
    """
    T_steps = t.size

    # Initial estimate: perturb truth x0
    rng = np.random.default_rng(seed_init)
    x0_true = x_true[0, :].copy()
    rel = float(init_perturb_scale)
    x0_hat = x0_true * (1.0 + rel * rng.standard_normal(size=6))

    # Initial covariance consistent with perturbation
    P0 = np.diag((np.maximum(0.05, rel * np.abs(x0_true))) ** 2)

    x_hat = np.zeros((T_steps, 6), dtype=float)
    P_hat = np.zeros((T_steps, 6, 6), dtype=float)
    x_hat[0, :] = x0_hat
    P_hat[0, :, :] = P0

    for k in range(T_steps - 1):
        u_k = np.array([float(u_D[k, 0])], dtype=float)
        x_pred, P_pred = filter_obj.predict(x_hat[k, :], P_hat[k, :, :], u_k, dt)
        x_upd, P_upd = filter_obj.update(x_pred, P_pred, y_meas[k + 1, :])
        x_hat[k + 1, :] = x_upd
        P_hat[k + 1, :, :] = P_upd

    return x_hat, P_hat


def main() -> None:
    cfg = default_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--meas_opt", type=str, default="h_B", help="Measurement option (default: h_B).")
    parser.add_argument("--motif_a", type=str, default="sin_12h", help="Baseline motif.")
    parser.add_argument("--motif_b", type=str, default="sin_12h_big", help="Cranked motif.")
    parser.add_argument("--q_scale", type=float, default=1.0, help="Multiplier for default Q.")
    parser.add_argument("--seed_init", type=int, default=2025, help="Base seed for initial perturbation.")
    parser.add_argument("--init_perturb_scale", type=float, default=0.10, help="Relative perturbation for x0_hat.")
    args = parser.parse_args()

    meas_opt = args.meas_opt
    motifs = [args.motif_a, args.motif_b]()
