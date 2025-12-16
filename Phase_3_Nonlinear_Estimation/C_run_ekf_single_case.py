"""
Phase 3 - Milestone 3
C_run_ekf_single_case.py

Runs an EKF on a single (motif, measurement option) case using:
  - truth_<motif>.npz  (from A_generate_truth.py)
  - meas_<motif>_<measopt>.npz (from B_generate_measurements.py)

Outputs:
  results/phase3/filters/ekf_<motif>_<measopt>.npz
  results/phase3/figures/ekf_singlecase_<motif>_<measopt>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rms(x: np.ndarray, axis=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x * x, axis=axis))


def _derived_B_from_x(x: np.ndarray) -> np.ndarray:
    # x: (T,6) -> B = Z+S1+S2+S3
    return x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]


def main() -> None:
    cfg = default_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--motif", type=str, default=cfg.motifs[0], help="Motif name (must exist in truth files).")
    parser.add_argument("--meas_opt", type=str, default=cfg.meas_opts[0], help="Measurement option (e.g., h_BNW, h_B, ...).")
    parser.add_argument("--seed_init", type=int, default=2025, help="Seed for initial condition perturbation.")
    parser.add_argument("--init_perturb_scale", type=float, default=0.10, help="Relative perturbation for x0_hat (10% default).")
    parser.add_argument("--q_scale", type=float, default=1.0, help="Multiplier for default Q (tune later).")
    args = parser.parse_args()

    motif = args.motif
    meas_opt = args.meas_opt

    # Output dirs
    out_filters = results_dir() / "filters"
    out_figs = results_dir() / "figures"
    _ensure_dir(out_filters)
    _ensure_dir(out_figs)

    # -----------------------------------------------------------------
    # Load truth + measurements
    # -----------------------------------------------------------------
    truth_path = truth_file(motif)
    meas_path = meas_file(motif, meas_opt)

    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth file: {truth_path}. Run A_generate_truth.py first.")
    if not meas_path.exists():
        raise FileNotFoundError(f"Missing measurement file: {meas_path}. Run B_generate_measurements.py first.")

    truth = np.load(truth_path, allow_pickle=True)
    meas = np.load(meas_path, allow_pickle=True)

    t = np.asarray(truth["t"], dtype=float).reshape(-1)
    dt = float(np.asarray(truth["dt"]).reshape(()))
    x_true = np.asarray(truth["x_true"], dtype=float)
    u_D = np.asarray(truth["u_D"], dtype=float).reshape(-1, 1)

    y_true = np.asarray(meas["y_true"], dtype=float)
    y_meas = np.asarray(meas["y_meas"], dtype=float)
    R = np.asarray(meas["R"], dtype=float)
    meas_names = tuple(str(s) for s in meas["meas_names"].tolist())

    T_steps = t.size
    if x_true.shape != (T_steps, 6):
        raise ValueError(f"x_true must be (T,6). Got {x_true.shape}")
    if u_D.shape[0] != T_steps:
        raise ValueError(f"u_D length mismatch. u_D rows={u_D.shape[0]} vs T={T_steps}")
    if y_meas.shape[0] != T_steps:
        raise ValueError(f"y_meas length mismatch. y_meas rows={y_meas.shape[0]} vs T={T_steps}")
    if R.shape[0] != R.shape[1] or R.shape[0] != y_meas.shape[1]:
        raise ValueError(f"R must be (ny,ny). Got {R.shape}, ny={y_meas.shape[1]}")

    # -----------------------------------------------------------------
    # EKF setup (tunable defaults)
    # -----------------------------------------------------------------
    # Default continuous-ish noise levels -> discretize by Q = dt * Qc
    # (These are intentionally small; tune later if needed.)
    Qc_diag = np.array([1e-3, 1e-3, 1e-3, 1e-3, 5e-4, 5e-4], dtype=float)  # per hour
    Q = (args.q_scale * dt) * np.diag(Qc_diag)

    ekf = BdEKF(Q=Q, R=R, measurement_option=meas_opt)

    # Initial estimate: truth x0 with a small relative perturbation
    rng = np.random.default_rng(args.seed_init)
    x0_true = x_true[0, :].copy()
    rel = args.init_perturb_scale
    x0_hat = x0_true * (1.0 + rel * rng.standard_normal(size=6))

    # Initial covariance (roughly consistent with the perturbation)
    P0 = np.diag((np.maximum(0.05, rel * np.abs(x0_true))) ** 2)

    # Storage
    x_hat = np.zeros((T_steps, 6), dtype=float)
    P_hat = np.zeros((T_steps, 6, 6), dtype=float)
    x_hat[0, :] = x0_hat
    P_hat[0, :, :] = P0

    # -----------------------------------------------------------------
    # EKF loop (predict to k+1, update with measurement at k+1)
    # -----------------------------------------------------------------
    for k in range(T_steps - 1):
        u_k = np.array([float(u_D[k, 0])], dtype=float)

        x_pred, P_pred = ekf.predict(x_hat[k, :], P_hat[k, :, :], u_k, dt)
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_meas[k + 1, :])

        x_hat[k + 1, :] = x_upd
        P_hat[k + 1, :, :] = P_upd

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    err = x_hat - x_true
    rms_states = _rms(err, axis=0)
    state_names = ("Z", "S1", "S2", "S3", "N", "W")

    B_true = _derived_B_from_x(x_true)
    B_hat = _derived_B_from_x(x_hat)
    B_rms = float(_rms(B_hat - B_true))

    # -----------------------------------------------------------------
    # Save filter output
    # -----------------------------------------------------------------
    out_npz = out_filters / f"ekf_{motif}_{meas_opt}.npz"
    np.savez(
        out_npz,
        t=t,
        dt=dt,
        motif_name=str(motif),
        meas_opt=str(meas_opt),
        meas_names=np.array(meas_names, dtype=object),
        x_true=x_true,
        x_hat=x_hat,
        P_hat=P_hat,
        y_meas=y_meas,
        y_true=y_true,
        R=R,
        Q=Q,
        rms_states=rms_states,
        state_names=np.array(state_names, dtype=object),
        B_rms=B_rms,
    )

    # -----------------------------------------------------------------
    # Plot (single figure, saved)
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8))

    # Panel 1: B truth vs estimate, and measured channels if B is measured
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, B_true, label="B true")
    ax1.plot(t, B_hat, label="B hat (EKF)")
    ax1.set_title(f"EKF single case — motif={motif}, meas={meas_opt}")
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("B")
    ax1.grid(True)
    ax1.legend()

    # Panel 2: one measured channel truth/meas vs predicted-from-estimate
    ax2 = fig.add_subplot(2, 1, 2)
    # choose first measurement channel
    j0 = 0
    ax2.plot(t, y_true[:, j0], label=f"{meas_names[j0]} true")
    ax2.plot(t, y_meas[:, j0], label=f"{meas_names[j0]} meas", alpha=0.7)

    # predicted measurement from estimate using same H class logic inside EKF:
    # For simplicity, reconstruct from state directly for common channels.
    if meas_names[j0] == "B":
        y_hat0 = B_hat
    elif meas_names[j0] == "N":
        y_hat0 = x_hat[:, 4]
    elif meas_names[j0] == "W":
        y_hat0 = x_hat[:, 5]
    else:
        y_hat0 = np.full_like(t, np.nan)

    ax2.plot(t, y_hat0, label=f"{meas_names[j0]} hat (from x_hat)")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel(f"{meas_names[j0]}")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()

    out_fig = out_figs / f"ekf_singlecase_{motif}_{meas_opt}.png"
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print("=" * 80)
    print("EKF single-case run complete")
    print(f"truth  : {truth_path}")
    print(f"meas   : {meas_path}")
    print(f"output : {out_npz}")
    print(f"figure : {out_fig}")
    print("-" * 80)
    for name, val in zip(state_names, rms_states):
        print(f"RMS({name}) = {val:.6g}")
    print(f"RMS(B)      = {B_rms:.6g}")
    print("=" * 80)


if __name__ == "__main__":
    main()
