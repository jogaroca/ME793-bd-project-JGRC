"""
Phase 3 - Support figure (highly recommended)
E_compare_measurement_sets.py

Fix one motif and compare EKF performance across measurement options.
Outputs:
  - Figure: RMS(Z) and RMS(B) vs measurement option
  - CSV metrics

Run example:
  python Phase_3_Nonlinear_Estimation/E_compare_measurement_sets.py --motif sin_12h
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
from Phase_3_Nonlinear_Estimation.phase3_filter_utils import load_ekf_class, construct_filter


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rms(x: np.ndarray, axis=None) -> np.ndarray:
    x = np.asarray(x, float)
    return np.sqrt(np.mean(x * x, axis=axis))


def _B_from_x(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]


def _case_seed(base_seed: int, motif: str, meas_opt: str) -> int:
    s = f"{motif}|{meas_opt}".encode("utf-8")
    return int(base_seed + (zlib.crc32(s) % 2_000_000_000))


def _build_Q(dt: float, q_scale: float) -> np.ndarray:
    # Default continuous-ish Qc (per hour), discretized as Q = dt*Qc
    Qc_diag = np.array([1e-3, 1e-3, 1e-3, 1e-3, 5e-4, 5e-4], float)
    return (q_scale * dt) * np.diag(Qc_diag)


def run_case(motif: str, meas_opt: str, q_scale: float, seed_init: int, init_perturb_scale: float):
    truth = np.load(truth_file(motif), allow_pickle=True)
    meas = np.load(meas_file(motif, meas_opt), allow_pickle=True)

    t = np.asarray(truth["t"], float).reshape(-1)
    dt = float(np.asarray(truth["dt"]).reshape(()))
    x_true = np.asarray(truth["x_true"], float)
    u_D = np.asarray(truth["u_D"], float).reshape(-1, 1)

    y_meas = np.asarray(meas["y_meas"], float)
    R = np.asarray(meas["R"], float)

    Q = _build_Q(dt, q_scale)

    EKFClass = load_ekf_class()
    ekf = construct_filter(EKFClass, Q=Q, R=R, measurement_option=meas_opt)

    rng = np.random.default_rng(seed_init)
    x0_true = x_true[0].copy()
    rel = float(init_perturb_scale)
    x0_hat = x0_true * (1.0 + rel * rng.standard_normal(6))
    P0 = np.diag((np.maximum(0.05, rel * np.abs(x0_true))) ** 2)

    T_steps = t.size
    x_hat = np.zeros((T_steps, 6), float)
    P_hat = np.zeros((T_steps, 6, 6), float)
    x_hat[0] = x0_hat
    P_hat[0] = P0

    for k in range(T_steps - 1):
        u_k = np.array([float(u_D[k, 0])], float)
        x_pred, P_pred = ekf.predict(x_hat[k], P_hat[k], u_k, dt)
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_meas[k + 1])
        x_hat[k + 1] = x_upd
        P_hat[k + 1] = P_upd

    err = x_hat - x_true
    rms_states = _rms(err, axis=0)

    B_true = _B_from_x(x_true)
    B_hat = _B_from_x(x_hat)
    B_rms = float(_rms(B_hat - B_true))

    return rms_states, B_rms


def main():
    cfg = default_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", type=str, default="sin_12h")
    ap.add_argument("--meas_opts", type=str, nargs="*", default=None)
    ap.add_argument("--q_scale", type=float, default=1.0)
    ap.add_argument("--seed_init", type=int, default=2025)
    ap.add_argument("--init_perturb_scale", type=float, default=0.10)
    args = ap.parse_args()

    motif = args.motif
    meas_opts = args.meas_opts if args.meas_opts is not None else cfg.meas_opts

    out_figs = results_dir() / "figures"
    out_metrics = results_dir() / "metrics"
    _ensure_dir(out_figs)
    _ensure_dir(out_metrics)

    rows = []
    for meas_opt in meas_opts:
        seed_case = _case_seed(args.seed_init, motif, meas_opt)
        rms_states, B_rms = run_case(motif, meas_opt, args.q_scale, seed_case, args.init_perturb_scale)
        rows.append(
            {
                "motif": motif,
                "meas_opt": meas_opt,
                "RMS_Z": float(rms_states[0]),
                "RMS_B": float(B_rms),
                "RMS_N": float(rms_states[4]),
                "RMS_W": float(rms_states[5]),
            }
        )

    csv_path = out_metrics / f"ekf_compare_measopts_{motif}.csv"
    header = ["motif", "meas_opt", "RMS_Z", "RMS_B", "RMS_N", "RMS_W"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    labels = [r["meas_opt"] for r in rows]
    rmsZ = np.array([r["RMS_Z"] for r in rows], float)
    rmsB = np.array([r["RMS_B"] for r in rows], float)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(labels, rmsZ)
    ax1.set_title(f"EKF vs measurement option (motif={motif}) — RMS(Z)")
    ax1.set_xlabel("Measurement option")
    ax1.set_ylabel("RMS(Z)")
    ax1.grid(True, axis="y")
    ax1.tick_params(axis="x", rotation=25)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(labels, rmsB)
    ax2.set_title(f"EKF vs measurement option (motif={motif}) — RMS(B)")
    ax2.set_xlabel("Measurement option")
    ax2.set_ylabel("RMS(B)")
    ax2.grid(True, axis="y")
    ax2.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig_path = out_figs / f"ekf_compare_measopts_{motif}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print("Saved:", fig_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
