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
    motifs = [args.motif_a, args.motif_b]

    out_figs = results_dir() / "figures"
    out_metrics = results_dir() / "metrics"
    out_filters = results_dir() / "filters"
    _ensure_dir(out_figs)
    _ensure_dir(out_metrics)
    _ensure_dir(out_filters)

    print("=" * 80)
    print("Phase 3 - Milestone 5: B-only crank comparison (EKF vs UKF)")
    print(f"meas_opt           : {meas_opt}")
    print(f"motifs             : {motifs}")
    print(f"q_scale            : {args.q_scale}")
    print(f"init perturb (rel) : {args.init_perturb_scale}")
    print("=" * 80)

    # Aggregate results
    runs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    summary_rows = []

    for motif in motifs:
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

        y_meas = np.asarray(meas["y_meas"], dtype=float)
        R = np.asarray(meas["R"], dtype=float)
        meas_names = tuple(str(s) for s in meas["meas_names"].tolist())

        if meas_names != ("B",):
            raise ValueError(
                f"This script expects B-only measurements => meas_names=('B',). Got meas_names={meas_names}"
            )

        T_steps = t.size
        if x_true.shape != (T_steps, 6):
            raise ValueError(f"x_true must be (T,6). Got {x_true.shape}")
        if u_D.shape[0] != T_steps or y_meas.shape[0] != T_steps:
            raise ValueError("Length mismatch among t, u_D, y_meas.")
        if R.shape != (1, 1):
            raise ValueError(f"B-only R must be (1,1). Got {R.shape}")

        Q = _build_Q(dt=dt, q_scale=args.q_scale)

        # Build filters
        ekf = BdEKF(Q=Q, R=R, measurement_option=meas_opt)
        ukf = BdUKF(Q=Q, R=R, measurement_option=meas_opt)

        runs[motif] = {}

        # Run EKF
        seed_ekf = _case_seed(args.seed_init, motif, "EKF")
        x_hat_ekf, P_hat_ekf = _run_filter(
            filter_obj=ekf,
            t=t,
            dt=dt,
            x_true=x_true,
            u_D=u_D,
            y_meas=y_meas,
            seed_init=seed_ekf,
            init_perturb_scale=args.init_perturb_scale,
        )
        runs[motif]["EKF"] = {"x_hat": x_hat_ekf, "P_hat": P_hat_ekf}

        # Run UKF
        seed_ukf = _case_seed(args.seed_init, motif, "UKF")
        x_hat_ukf, P_hat_ukf = _run_filter(
            filter_obj=ukf,
            t=t,
            dt=dt,
            x_true=x_true,
            u_D=u_D,
            y_meas=y_meas,
            seed_init=seed_ukf,
            init_perturb_scale=args.init_perturb_scale,
        )
        runs[motif]["UKF"] = {"x_hat": x_hat_ukf, "P_hat": P_hat_ukf}

        # Compute metrics (focus on Z since B is measured)
        for filt_name, x_hat in [("EKF", x_hat_ekf), ("UKF", x_hat_ukf)]:
            err = x_hat - x_true
            rms_states = _rms(err, axis=0)
            Z_rms = float(rms_states[0])

            B_true = _B_from_x(x_true)
            B_hat = _B_from_x(x_hat)
            B_rms = float(_rms(B_hat - B_true))

            summary_rows.append(
                {
                    "motif": motif,
                    "filter": filt_name,
                    "meas_opt": meas_opt,
                    "RMS_Z": Z_rms,
                    "RMS_B": B_rms,
                    "seed_init": seed_ekf if filt_name == "EKF" else seed_ukf,
                    "q_scale": float(args.q_scale),
                    "init_perturb_scale": float(args.init_perturb_scale),
                }
            )

        # Store shared arrays for saving
        runs[motif]["_shared"] = {
            "t": t,
            "dt": np.array(dt),
            "x_true": x_true,
            "u_D": u_D,
            "y_meas": y_meas,
            "Q": Q,
            "R": R,
        }

    # -----------------------------------------------------------------
    # Save CSV metrics
    # -----------------------------------------------------------------
    csv_path = out_metrics / "bonly_crank_ekf_vs_ukf.csv"
    header = ["motif", "filter", "meas_opt", "RMS_Z", "RMS_B", "seed_init", "q_scale", "init_perturb_scale"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    # -----------------------------------------------------------------
    # Save NPZ bundle
    # -----------------------------------------------------------------
    npz_path = out_filters / "bonly_crank_ekf_vs_ukf.npz"
    save_dict = {}
    for motif in motifs:
        shared = runs[motif]["_shared"]
        save_dict[f"{motif}__t"] = shared["t"]
        save_dict[f"{motif}__dt"] = shared["dt"]
        save_dict[f"{motif}__x_true"] = shared["x_true"]
        save_dict[f"{motif}__u_D"] = shared["u_D"]
        save_dict[f"{motif}__y_meas"] = shared["y_meas"]
        save_dict[f"{motif}__Q"] = shared["Q"]
        save_dict[f"{motif}__R"] = shared["R"]

        for filt_name in ("EKF", "UKF"):
            save_dict[f"{motif}__{filt_name}__x_hat"] = runs[motif][filt_name]["x_hat"]
            save_dict[f"{motif}__{filt_name}__P_hat"] = runs[motif][filt_name]["P_hat"]

    np.savez(npz_path, **save_dict)

    # -----------------------------------------------------------------
    # Build figure (2 columns: motifs; row1: B true/hat; row2: Z error^2 and Pzz)
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))

    for col, motif in enumerate(motifs, start=1):
        shared = runs[motif]["_shared"]
        t = shared["t"]
        x_true = shared["x_true"]
        B_true = _B_from_x(x_true)
        Z_true = x_true[:, 0]

        # --- Row 1: B trajectory (measured) ---
        ax1 = fig.add_subplot(2, 2, col)
        ax1.plot(t, B_true, label="B true")
        for filt_name in ("EKF", "UKF"):
            x_hat = runs[motif][filt_name]["x_hat"]
            B_hat = _B_from_x(x_hat)
            ax1.plot(t, B_hat, label=f"B hat ({filt_name})")
        ax1.set_title(f"{motif} (B-only): B true vs estimates")
        ax1.set_xlabel("Time [h]")
        ax1.set_ylabel("B")
        ax1.grid(True)
        ax1.legend()

        # --- Row 2: Z squared error + filter Pzz (consistency diagnostic) ---
        ax2 = fig.add_subplot(2, 2, col + 2)
        for filt_name in ("EKF", "UKF"):
            x_hat = runs[motif][filt_name]["x_hat"]
            P_hat = runs[motif][filt_name]["P_hat"]

            Z_hat = x_hat[:, 0]
            e2 = (Z_hat - Z_true) ** 2
            Pzz = P_hat[:, 0, 0]

            ax2.plot(t, e2, label=f"(Z_hat - Z_true)^2 ({filt_name})", alpha=0.9)
            ax2.plot(t, Pzz, label=f"Pzz ({filt_name})", linestyle="--", alpha=0.9)

        ax2.set_title(f"{motif} (B-only): Z error^2 and Pzz")
        ax2.set_xlabel("Time [h]")
        ax2.set_ylabel("Variance proxy")
        ax2.grid(True)
        ax2.legend()

    fig.tight_layout()
    fig_path = out_figs / "bonly_crank_ekf_vs_ukf.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Print topline summary + crank effect
    # -----------------------------------------------------------------
    def _get_rmsZ(m: str, f: str) -> float:
        for r in summary_rows:
            if r["motif"] == m and r["filter"] == f:
                return float(r["RMS_Z"])
        return float("nan")

    ekf_a = _get_rmsZ(args.motif_a, "EKF")
    ekf_b = _get_rmsZ(args.motif_b, "EKF")
    ukf_a = _get_rmsZ(args.motif_a, "UKF")
    ukf_b = _get_rmsZ(args.motif_b, "UKF")

    print("=" * 80)
    print("Saved:")
    print(f"  Figure : {fig_path}")
    print(f"  CSV    : {csv_path}")
    print(f"  NPZ    : {npz_path}")
    print("-" * 80)
    print("RMS(Z) summary (lower is better):")
    print(f"  EKF: {args.motif_a} -> {ekf_a:.6g} | {args.motif_b} -> {ekf_b:.6g}")
    print(f"  UKF: {args.motif_a} -> {ukf_a:.6g} | {args.motif_b} -> {ukf_b:.6g}")

    if np.isfinite(ekf_a) and np.isfinite(ekf_b) and ekf_a > 0:
        print(f"  EKF crank improvement: {(ekf_a - ekf_b) / ekf_a * 100:.2f}%")
    if np.isfinite(ukf_a) and np.isfinite(ukf_b) and ukf_a > 0:
        print(f"  UKF crank improvement: {(ukf_a - ukf_b) / ukf_a * 100:.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
