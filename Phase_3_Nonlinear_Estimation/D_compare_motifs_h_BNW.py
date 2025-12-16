"""
Phase 3 - Milestone 4
D_compare_motifs_h_BNW.py

Runs EKF across ALL trajectory motifs for a fixed measurement option (default: h_BNW)
and produces:
  - A comparison figure (RMS(B) and RMS(Z) per motif)
  - A CSV table with metrics per motif
  - An NPZ bundle with aggregated runs (optional but useful)

Inputs:
  results/phase3/truth/truth_<motif>.npz
  results/phase3/meas/meas_<motif>_<measopt>.npz

Outputs:
  results/phase3/figures/ekf_compare_motifs_<measopt>.png
  results/phase3/metrics/ekf_compare_motifs_<measopt>.csv
  results/phase3/filters/ekf_compare_motifs_<measopt>.npz
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

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


STATE_NAMES = ("Z", "S1", "S2", "S3", "N", "W")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rms(x: np.ndarray, axis=None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.mean(x * x, axis=axis))


def _B_from_x(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]


def _case_seed(base_seed: int, motif: str) -> int:
    s = f"{motif}".encode("utf-8")
    return int(base_seed + (zlib.crc32(s) % 2_000_000_000))


def _build_Q(dt: float, q_scale: float) -> np.ndarray:
    """
    Default continuous-ish process noise levels (per hour),
    discretized as Q = dt * Qc and scaled.
    Tune later if needed.
    """
    Qc_diag = np.array([1e-3, 1e-3, 1e-3, 1e-3, 5e-4, 5e-4], dtype=float)
    return (q_scale * dt) * np.diag(Qc_diag)


def run_ekf_case(
    motif: str,
    meas_opt: str,
    q_scale: float,
    seed_init: int,
    init_perturb_scale: float,
) -> Dict[str, np.ndarray]:
    """
    Loads truth+meas for a motif and runs EKF. Returns dict with results and metrics.
    """
    truth_path = truth_file(motif)
    meas_path = meas_file(motif, meas_opt)

    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth file: {truth_path}. Run A_generate_truth.py first.")
    if not meas_path.exists():
        raise FileNotFoundError(
            f"Missing measurement file: {meas_path}. Run B_generate_measurements.py first."
        )

    truth = np.load(truth_path, allow_pickle=True)
    meas = np.load(meas_path, allow_pickle=True)

    t = np.asarray(truth["t"], dtype=float).reshape(-1)
    dt = float(np.asarray(truth["dt"]).reshape(()))
    x_true = np.asarray(truth["x_true"], dtype=float)
    u_D = np.asarray(truth["u_D"], dtype=float).reshape(-1, 1)

    y_meas = np.asarray(meas["y_meas"], dtype=float)
    R = np.asarray(meas["R"], dtype=float)

    T_steps = t.size
    if x_true.shape != (T_steps, 6):
        raise ValueError(f"x_true must be (T,6). Got {x_true.shape}")
    if u_D.shape[0] != T_steps:
        raise ValueError(f"u_D length mismatch. u_D rows={u_D.shape[0]} vs T={T_steps}")
    if y_meas.shape[0] != T_steps:
        raise ValueError(f"y_meas length mismatch. y_meas rows={y_meas.shape[0]} vs T={T_steps}")
    if R.shape != (y_meas.shape[1], y_meas.shape[1]):
        raise ValueError(f"R must be (ny,ny). Got {R.shape}, ny={y_meas.shape[1]}")

    Q = _build_Q(dt=dt, q_scale=q_scale)

    ekf = BdEKF(Q=Q, R=R, measurement_option=meas_opt)

    # Initial estimate: perturb truth x0
    rng = np.random.default_rng(seed_init)
    x0_true = x_true[0, :].copy()
    rel = float(init_perturb_scale)
    x0_hat = x0_true * (1.0 + rel * rng.standard_normal(size=6))

    P0 = np.diag((np.maximum(0.05, rel * np.abs(x0_true))) ** 2)

    x_hat = np.zeros((T_steps, 6), dtype=float)
    P_hat = np.zeros((T_steps, 6, 6), dtype=float)
    x_hat[0, :] = x0_hat
    P_hat[0, :, :] = P0

    for k in range(T_steps - 1):
        u_k = np.array([float(u_D[k, 0])], dtype=float)
        x_pred, P_pred = ekf.predict(x_hat[k, :], P_hat[k, :, :], u_k, dt)
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_meas[k + 1, :])
        x_hat[k + 1, :] = x_upd
        P_hat[k + 1, :, :] = P_upd

    err = x_hat - x_true
    rms_states = _rms(err, axis=0)

    B_true = _B_from_x(x_true)
    B_hat = _B_from_x(x_hat)
    B_err = B_hat - B_true
    B_rms = float(_rms(B_err))
    Z_rms = float(rms_states[0])

    return {
        "t": t,
        "dt": np.array(dt),
        "x_true": x_true,
        "x_hat": x_hat,
        "P_hat": P_hat,
        "B_true": B_true,
        "B_hat": B_hat,
        "B_err": B_err,
        "rms_states": rms_states,
        "B_rms": np.array(B_rms),
        "Z_rms": np.array(Z_rms),
        "Q": Q,
        "R": R,
    }


def main() -> None:
    cfg = default_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--meas_opt", type=str, default="h_BNW", help="Measurement option to use (default h_BNW).")
    parser.add_argument("--q_scale", type=float, default=1.0, help="Multiplier for default Q.")
    parser.add_argument("--seed_init", type=int, default=2025, help="Base seed for initial perturbations.")
    parser.add_argument("--init_perturb_scale", type=float, default=0.10, help="Relative perturbation for x0_hat.")
    parser.add_argument("--motifs", type=str, nargs="*", default=None, help="Override motif list (space-separated).")
    args = parser.parse_args()

    meas_opt = args.meas_opt
    motifs_list = args.motifs if args.motifs is not None else cfg.motifs

    out_figs = results_dir() / "figures"
    out_metrics = results_dir() / "metrics"
    out_filters = results_dir() / "filters"
    _ensure_dir(out_figs)
    _ensure_dir(out_metrics)
    _ensure_dir(out_filters)

    # Run EKF for each motif
    rows = []
    runs: Dict[str, Dict[str, np.ndarray]] = {}

    print("=" * 80)
    print("Phase 3 - Milestone 4: EKF compare motifs")
    print(f"Measurement option : {meas_opt}")
    print(f"Motifs             : {motifs_list}")
    print(f"q_scale            : {args.q_scale}")
    print(f"init perturb (rel) : {args.init_perturb_scale}")
    print("=" * 80)

    for motif in motifs_list:
        seed_case = _case_seed(args.seed_init, motif)
        print(f"Running EKF: motif={motif}, seed_init={seed_case}")

        run = run_ekf_case(
            motif=motif,
            meas_opt=meas_opt,
            q_scale=args.q_scale,
            seed_init=seed_case,
            init_perturb_scale=args.init_perturb_scale,
        )
        runs[motif] = run

        rms_states = run["rms_states"]
        rows.append(
            {
                "motif": motif,
                "meas_opt": meas_opt,
                "RMS_Z": float(rms_states[0]),
                "RMS_S1": float(rms_states[1]),
                "RMS_S2": float(rms_states[2]),
                "RMS_S3": float(rms_states[3]),
                "RMS_N": float(rms_states[4]),
                "RMS_W": float(rms_states[5]),
                "RMS_B": float(run["B_rms"]),
            }
        )

    # Save CSV metrics
    csv_path = out_metrics / f"ekf_compare_motifs_{meas_opt}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["motif", "meas_opt", "RMS_Z", "RMS_S1", "RMS_S2", "RMS_S3", "RMS_N", "RMS_W", "RMS_B"]
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    # Save aggregated NPZ bundle (useful for later plots / MEV-style diagnostics)
    npz_path = out_filters / f"ekf_compare_motifs_{meas_opt}.npz"
    save_dict = {"motifs": np.array(motifs_list, dtype=object)}
    for motif, run in runs.items():
        save_dict[f"{motif}__t"] = run["t"]
        save_dict[f"{motif}__x_true"] = run["x_true"]
        save_dict[f"{motif}__x_hat"] = run["x_hat"]
        save_dict[f"{motif}__P_hat"] = run["P_hat"]
        save_dict[f"{motif}__B_true"] = run["B_true"]
        save_dict[f"{motif}__B_hat"] = run["B_hat"]
        save_dict[f"{motif}__B_err"] = run["B_err"]
        save_dict[f"{motif}__rms_states"] = run["rms_states"]
        save_dict[f"{motif}__B_rms"] = run["B_rms"]
        save_dict[f"{motif}__Z_rms"] = run["Z_rms"]
        save_dict[f"{motif}__Q"] = run["Q"]
        save_dict[f"{motif}__R"] = run["R"]
        save_dict[f"{motif}__dt"] = run["dt"]
    np.savez(npz_path, **save_dict)

    # Build comparison figure (RMS_B and RMS_Z bars)
    motifs_order = [r["motif"] for r in rows]
    rmsB = np.array([r["RMS_B"] for r in rows], dtype=float)
    rmsZ = np.array([r["RMS_Z"] for r in rows], dtype=float)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(motifs_order, rmsB)
    ax1.set_title(f"EKF performance vs motif ({meas_opt}) — RMS(B)")
    ax1.set_xlabel("Motif")
    ax1.set_ylabel("RMS(B)")
    ax1.grid(True, axis="y")
    ax1.tick_params(axis="x", rotation=25)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(motifs_order, rmsZ)
    ax2.set_title(f"EKF performance vs motif ({meas_opt}) — RMS(Z)")
    ax2.set_xlabel("Motif")
    ax2.set_ylabel("RMS(Z)")
    ax2.grid(True, axis="y")
    ax2.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    fig_path = out_figs / f"ekf_compare_motifs_{meas_opt}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Console summary
    print("=" * 80)
    print("Saved:")
    print(f"  Figure : {fig_path}")
    print(f"  CSV    : {csv_path}")
    print(f"  NPZ    : {npz_path}")
    print("-" * 80)
    print("Topline (lower is better):")
    for r in rows:
        print(f"  {r['motif']:>12s} | RMS(B)={r['RMS_B']:.6g} | RMS(Z)={r['RMS_Z']:.6g}")
    print("=" * 80)


if __name__ == "__main__":
    main()
