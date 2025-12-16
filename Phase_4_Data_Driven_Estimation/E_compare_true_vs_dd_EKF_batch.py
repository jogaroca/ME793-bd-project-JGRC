"""
Phase 4
E_compare_true_vs_dd_EKF_batch.py

Compares:
  (1) EKF with TRUE physics model (Utility.bd_ekf.BdEKF)
  (2) EKF with DATA-DRIVEN dynamics (Utility.bd_ekf_data_driven.BdEKFDataDriven)

Both filters use the same measurement set and the same R from the dataset.
Outputs summary RMSE mean/std across test trajectories, plus a comparison plot.

Writes:
  results/phase4/metrics/compare_true_vs_dd_rmse_by_traj.csv
  results/phase4/metrics/compare_true_vs_dd_rmse_summary.csv
  results/phase4/figures/compare_true_vs_dd_rmse_summary.png
"""

from __future__ import annotations
import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility.bd_ekf import BdEKF
from Utility.bd_dynamics_ann import BdDynamicsANN
from Utility.bd_ekf_data_driven import BdEKFDataDriven


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rmse(e: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(e, dtype=float) ** 2, axis=axis))


def run_one_traj_true(
    ekf_true: BdEKF,
    x_true: np.ndarray,
    u_traj: np.ndarray,
    y_meas: np.ndarray,
    dt: float,
    init_perturb_scale: float,
    seed: int,
) -> np.ndarray:
    Tsteps = x_true.shape[0]
    rng = np.random.default_rng(seed)

    x_hat = np.zeros_like(x_true)
    P = np.zeros((Tsteps, 6, 6), dtype=float)

    x0_true = x_true[0, :].copy()
    rel = float(init_perturb_scale)

    x_hat[0, :] = x0_true * (1.0 + rel * rng.standard_normal(size=6))
    P[0, :, :] = np.diag((np.maximum(0.05, rel * np.abs(x0_true))) ** 2)

    for k in range(Tsteps - 1):
        u_k = u_traj[k, :].reshape(-1)
        x_pred, P_pred = ekf_true.predict(x_hat[k, :], P[k, :, :], u_k, dt)
        x_upd, P_upd = ekf_true.update(x_pred, P_pred, y_meas[k + 1, :])
        x_hat[k + 1, :] = x_upd
        P[k + 1, :, :] = P_upd

    return rmse(x_hat - x_true, axis=0)


def run_one_traj_dd(
    ekf_dd: BdEKFDataDriven,
    x_true: np.ndarray,
    u_traj: np.ndarray,
    y_meas: np.ndarray,
    init_perturb_scale: float,
    seed: int,
    eps_fd: float = 1e-4,
) -> np.ndarray:
    Tsteps = x_true.shape[0]
    rng = np.random.default_rng(seed)

    x_hat = np.zeros_like(x_true)
    P = np.zeros((Tsteps, 6, 6), dtype=float)

    x0_true = x_true[0, :].copy()
    rel = float(init_perturb_scale)

    x_hat[0, :] = x0_true * (1.0 + rel * rng.standard_normal(size=6))
    P[0, :, :] = np.diag((np.maximum(0.05, rel * np.abs(x0_true))) ** 2)

    for k in range(Tsteps - 1):
        u_k = u_traj[k, :].reshape(1,)
        x_pred, P_pred = ekf_dd.predict(x_hat[k, :], P[k, :, :], u_k, eps_fd=eps_fd)
        x_upd, P_upd = ekf_dd.update(x_pred, P_pred, y_meas[k + 1, :])
        x_hat[k + 1, :] = x_upd
        P[k + 1, :, :] = P_upd

    return rmse(x_hat - x_true, axis=0)


def main() -> None:
    # Inputs
    test_path = PROJECT_ROOT / "data" / "phase4" / "test" / "bd_test.npz"
    model_dir = PROJECT_ROOT / "models" / "phase4" / "dynamics_ann"
    model_path = model_dir / "model.keras"
    scaler_path = model_dir / "scaler.npz"

    out_metrics = PROJECT_ROOT / "results" / "phase4" / "metrics"
    out_figs = PROJECT_ROOT / "results" / "phase4" / "figures"
    ensure_dir(out_metrics); ensure_dir(out_figs)

    D = np.load(test_path, allow_pickle=True)
    dt = float(D["dt"])
    X = D["X"].astype(float)
    U = D["U"].astype(float)
    Y = D["Y_meas"].astype(float)
    R = D["R"].astype(float)
    meas_opt = str(D["meas_opt"])

    # Same Q for both filters (discrete)
    proc_sigma = np.array([5e-4, 5e-4, 5e-4, 5e-4, 8e-4, 5e-4], dtype=float)
    Q = np.diag(proc_sigma ** 2)

    # Build filters
    ekf_true = BdEKF(Q=Q, R=R, measurement_option=meas_opt)

    dd = BdDynamicsANN(model_path=model_path, scaler_path=scaler_path)
    ekf_dd = BdEKFDataDriven(f_dd_model=dd, Q=Q, R=R, measurement_option=meas_opt)

    Ntraj = X.shape[0]
    state_names = ("Z", "S1", "S2", "S3", "N", "W")

    init_perturb_scale = 0.10
    eps_fd = 1e-4

    rmse_true_all = np.zeros((Ntraj, 6), dtype=float)
    rmse_dd_all   = np.zeros((Ntraj, 6), dtype=float)

    for i in range(Ntraj):
        seed_i = 3000 + i  # same seed used for both filters on same trajectory
        rmse_true_all[i, :] = run_one_traj_true(
            ekf_true, X[i, :, :], U[i, :, :], Y[i, :, :],
            dt=dt, init_perturb_scale=init_perturb_scale, seed=seed_i
        )
        rmse_dd_all[i, :] = run_one_traj_dd(
            ekf_dd, X[i, :, :], U[i, :, :], Y[i, :, :],
            init_perturb_scale=init_perturb_scale, seed=seed_i, eps_fd=eps_fd
        )

    # Save per-trajectory results
    bytraj_path = out_metrics / "compare_true_vs_dd_rmse_by_traj.csv"
    with open(bytraj_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["traj_idx"] + [f"true_{s}" for s in state_names] + [f"dd_{s}" for s in state_names]
        w.writerow(header)
        for i in range(Ntraj):
            w.writerow([i] + [float(v) for v in rmse_true_all[i, :]] + [float(v) for v in rmse_dd_all[i, :]])

    # Summary stats
    true_mean = rmse_true_all.mean(axis=0)
    true_std  = rmse_true_all.std(axis=0)
    dd_mean   = rmse_dd_all.mean(axis=0)
    dd_std    = rmse_dd_all.std(axis=0)
    ratio     = dd_mean / np.maximum(true_mean, 1e-12)

    summary_path = out_metrics / "compare_true_vs_dd_rmse_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "true_mean", "true_std", "dd_mean", "dd_std", "dd_over_true"])
        for s, tm, ts, dm, ds, r in zip(state_names, true_mean, true_std, dd_mean, dd_std, ratio):
            w.writerow([s, float(tm), float(ts), float(dm), float(ds), float(r)])

    # Plot comparison bars
    fig_path = out_figs / "compare_true_vs_dd_rmse_summary.png"
    x = np.arange(6)
    width = 0.38

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width/2, true_mean, width=width, label="EKF (true model)")
    plt.bar(x + width/2, dd_mean,   width=width, label="EKF (data-driven)")
    plt.xticks(x, state_names)
    plt.ylabel("RMSE (mean over test trajs)")
    plt.title(f"Phase 4: True-model EKF vs Data-driven EKF ({meas_opt})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("=" * 80)
    print("Comparison complete")
    print("by-traj :", bytraj_path)
    print("summary :", summary_path)
    print("figure  :", fig_path)
    print("State: true_mean vs dd_mean (dd/true)")
    for s, tm, dm, r in zip(state_names, true_mean, dd_mean, ratio):
        print(f"{s:>3s}: {tm:.6g} vs {dm:.6g}  (x{r:.3g})")
    print("=" * 80)


if __name__ == "__main__":
    main()
