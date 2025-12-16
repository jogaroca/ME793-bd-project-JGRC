"""
Phase 4
D_eval_data_driven_EKF_batch.py

Runs the data-driven EKF on ALL test trajectories and summarizes RMSE.

Writes:
  results/phase4/metrics/ekf_dd_rmse_by_traj.csv
  results/phase4/metrics/ekf_dd_rmse_summary.csv
  results/phase4/figures/ekf_dd_rmse_summary.png
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

from Utility.bd_dynamics_ann import BdDynamicsANN
from Utility.bd_ekf_data_driven import BdEKFDataDriven


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rmse(e: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(e, dtype=float) ** 2, axis=axis))


def run_one_traj(ekf: BdEKFDataDriven, x_true: np.ndarray, u_traj: np.ndarray, y_meas: np.ndarray,
                 init_perturb_scale: float = 0.10, eps_fd: float = 1e-4, seed: int = 2025) -> np.ndarray:
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
        x_pred, P_pred = ekf.predict(x_hat[k, :], P[k, :, :], u_k, eps_fd=eps_fd)
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_meas[k + 1, :])
        x_hat[k + 1, :] = x_upd
        P[k + 1, :, :] = P_upd

    e = x_hat - x_true
    return rmse(e, axis=0)


def main() -> None:
    test_path = PROJECT_ROOT / "data" / "phase4" / "test" / "bd_test.npz"
    model_dir = PROJECT_ROOT / "models" / "phase4" / "dynamics_ann"
    model_path = model_dir / "model.keras"
    scaler_path = model_dir / "scaler.npz"

    out_metrics = PROJECT_ROOT / "results" / "phase4" / "metrics"
    out_figs = PROJECT_ROOT / "results" / "phase4" / "figures"
    ensure_dir(out_metrics); ensure_dir(out_figs)

    D = np.load(test_path, allow_pickle=True)
    X = D["X"].astype(float)
    U = D["U"].astype(float)
    Y = D["Y_meas"].astype(float)
    R = D["R"].astype(float)
    meas_opt = str(D["meas_opt"])

    dd = BdDynamicsANN(model_path=model_path, scaler_path=scaler_path)

    # same Q as C_data_driven_EKF.py (can tune later)
    proc_sigma = np.array([5e-4, 5e-4, 5e-4, 5e-4, 8e-4, 5e-4], dtype=float)
    Q = np.diag(proc_sigma ** 2)

    ekf = BdEKFDataDriven(f_dd_model=dd, Q=Q, R=R, measurement_option=meas_opt)

    Ntraj = X.shape[0]
    state_names = ("Z", "S1", "S2", "S3", "N", "W")

    rmse_all = np.zeros((Ntraj, 6), dtype=float)
    for i in range(Ntraj):
        rmse_all[i, :] = run_one_traj(ekf, X[i, :, :], U[i, :, :], Y[i, :, :],
                                      init_perturb_scale=0.10, eps_fd=1e-4, seed=2025 + i)

    # Save per-trajectory RMSE
    bytraj_path = out_metrics / "ekf_dd_rmse_by_traj.csv"
    with open(bytraj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj_idx"] + [f"rmse_{s}" for s in state_names])
        for i in range(Ntraj):
            w.writerow([i] + [float(v) for v in rmse_all[i, :]])

    # Summary
    mean_rmse = rmse_all.mean(axis=0)
    std_rmse = rmse_all.std(axis=0)

    summary_path = out_metrics / "ekf_dd_rmse_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "rmse_mean", "rmse_std"])
        for s, m, sd in zip(state_names, mean_rmse, std_rmse):
            w.writerow([s, float(m), float(sd)])

    # Plot summary
    fig_path = out_figs / "ekf_dd_rmse_summary.png"
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(6), mean_rmse)
    plt.xticks(np.arange(6), state_names)
    plt.ylabel("RMSE (mean over test trajs)")
    plt.title(f"Data-driven EKF RMSE summary ({meas_opt})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("=" * 80)
    print("Batch evaluation complete")
    print("by-traj:", bytraj_path)
    print("summary:", summary_path)
    print("figure :", fig_path)
    for s, m, sd in zip(state_names, mean_rmse, std_rmse):
        print(f"{s:>3s}: mean={m:.6g}, std={sd:.6g}")
    print("=" * 80)


if __name__ == "__main__":
    main()
