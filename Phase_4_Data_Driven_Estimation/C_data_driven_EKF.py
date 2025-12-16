"""
Phase 4
C_data_driven_EKF.py

Runs EKF using ANN discrete dynamics model on test data.

Reads:
  data/phase4/test/bd_test.npz
  models/phase4/dynamics_ann/model.keras
  models/phase4/dynamics_ann/scaler.npz

Writes:
  results/phase4/figures/ekf_dd_singletraj.png
  results/phase4/metrics/ekf_dd_rmse.csv
  results/phase4/filters/ekf_dd_singletraj.npz
"""

from __future__ import annotations
import argparse
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


def rmse(x: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2, axis=axis))


def B_from_x(X: np.ndarray) -> np.ndarray:
    return X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_idx", type=int, default=0)
    parser.add_argument("--init_perturb_scale", type=float, default=0.10)
    parser.add_argument("--q_scale", type=float, default=1.0)
    parser.add_argument("--eps_fd", type=float, default=1e-4)
    args = parser.parse_args()

    # Paths
    test_path = PROJECT_ROOT / "data" / "phase4" / "test" / "bd_test.npz"
    model_dir = PROJECT_ROOT / "models" / "phase4" / "dynamics_ann"
    model_path = model_dir / "model.keras"
    scaler_path = model_dir / "scaler.npz"

    out_figs = PROJECT_ROOT / "results" / "phase4" / "figures"
    out_metrics = PROJECT_ROOT / "results" / "phase4" / "metrics"
    out_filters = PROJECT_ROOT / "results" / "phase4" / "filters"
    ensure_dir(out_figs); ensure_dir(out_metrics); ensure_dir(out_filters)

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test data: {test_path}")
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Missing ANN model/scaler. Run B_train_dynamics_model_ann.py first.")

    # Load data
    D = np.load(test_path, allow_pickle=True)
    t = D["t"].astype(float)
    dt = float(D["dt"])
    X = D["X"].astype(float)         # (Ntraj,T,6)
    U = D["U"].astype(float)         # (Ntraj,T,1)
    Y = D["Y_meas"].astype(float)    # (Ntraj,T,ny)
    R = D["R"].astype(float)         # (ny,ny)
    meas_opt = str(D["meas_opt"])

    Ntraj, Tsteps, nx = X.shape
    if args.traj_idx < 0 or args.traj_idx >= Ntraj:
        raise ValueError(f"traj_idx out of range. Got {args.traj_idx}, Ntraj={Ntraj}")

    x_true = X[args.traj_idx, :, :]
    u_traj = U[args.traj_idx, :, :]
    y_meas = Y[args.traj_idx, :, :]

    # Load ANN dynamics
    dd = BdDynamicsANN(model_path=model_path, scaler_path=scaler_path)

    # EKF tuning (discrete Q)
    # Keep it small; scale with q_scale if you need more smoothing/robustness.
    proc_sigma = np.array([5e-4, 5e-4, 5e-4, 5e-4, 8e-4, 5e-4], dtype=float)
    Q = (args.q_scale) * np.diag(proc_sigma ** 2)

    ekf = BdEKFDataDriven(f_dd_model=dd, Q=Q, R=R, measurement_option=meas_opt)

    # Initialization
    rng = np.random.default_rng(2025)
    x0_true = x_true[0, :].copy()
    rel = float(args.init_perturb_scale)
    x_hat = np.zeros_like(x_true)
    P = np.zeros((Tsteps, 6, 6), dtype=float)

    x_hat[0, :] = x0_true * (1.0 + rel * rng.standard_normal(size=6))
    P[0, :, :] = np.diag((np.maximum(0.05, rel * np.abs(x0_true))) ** 2)

    # EKF loop (update with y_{k+1})
    for k in range(Tsteps - 1):
        u_k = u_traj[k, :].reshape(1,)  # (1,)
        x_pred, P_pred = ekf.predict(x_hat[k, :], P[k, :, :], u_k, eps_fd=args.eps_fd)
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_meas[k + 1, :])
        x_hat[k + 1, :] = x_upd
        P[k + 1, :, :] = P_upd

    # Metrics
    err = x_hat - x_true
    rmse_states = rmse(err, axis=0)
    state_names = ("Z", "S1", "S2", "S3", "N", "W")

    # Save metrics CSV
    metrics_path = out_metrics / "ekf_dd_rmse.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "rmse"])
        for s, v in zip(state_names, rmse_states):
            w.writerow([s, float(v)])

    # Save filter output
    filt_path = out_filters / "ekf_dd_singletraj.npz"
    np.savez(
        filt_path,
        t=t, dt=dt,
        traj_idx=int(args.traj_idx),
        meas_opt=meas_opt,
        x_true=x_true,
        x_hat=x_hat,
        P=P,
        y_meas=y_meas,
        Q=Q, R=R,
        rmse_states=rmse_states,
        state_names=np.array(state_names, dtype=object),
    )

    # Plot
    B_true = B_from_x(x_true)
    B_hat  = B_from_x(x_hat)

    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, B_true, label="B true")
    ax1.plot(t, B_hat, label="B hat (EKF-dd)")
    ax1.set_title(f"Data-driven EKF (traj={args.traj_idx}, meas={meas_opt})")
    ax1.set_ylabel("B")
    ax1.grid(True); ax1.legend()

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t, x_true[:, 4], label="N true")
    ax2.plot(t, x_hat[:, 4], label="N hat")
    ax2.plot(t, y_meas[:, 1], label="N meas", alpha=0.6)  # for h_BNW channel order [B,N,W]
    ax2.set_ylabel("N")
    ax2.grid(True); ax2.legend()

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(t, x_true[:, 5], label="W true")
    ax3.plot(t, x_hat[:, 5], label="W hat")
    ax3.plot(t, y_meas[:, 2], label="W meas", alpha=0.6)
    ax3.set_xlabel("Time [h]")
    ax3.set_ylabel("W")
    ax3.grid(True); ax3.legend()

    fig.tight_layout()
    fig_path = out_figs / "ekf_dd_singletraj.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print("=" * 80)
    print("Data-driven EKF complete")
    print(f"figure : {fig_path}")
    print(f"metrics: {metrics_path}")
    print(f"filter : {filt_path}")
    for s, v in zip(state_names, rmse_states):
        print(f"RMSE({s}) = {v:.6g}")
    print("=" * 80)


if __name__ == "__main__":
    main()
