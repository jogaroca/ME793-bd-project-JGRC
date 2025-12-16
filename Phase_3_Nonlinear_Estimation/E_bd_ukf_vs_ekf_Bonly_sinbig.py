"""
Phase 3: Compare EKF vs UKF for B-only measurements on the "cranked" sinusoidal motif.

- Loads truth from: results/phase3_estimation/truth_motif_sin_12h_big.npz
- Generates noisy measurement y = B + noise
- Runs EKF (measurement_option="h_B") and UKF (measurement_option="h_B")
- Compares RMS error in Z and plots Z_true vs Z_hat
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from Utility.bd_io import ensure_dir, load_truth_or_fail


from Utility import bd_ekf
from Utility import bd_ukf


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def run_filter(filter_obj, t, u_D, y_meas, dt, x0_hat, P0_hat):
    T_steps = t.size
    nx = 6

    x_hat = np.zeros((T_steps, nx))
    P_hat = np.zeros((T_steps, nx, nx))

    x_hat[0, :] = x0_hat
    P_hat[0, :, :] = P0_hat

    for k in range(T_steps - 1):
        u_k = np.array([u_D[k, 0]])  # dilution scalar in 1D array

        x_pred, P_pred = filter_obj.predict(x_hat[k, :], P_hat[k, :, :], u_k, dt)

        y_k1 = y_meas[k + 1, :]
        x_upd, P_upd = filter_obj.update(x_pred, P_pred, y_k1)

        x_hat[k + 1, :] = x_upd
        P_hat[k + 1, :, :] = P_upd

    return x_hat, P_hat


def main():
    motif_name = "sin_12h_big"
    meas_opt = "h_B"

    truth_path = os.path.join(
        PROJECT_ROOT, "results", "phase3_estimation", f"truth_motif_{motif_name}.npz"
    )
    data = load_truth_or_fail(PROJECT_ROOT, motif_name)
    
    t = data["t"]
    x_true = data["x_true"]
    u_D = data["u_D"]
    B_true = data["B"]
    dt = float(data["dt"])

    # ------------------------------------------------------------
    # Generate noisy B-only measurements
    # ------------------------------------------------------------
    y_true = B_true.reshape(-1, 1)

    sigma_rel_B = 0.05  # 5% relative noise
    rng = np.random.default_rng(seed=123)
    noise = sigma_rel_B * B_true * rng.normal(size=t.size)
    y_meas = y_true + noise.reshape(-1, 1)

    # ------------------------------------------------------------
    # Filter setup (same initial conditions for fair comparison)
    # ------------------------------------------------------------
    nx = 6
    Q = 1e-5 * np.eye(nx)
    R = np.array([[(0.05) ** 2]])  # variance for B

    ekf = bd_ekf.BdEKF(Q=Q, R=R, measurement_option=meas_opt)
    ukf = bd_ukf.BdUKF(Q=Q, R=R, measurement_option=meas_opt)

    x0_hat = np.array([0.8, 0.1, 0.1, 0.1, 4.5, 0.1])
    P0_hat = np.diag([0.5, 0.2, 0.2, 0.2, 0.5, 0.2]) ** 2

    # ------------------------------------------------------------
    # Run EKF and UKF
    # ------------------------------------------------------------
    x_hat_ekf, _ = run_filter(ekf, t, u_D, y_meas, dt, x0_hat, P0_hat)
    x_hat_ukf, _ = run_filter(ukf, t, u_D, y_meas, dt, x0_hat, P0_hat)

    # ------------------------------------------------------------
    # Metrics (RMS error in Z)
    # ------------------------------------------------------------
    Z_true = x_true[:, 0]
    Z_ekf = x_hat_ekf[:, 0]
    Z_ukf = x_hat_ukf[:, 0]

    rms_ekf = np.sqrt(np.mean((Z_true - Z_ekf) ** 2))
    rms_ukf = np.sqrt(np.mean((Z_true - Z_ukf) ** 2))

    print("=" * 80)
    print(f"Motif: {motif_name}, Measurement: {meas_opt} (B-only)")
    print(f"RMS(Z) EKF = {rms_ekf:.3e}")
    print(f"RMS(Z) UKF = {rms_ukf:.3e}")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t, Z_true, "k-", label="Z true")
    plt.plot(t, Z_ekf, "r--", label=f"Z EKF (RMS={rms_ekf:.2e})")
    plt.plot(t, Z_ukf, "b-.", label=f"Z UKF (RMS={rms_ukf:.2e})")
    plt.xlabel("t [h]")
    plt.ylabel("Z")
    plt.title("EKF vs UKF (B-only) on sin_12h_big")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
