"""
EKF example for the Bd chemostat:
- Motif: const_0p12
- Measurement set: h_BNW = [B, N, W]^T

Uses:
    - Truth data from results/phase3_estimation/truth_motif_const_0p12.npz
    - BdEKF class from Utility/bd_ekf.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths and imports
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UTILITY_DIR = os.path.join(PROJECT_ROOT, "Utility")

if UTILITY_DIR not in sys.path:
    sys.path.append(UTILITY_DIR)

import bd_ekf  # noqa: E402


def main():
    motif_name = "const_0p12"

    # ------------------------------------------------------------------
    # Load "truth" data
    # ------------------------------------------------------------------
    truth_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "phase3_estimation",
        f"truth_motif_{motif_name}.npz",
    )
    data = np.load(truth_path)

    t = data["t"]                # (T,)
    x_true = data["x_true"]      # (T, 6)
    u_D = data["u_D"]            # (T, 1)
    B_true = data["B"]           # (T,)
    N_true = data["N"]           # (T,)
    W_true = data["W"]           # (T,)
    dt = float(data["dt"])
    x0_nominal = data["x0"]      # (6,)

    T_steps = t.size
    nx = 6
    ny = 3   # B, N, W

    # ------------------------------------------------------------------
    # Generate noisy measurements y_k = [B, N, W] + noise
    # Noise model: 5% relative on B and W, 0.05 mM on N
    # ------------------------------------------------------------------
    y_true = np.vstack((B_true, N_true, W_true)).T

    sigma_rel_B = 0.05
    sigma_abs_N = 0.05
    sigma_rel_W = 0.05

    rng = np.random.default_rng(seed=123)

    noise = np.zeros_like(y_true)
    noise[:, 0] = sigma_rel_B * B_true * rng.normal(size=T_steps)
    noise[:, 1] = sigma_abs_N * rng.normal(size=T_steps)
    noise[:, 2] = sigma_rel_W * W_true * rng.normal(size=T_steps)

    y_meas = y_true + noise

    # ------------------------------------------------------------------
    # EKF setup
    # ------------------------------------------------------------------
    # Process noise covariance (tunable)
    Q = 1e-5 * np.eye(nx)

    # Measurement noise covariance (aprox. constante)
    R = np.diag([
        (0.05)**2,  # B
        (0.05)**2,  # N
        (0.05)**2,  # W
    ])

    ekf = bd_ekf.BdEKF(Q=Q, R=R, measurement_option="h_BNW")

    # Allocate arrays for estimates
    x_hat = np.zeros((T_steps, nx))
    P_hat = np.zeros((T_steps, nx, nx))

    # Initial guess (puedes desviarte un poco del nominal si quieres)
    x_hat[0, :] = np.array([0.8, 0.1, 0.1, 0.1, 4.5, 0.1])
    P_hat[0, :, :] = np.diag([0.5, 0.2, 0.2, 0.2, 0.5, 0.2])**2

    # ------------------------------------------------------------------
    # Run EKF through the time series
    # ------------------------------------------------------------------
    for k in range(T_steps - 1):
        u_k = np.array([u_D[k, 0]])   # scalar control as 1D array

        # Predict
        x_pred, P_pred = ekf.predict(x_hat[k, :], P_hat[k, :, :], u_k, dt)

        # Update (usamos la medición en k+1)
        y_k1 = y_meas[k + 1, :]
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_k1)

        x_hat[k + 1, :] = x_upd
        P_hat[k + 1, :, :] = P_upd

    # ------------------------------------------------------------------
    # Plots: Z (state of interest) and B,N,W
    # ------------------------------------------------------------------
    Z_true = x_true[:, 0]
    Z_hat = x_hat[:, 0]

    plt.figure(figsize=(6, 4))
    plt.plot(t, Z_true, "k-", label="Z true")
    plt.plot(t, Z_hat, "r--", label="Z EKF")
    plt.xlabel("t [h]")
    plt.ylabel("Zoospores Z")
    plt.title(f"EKF estimate of Z (motif={motif_name}, meas=h_BNW)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # B, N, W estimates vs truth
    B_hat = x_hat[:, 0] + x_hat[:, 1] + x_hat[:, 2] + x_hat[:, 3]
    N_hat = x_hat[:, 4]
    W_hat = x_hat[:, 5]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(t, B_true, "k-", label="B true")
    plt.plot(t, B_hat, "r--", label="B EKF")
    plt.xlabel("t [h]")
    plt.ylabel("B")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(t, N_true, "k-", label="N true")
    plt.plot(t, N_hat, "r--", label="N EKF")
    plt.xlabel("t [h]")
    plt.ylabel("N [mM]")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(t, W_true, "k-", label="W true")
    plt.plot(t, W_hat, "r--", label="W EKF")
    plt.xlabel("t [h]")
    plt.ylabel("W")
    plt.legend()

    plt.suptitle(f"EKF estimates of B, N, W (motif={motif_name}, meas=h_BNW)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
