"""
Compare EKF performance for different measurement sets using a fixed
dilution motif (step_24h).

Measurement sets:
    h_BNW = [B, N, W]^T
    h_BN  = [B, N]^T
    h_B   = [B]
    h_NW  = [N, W]^T
    h_N   = [N]
    h_W   = [W]

For each measurement set:
    - Load "truth" data from results/phase3_estimation/truth_motif_step_24h.npz
    - Generate noisy measurements consistent with that set
    - Run the BdEKF with the corresponding measurement_option
    - Compute RMS error in Z
    - Plot Z_true vs Z_hat

Outputs:
    - One figure with Z(t) for all measurement sets (subplots)
    - One bar plot with RMS(Z error) vs measurement set
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from Utility.bd_io import ensure_dir, load_truth_or_fail


from Utility import bd_ekf

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


# ----------------------------------------------------------------------
# Helper: build noisy measurements and R for a given measurement set
# ----------------------------------------------------------------------
def build_measurements_and_R(
    meas_opt: str,
    B_true: np.ndarray,
    N_true: np.ndarray,
    W_true: np.ndarray,
    rng: np.random.Generator,
):
    """
    Given the true trajectories of B, N, W, generate noisy measurements
    for the chosen measurement option, and return (y_meas, R).

    Noise model:
        - 5% relative on B and W
        - 0.05 mM absolute on N
    """
    T_steps = B_true.size

    sigma_rel_B = 0.05
    sigma_abs_N = 0.05
    sigma_rel_W = 0.05

    if meas_opt == "h_BNW":
        ny = 3
        y_true = np.vstack((B_true, N_true, W_true)).T
        noise = np.zeros_like(y_true)
        noise[:, 0] = sigma_rel_B * B_true * rng.normal(size=T_steps)
        noise[:, 1] = sigma_abs_N * rng.normal(size=T_steps)
        noise[:, 2] = sigma_rel_W * W_true * rng.normal(size=T_steps)
        R = np.diag([
            (sigma_rel_B * np.maximum(np.mean(B_true), 1e-3))**2,
            (sigma_abs_N)**2,
            (sigma_rel_W * np.maximum(np.mean(W_true), 1e-3))**2,
        ])

    elif meas_opt == "h_BN":
        ny = 2
        y_true = np.vstack((B_true, N_true)).T
        noise = np.zeros_like(y_true)
        noise[:, 0] = sigma_rel_B * B_true * rng.normal(size=T_steps)
        noise[:, 1] = sigma_abs_N * rng.normal(size=T_steps)
        R = np.diag([
            (sigma_rel_B * np.maximum(np.mean(B_true), 1e-3))**2,
            (sigma_abs_N)**2,
        ])

    elif meas_opt == "h_B":
        ny = 1
        y_true = B_true.reshape(-1, 1)
        noise = np.zeros_like(y_true)
        noise[:, 0] = sigma_rel_B * B_true * rng.normal(size=T_steps)
        R = np.array([[(sigma_rel_B * np.maximum(np.mean(B_true), 1e-3))**2]])

    elif meas_opt == "h_NW":
        ny = 2
        y_true = np.vstack((N_true, W_true)).T
        noise = np.zeros_like(y_true)
        noise[:, 0] = sigma_abs_N * rng.normal(size=T_steps)
        noise[:, 1] = sigma_rel_W * W_true * rng.normal(size=T_steps)
        R = np.diag([
            (sigma_abs_N)**2,
            (sigma_rel_W * np.maximum(np.mean(W_true), 1e-3))**2,
        ])

    elif meas_opt == "h_N":
        ny = 1
        y_true = N_true.reshape(-1, 1)
        noise = np.zeros_like(y_true)
        noise[:, 0] = sigma_abs_N * rng.normal(size=T_steps)
        R = np.array([[sigma_abs_N**2]])

    elif meas_opt == "h_W":
        ny = 1
        y_true = W_true.reshape(-1, 1)
        noise = np.zeros_like(y_true)
        noise[:, 0] = sigma_rel_W * W_true * rng.normal(size=T_steps)
        R = np.array([[(sigma_rel_W * np.maximum(np.mean(W_true), 1e-3))**2]])

    else:
        raise ValueError(f"Unknown measurement option: {meas_opt}")

    y_meas = y_true + noise
    return y_meas, R


# ----------------------------------------------------------------------
# Helper: run EKF for a given measurement set (fixed motif step_24h)
# ----------------------------------------------------------------------
def run_ekf_for_meas(meas_opt: str, seed: int = 123):
    """
    Run BdEKF with a given measurement set and motif step_24h.

    Returns
    -------
    t : (T,) ndarray
    x_true : (T, 6) ndarray
    x_hat  : (T, 6) ndarray
    rms_Z  : float
        RMS error in Z.
    """
    motif_name = "step_24h"

    truth_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "phase3_estimation",
        f"truth_motif_{motif_name}.npz",
    )
    data = load_truth_or_fail(PROJECT_ROOT, motif_name)

    t = data["t"]
    x_true = data["x_true"]
    u_D = data["u_D"]
    B_true = data["B"]
    N_true = data["N"]
    W_true = data["W"]
    dt = float(data["dt"])

    T_steps = t.size
    nx = 6

    rng = np.random.default_rng(seed=seed)

    # Build noisy measurements and measurement covariance R
    y_meas, R = build_measurements_and_R(meas_opt, B_true, N_true, W_true, rng)

    # Process noise covariance (tunable)
    Q = 1e-5 * np.eye(nx)

    # EKF object
    ekf = bd_ekf.BdEKF(Q=Q, R=R, measurement_option=meas_opt)

    x_hat = np.zeros((T_steps, nx))
    P_hat = np.zeros((T_steps, nx, nx))

    # Initial guess (same for all meas. sets)
    x_hat[0, :] = np.array([0.8, 0.1, 0.1, 0.1, 4.5, 0.1])
    P_hat[0, :, :] = np.diag([0.5, 0.2, 0.2, 0.2, 0.5, 0.2])**2

    # EKF loop
    for k in range(T_steps - 1):
        u_k = np.array([u_D[k, 0]])

        # Predict
        x_pred, P_pred = ekf.predict(x_hat[k, :], P_hat[k, :, :], u_k, dt)

        # Update with measurement at k+1
        y_k1 = y_meas[k + 1, :].reshape(-1)
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_k1)

        x_hat[k + 1, :] = x_upd
        P_hat[k + 1, :, :] = P_upd

    Z_true = x_true[:, 0]
    Z_hat = x_hat[:, 0]
    err_Z = Z_true - Z_hat
    rms_Z = np.sqrt(np.mean(err_Z**2))

    return t, x_true, x_hat, rms_Z


# ----------------------------------------------------------------------
# Main: compare measurement sets
# ----------------------------------------------------------------------
def main():
    meas_options = ["h_BNW", "h_BN", "h_B", "h_NW", "h_N", "h_W"]
    meas_labels = {
        "h_BNW": "B,N,W",
        "h_BN":  "B,N",
        "h_B":   "B",
        "h_NW":  "N,W",
        "h_N":   "N",
        "h_W":   "W",
    }

    results = {}
    rms_list = []

    # Run EKF for each measurement set
    for meas in meas_options:
        print("=" * 80)
        print(f"Running EKF for measurement set: {meas}")
        t, x_true, x_hat, rms_Z = run_ekf_for_meas(meas, seed=123)
        results[meas] = (t, x_true, x_hat, rms_Z)
        rms_list.append(rms_Z)
        print(f"RMS error in Z for {meas}: {rms_Z:.3e}")

    # ------------------------------------------------------------------
    # Plot Z trajectories (true vs EKF) for all measurement sets
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 8))

    for i, meas in enumerate(meas_options, start=1):
        t, x_true, x_hat, rms_Z = results[meas]
        Z_true = x_true[:, 0]
        Z_hat = x_hat[:, 0]

        plt.subplot(3, 2, i)
        plt.plot(t, Z_true, "k-", label="Z true")
        plt.plot(t, Z_hat, "r--", label="Z EKF")
        plt.ylabel("Z")
        plt.title(f"meas = {meas_labels[meas]} (RMS={rms_Z:.2e})")
        if i == 1:
            plt.legend(fontsize=8)

    plt.xlabel("t [h]")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Bar plot: RMS(Z) vs measurement set
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    x_pos = np.arange(len(meas_options))
    plt.bar(x_pos, rms_list)
    plt.xticks(x_pos, [meas_labels[m] for m in meas_options])
    plt.ylabel("RMS error in Z")
    plt.title("EKF performance vs measurement set (motif step_24h)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
