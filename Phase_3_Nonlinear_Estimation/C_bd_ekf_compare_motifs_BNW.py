# Phase_3_Nonlinear_Estimation/C_bd_ekf_compare_motifs_BNW.py

"""
Compare EKF performance for different dilution motifs using measurement set h_BNW = [B, N, W]^T.

For each motif:
    - Load "truth" data from results/phase3_estimation/truth_motif_<name>.npz
    - Generate noisy measurements y = [B, N, W] + noise
    - Run the BdEKF with measurement_option="h_BNW"
    - Compute RMS error in Z
    - Plot Z_true vs Z_hat

Outputs:
    - One figure with Z(t) for all motifs
    - One bar plot with RMS(Z error) for each motif
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

import bd_ekf  # BdEKF class


# ----------------------------------------------------------------------
# Helper: run EKF for a given motif
# ----------------------------------------------------------------------
def run_ekf_for_motif(motif_name: str, seed: int = 123):
    """
    Run BdEKF with measurement set h_BNW for the given motif.

    Returns
    -------
    t : (T,) ndarray
    x_true : (T, 6) ndarray
    x_hat  : (T, 6) ndarray
    rms_Z  : float
        Root-mean-square error for Z over the entire horizon.
    """
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

    T_steps = t.size
    nx = 6
    ny = 3   # [B, N, W]

    # -----------------------------
    # Generate noisy measurements
    # -----------------------------
    y_true = np.vstack((B_true, N_true, W_true)).T

    sigma_rel_B = 0.05
    sigma_abs_N = 0.05
    sigma_rel_W = 0.05

    rng = np.random.default_rng(seed=seed)

    noise = np.zeros_like(y_true)
    noise[:, 0] = sigma_rel_B * B_true * rng.normal(size=T_steps)
    noise[:, 1] = sigma_abs_N * rng.normal(size=T_steps)
    noise[:, 2] = sigma_rel_W * W_true * rng.normal(size=T_steps)

    y_meas = y_true + noise

    # -----------------------------
    # EKF setup
    # -----------------------------
    Q = 1e-5 * np.eye(nx)
    R = np.diag([
        (0.05)**2,  # B
        (0.05)**2,  # N
        (0.05)**2,  # W
    ])

    ekf = bd_ekf.BdEKF(Q=Q, R=R, measurement_option="h_BNW")

    x_hat = np.zeros((T_steps, nx))
    P_hat = np.zeros((T_steps, nx, nx))

    # Initial guess (same for all motifs so comparison is fair)
    x_hat[0, :] = np.array([0.8, 0.1, 0.1, 0.1, 4.5, 0.1])
    P_hat[0, :, :] = np.diag([0.5, 0.2, 0.2, 0.2, 0.5, 0.2])**2

    # -----------------------------
    # EKF time loop
    # -----------------------------
    for k in range(T_steps - 1):
        u_k = np.array([u_D[k, 0]])   # scalar dilution as 1D array

        # Predict
        x_pred, P_pred = ekf.predict(x_hat[k, :], P_hat[k, :, :], u_k, dt)

        # Update using measurement at k+1
        y_k1 = y_meas[k + 1, :]
        x_upd, P_upd = ekf.update(x_pred, P_pred, y_k1)

        x_hat[k + 1, :] = x_upd
        P_hat[k + 1, :, :] = P_upd

    # RMS error in Z
    Z_true = x_true[:, 0]
    Z_hat = x_hat[:, 0]
    err_Z = Z_true - Z_hat
    rms_Z = np.sqrt(np.mean(err_Z**2))

    return t, x_true, x_hat, rms_Z


# ----------------------------------------------------------------------
# Main: compare motifs
# ----------------------------------------------------------------------
def main():
    motifs = ["const_0p12", "step_24h", "sin_12h"]
    motif_labels = {
        "const_0p12": "Constant D=0.12 h$^{-1}$",
        "step_24h": "Step at 24 h",
        "sin_12h": "Sinusoidal (12 h period)",
    }

    results = {}
    rms_list = []

    # Run EKF for each motif
    for motif in motifs:
        print("=" * 80)
        print(f"Running EKF for motif: {motif}")
        t, x_true, x_hat, rms_Z = run_ekf_for_motif(motif, seed=123)
        results[motif] = (t, x_true, x_hat, rms_Z)
        rms_list.append(rms_Z)
        print(f"RMS error in Z for motif {motif}: {rms_Z:.3e}")

    # ------------------------------------------------------------------
    # Plot Z trajectories (true vs EKF) for all motifs
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for i, motif in enumerate(motifs, start=1):
        t, x_true, x_hat, rms_Z = results[motif]
        Z_true = x_true[:, 0]
        Z_hat = x_hat[:, 0]

        plt.subplot(3, 1, i)
        plt.plot(t, Z_true, "k-", label="Z true")
        plt.plot(t, Z_hat, "r--", label="Z EKF")
        plt.ylabel("Z")
        plt.title(f"{motif_labels[motif]} (RMS error={rms_Z:.2e})")
        if i == 1:
            plt.legend()

    plt.xlabel("t [h]")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Bar plot of RMS(Z) vs motif
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    x_pos = np.arange(len(motifs))
    plt.bar(x_pos, rms_list)
    plt.xticks(x_pos, [motif_labels[m] for m in motifs], rotation=20)
    plt.ylabel("RMS error in Z")
    plt.title("EKF performance for different dilution motifs (meas h_BNW)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
