"""
Phase 2 (add-on): Sliding-window minimum error variance time series.

This script computes, for each trajectory motif and measurement option:
  1) Output sensitivities S(t) = dy(t)/dx0 via finite differences
  2) Sliding-window empirical Fisher information matrices
  3) The (regularized) inverse Fisher information and its diagonal,
     interpreted as the minimum error variance (Cramer-Rao lower bound)
     attainable by any unbiased estimator.

Rationale
---------
A time series of the smallest eigenvalue of the observability Gramian (or Fisher
information) does *not* identify which states drive observability. In contrast,
MEV(t) provides a per-state quantity that directly indicates which states are
expected to be well/poorly observable at each time.

Outputs
-------
Saves, for each (motif, measurement option):
  - results/phase2_empirical/mev_{motif}_{meas}.npz
  - results/phase2_empirical/mev_{motif}_{meas}.png                  (MEV only, 3x2)
  - results/phase2_empirical/mev_{motif}_{meas}_lesson8B.png         (state + MEV, 6x2)

The .npz contains t_sim, t_mid, mev (n_windows x n_states), y_nominal, and metadata.

Notes
-----
* The Fisher information uses diagonal measurement noise covariance R.
  If you do not know noise levels, setting R=I (default) is still useful
  for relative comparisons across motifs/measurement sets.
* Regularization: we invert F + lam*I and apply a relative eigenvalue
  threshold for numerical robustness.
"""

import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from Utility import bd_chemostat as bd
from Utility import empirical_observability as eo
from Utility import bd_motifs as motifs


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
T_FINAL = 48.0   # [h]
DT = 0.05        # [h]

# Nominal initial condition [Z, S1, S2, S3, N, W]
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)

# Motifs to analyze
TRAJECTORY_MOTIFS: List[Dict[str, object]] = [
    {"name": "const_0p12",   "x0": X0_NOMINAL, "D_fun": motifs.D_const_0p12},
    {"name": "step_24h",     "x0": X0_NOMINAL, "D_fun": motifs.D_step_24h},
    {"name": "sin_12h",      "x0": X0_NOMINAL, "D_fun": motifs.D_sin_12h},
    {"name": "sin_12h_big",  "x0": X0_NOMINAL, "D_fun": motifs.D_sin_12h_big},
]

# Measurement options
MEASUREMENT_OPTIONS = [
    "h_BNW",
    "h_BN",
    "h_B",
    "h_NW",
    "h_N",
    "h_W",
]

# Finite-difference perturbation sizes
EPS_REL = 1e-4
EPS_ABS = 1e-6

# Sliding window parameters
WINDOW_HOURS = 6.0     # window length in hours
STEP_HOURS = 0.5       # stride in hours

# Fisher/CRLB inversion parameters
RIDGE_LAM = 1e-8
EIG_TOL = 1e-12

# Measurement noise variances (diagonal R). Used as R^{-1} in Fisher.
# If you want: set different variances per sensor (B, N, W).
DEFAULT_NOISE_VARS = {
    "B": 1.0,
    "N": 1.0,
    "W": 1.0,
}

# State names (order must match bd.F)
STATE_NAMES = ("Z", "S1", "S2", "S3", "N", "W")

# If True, recompute even if results exist
FORCE_RECOMPUTE = False


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def build_simulate_outputs_fn(f_handle, h_handle, t_final: float, dt: float, D_fun):
    """Closure that simulates y(t) given x0."""

    def simulate_outputs_fn(x0_vec: np.ndarray) -> np.ndarray:
        _, _, _, y_sim = bd.simulate_bd(
            f=f_handle,
            h=h_handle,
            tsim_length=t_final,
            dt=dt,
            x0=x0_vec,
            D=0.0,
            D_fun=D_fun,
        )
        return y_sim

    return simulate_outputs_fn


def simulate_nominal_state_trajectory(f_handle, h_handle, t_final: float, dt: float, x0: np.ndarray, D_fun):
    """
    Simulate the full nominal trajectory (t, x(t), u(t), y(t)) for plotting states.
    Tries to be robust to slight differences in simulate_bd return ordering.
    """
    sim_out = bd.simulate_bd(
        f=f_handle,
        h=h_handle,
        tsim_length=t_final,
        dt=dt,
        x0=x0,
        D=0.0,
        D_fun=D_fun,
    )

    # Expected: (t, x, u, y). If not, we fail loudly with a helpful message.
    if not isinstance(sim_out, (tuple, list)) or len(sim_out) < 4:
        raise RuntimeError("bd.simulate_bd did not return a 4-tuple (t, x, u, y). Please check its signature.")

    t_sim, x_sim, u_sim, y_sim = sim_out[0], sim_out[1], sim_out[2], sim_out[3]
    return np.asarray(t_sim), np.asarray(x_sim), np.asarray(u_sim), np.asarray(y_sim)


def measurement_noise_vars_for_option(meas_opt: str) -> np.ndarray:
    """Map measurement option name to a per-output variance vector."""
    h_obj = bd.H(measurement_option=meas_opt)
    # We need a dummy state/input to retrieve sensor names
    sensor_names = h_obj.h(np.zeros(6), [0.0], return_measurement_names=True)
    return np.array([DEFAULT_NOISE_VARS[str(n)] for n in sensor_names], dtype=float)


def plot_mev_timeseries(t_mid: np.ndarray, mev: np.ndarray, title: str, out_png: str):
    """Create a compact per-state MEV plot (6 subplots)."""
    mev = np.asarray(mev, dtype=float)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 8), sharex=True)
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        ax.semilogy(t_mid, np.maximum(mev[:, i], 1e-30))
        ax.set_ylabel(f"MEV({STATE_NAMES[i]})")
        ax.grid(True, which="both", alpha=0.3)

    axes[-2].set_xlabel("t [h]")
    axes[-1].set_xlabel("t [h]")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_states_and_mev(
    t_sim: np.ndarray,
    x_nominal: np.ndarray,
    t_mid: np.ndarray,
    mev: np.ndarray,
    title: str,
    out_png: str,
):
    """
    Lesson-8B style plot: 6x2 grid.
      Left: x_i(t) nominal state trajectory
      Right: MEV_i(t) (semilogy)
    """
    x_nominal = np.asarray(x_nominal, dtype=float)
    mev = np.asarray(mev, dtype=float)

    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 14), sharex="col")

    for i, name in enumerate(STATE_NAMES):
        # Left column: state trajectory
        axL = axes[i, 0]
        axL.plot(t_sim, x_nominal[:, i])
        axL.set_ylabel(f"{name}")
        axL.grid(True, alpha=0.3)

        # Right column: MEV time series
        axR = axes[i, 1]
        axR.semilogy(t_mid, np.maximum(mev[:, i], 1e-30))
        axR.set_ylabel(f"MEV({name})")
        axR.grid(True, which="both", alpha=0.3)

    axes[-1, 0].set_xlabel("t [h]")
    axes[-1, 1].set_xlabel("t [h]")
    axes[0, 0].set_title("State trajectory")
    axes[0, 1].set_title("Minimum error variance (CRLB)")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Dynamics
    f_obj = bd.F()
    f = f_obj.f

    window_steps = int(round(WINDOW_HOURS / DT))
    step_steps = int(round(STEP_HOURS / DT))
    if window_steps < 1:
        raise ValueError("WINDOW_HOURS too small relative to DT.")
    if step_steps < 1:
        raise ValueError("STEP_HOURS too small relative to DT.")

    for motif in TRAJECTORY_MOTIFS:
        motif_name = str(motif["name"])
        x0_val = np.asarray(motif["x0"], dtype=float).reshape(-1)
        D_fun = motif["D_fun"]

        print("=" * 80)
        print(f"Trajectory motif: {motif_name}")

        for meas_opt in MEASUREMENT_OPTIONS:
            print("-" * 80)
            print(f"Measurement option: {meas_opt}")

            out_npz = os.path.join(RESULTS_DIR, f"mev_{motif_name}_{meas_opt}.npz")
            out_png = os.path.join(RESULTS_DIR, f"mev_{motif_name}_{meas_opt}.png")
            out_png_l8b = os.path.join(RESULTS_DIR, f"mev_{motif_name}_{meas_opt}_lesson8B.png")

            if (
                (not FORCE_RECOMPUTE)
                and os.path.exists(out_npz)
                and os.path.exists(out_png)
                and os.path.exists(out_png_l8b)
            ):
                print(f"Found cached outputs. Skipping: {out_npz}")
                continue

            # Measurements
            h_obj = bd.H(measurement_option=meas_opt)
            h = h_obj.h
            sensor_names = h(np.zeros(6), [0.0], return_measurement_names=True)

            # Nominal state trajectory for plotting
            t_sim, x_nominal, u_nominal, y_nominal_check = simulate_nominal_state_trajectory(
                f_handle=f,
                h_handle=h,
                t_final=T_FINAL,
                dt=DT,
                x0=x0_val,
                D_fun=D_fun,
            )

            # Simulate y(t) given x0 for sensitivities
            simulate_outputs_fn = build_simulate_outputs_fn(
                f_handle=f,
                h_handle=h,
                t_final=T_FINAL,
                dt=DT,
                D_fun=D_fun,
            )

            # Output sensitivities
            y_nominal, S = eo.empirical_output_sensitivities(
                simulate_outputs_fn=simulate_outputs_fn,
                x0=x0_val,
                eps_rel=EPS_REL,
                eps_abs=EPS_ABS,
            )

            # Defensive: align t_sim to y_nominal length if needed
            if y_nominal.shape[0] != t_sim.size:
                t_sim = np.arange(0.0, DT * y_nominal.shape[0], DT)

                # Also align x_nominal if lengths mismatch (rare; depends on simulate_bd implementation)
                if x_nominal.shape[0] != t_sim.size:
                    min_len = min(x_nominal.shape[0], t_sim.size)
                    t_sim = t_sim[:min_len]
                    x_nominal = x_nominal[:min_len, :]
                    y_nominal = y_nominal[:min_len, :]
                    S = S[:min_len, :, :]

            # Noise variances for this measurement set
            noise_vars = measurement_noise_vars_for_option(meas_opt)

            # MEV time series
            t_mid, mev = eo.minimum_error_variance_timeseries(
                S=S,
                dt=DT,
                window_steps=window_steps,
                step=step_steps,
                measurement_noise_vars=noise_vars,
                lam=RIDGE_LAM,
                tol=EIG_TOL,
            )

            np.savez(
                out_npz,
                t_sim=t_sim,
                t_mid=t_mid,
                mev=mev,
                y_nominal=y_nominal,
                x_nominal=x_nominal,
                u_nominal=u_nominal,
                motif_name=motif_name,
                measurement_option=meas_opt,
                state_names=np.array(STATE_NAMES, dtype=object),
                sensor_names=np.array(sensor_names, dtype=object),
                dt=DT,
                window_hours=WINDOW_HOURS,
                step_hours=STEP_HOURS,
                ridge_lam=RIDGE_LAM,
                eig_tol=EIG_TOL,
                measurement_noise_vars=noise_vars,
            )
            print(f"Saved MEV results to: {out_npz}")

            title = (
                f"Minimum error variance (CRLB) via sliding Fisher\n"
                f"motif={motif_name}, meas={meas_opt}, window={WINDOW_HOURS}h, step={STEP_HOURS}h"
            )

            # Original compact MEV-only figure
            plot_mev_timeseries(t_mid, mev, title, out_png)
            print(f"Saved figure to: {out_png}")

            # Lesson-8B style: state + MEV
            plot_states_and_mev(
                t_sim=t_sim,
                x_nominal=x_nominal,
                t_mid=t_mid,
                mev=mev,
                title=title,
                out_png=out_png_l8b,
            )
            print(f"Saved Lesson-8B style figure to: {out_png_l8b}")


if __name__ == "__main__":
    main()
