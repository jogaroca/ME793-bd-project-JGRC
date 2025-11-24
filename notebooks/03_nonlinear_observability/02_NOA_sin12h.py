# ---------------------------------------------------------------------------
# Nonlinear Observability Analysis (NOA) for the Bd chemostat
# Control motif: sinusoidal dilution u_D(t) = 0.15 + 0.05 sin(2π t / 12)
#
# This script mirrors the structure of 01_NOA_const012.py but uses the
# sinusoidal control motif defined in bd.controls.uD_sin_12h.
# ---------------------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Path setup so we can import the local "bd" and "utils" packages ---

# Assume this script lives in
# ME793-bd-project-JGRC/notebooks/03_nonlinear_observability
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import bd
from bd import model, controls, simulator, observability
from bd.measurements import MEASUREMENT_NAMES

from utils.plot_utils import plot_tme  # from src/utils/plot_utils.py

# ---------------------------------------------------------------------------
# 1) Open-loop simulation with sinusoidal dilution
# ---------------------------------------------------------------------------

# Time grid for NOA
T_final = 48.0  # hours
dt = 0.1        # hours
t_grid = np.arange(0.0, T_final + dt, dt)

# Initial condition (same as in the ME793 simulation script)
x0 = model.default_initial_state()

# Simulate Bd dynamics with u_D(t) = 0.15 + 0.05 sin(2π t / 12)
t_sim, X, U, Y = simulator.simulate_bd_odeint(
    x0=x0,
    t_grid=t_grid,
    uD_fun=controls.uD_sin_12h,
    theta_local=bd.theta,
    noisy=False,          # NOA usually uses noise-free outputs for R estimation
    rng_seed=123,
)

print("Simulation finished.")
print("t_sim shape:", t_sim.shape)
print("X shape:", X.shape)
print("U shape:", U.shape)
print("Y shape:", Y.shape)

# ---------------------------------------------------------------------------
# 2) Quick sanity check of the outputs B, N, W
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

for i, name in enumerate(MEASUREMENT_NAMES):
    plot_tme(
        t_sim,
        Y[:, i],
        measured=None,
        estimated=None,
        ax=axes[i],
        label_var=name,
    )

axes[-1].set_xlabel("Time [h]")
fig.suptitle("Outputs B, N, W under sinusoidal dilution u_D(t)", y=0.94)
fig.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 3) Build PyBounds simulator and empirical observability matrix
# ---------------------------------------------------------------------------

dt_sim = float(t_sim[1] - t_sim[0])
pb_sim = simulator.make_pybounds_simulator(dt=dt_sim)

# Sliding window size (number of time steps per window).
# You can experiment with different values (e.g. 5, 10, 20).
window_size = 10

SEOM, O_sliding = observability.build_sliding_empirical_observability_matrix(
    simulator=pb_sim,
    t_sim=t_sim,
    X=X,
    U=U,
    window_size=window_size,
    eps=1e-4,
)

print("Number of sliding windows:", len(SEOM.O_df_sliding))
print("Example O_df_sliding[0].shape:", SEOM.O_df_sliding[0].shape)

# ---------------------------------------------------------------------------
# 4) Estimate measurement-noise variances and build sliding Fisher observability
# ---------------------------------------------------------------------------

noise_vars = observability.estimate_measurement_noise_variances_from_outputs(Y)

SFO, EV_df = observability.build_sliding_fisher_observability(
    SEOM,
    measurement_noise_variances=noise_vars,
    states=model.STATE_NAMES,
    sensors=MEASUREMENT_NAMES,
    time_steps=None,      # use all steps in each window
    lam=1e-8,
    window_size=window_size,
)

print("EV_df shape:", EV_df.shape)
print(EV_df.head())

# ---------------------------------------------------------------------------
# 5) Plot the minimum error variance per state over time
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

# EV_df has one column per state and an index aligned with SFO.time
for state in EV_df.columns:
    ax.plot(SFO.time, EV_df[state], label=state)

ax.set_yscale("log")
ax.set_xlabel("Time [h]")
ax.set_ylabel("Minimum error variance (log scale)")
ax.set_title("Sliding Fisher observability - sinusoidal u_D(t)")
ax.legend()
fig.tight_layout()
plt.show()
