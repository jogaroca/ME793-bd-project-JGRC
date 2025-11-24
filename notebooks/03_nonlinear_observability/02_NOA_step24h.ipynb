# --- Path setup so we can import the local "bd" package ---

import os
import sys

# Assume this notebook lives in
# ME793-bd-project-JGRC/notebooks/03_nonlinear_observability
THIS_DIR = os.path.abspath(os.getcwd())
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# If your package is still named "db", change this import accordingly:
import bd

from bd import model, controls, simulator, observability

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Open-loop simulation with step-change dilution at 24 h ---

T_final = 48.0
dt = 0.1
t_grid = np.arange(0.0, T_final + dt, dt)

x0 = model.default_initial_state()

t_sim, X, U, Y = simulator.simulate_bd_odeint(
    x0=x0,
    t_grid=t_grid,
    uD_fun=controls.uD_step_24h,
    theta_local=bd.theta,
    noisy=False,
    rng_seed=123,
)

print("Simulation finished.")
print("t_sim shape:", t_sim.shape)
print("X shape:", X.shape)
print("U shape:", U.shape)
print("Y shape:", Y.shape)

# --- Build PyBounds simulator and empirical observability matrix (step control) ---

dt_sim = float(t_sim[1] - t_sim[0])
pb_sim = simulator.make_pybounds_simulator(dt=dt_sim)

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

# --- Estimate measurement-noise variances and build sliding Fisher observability ---

noise_vars = observability.estimate_measurement_noise_variances_from_outputs(Y)

SFO, EV_df = observability.build_sliding_fisher_observability(
    SEOM,
    measurement_noise_variances=noise_vars,
    states=model.STATE_NAMES,
    sensors=MEASUREMENT_NAMES,
    time_steps=None,
    lam=1e-8,
    window_size=window_size,
)

EV_df.head()

# --- Plot the minimum error variance per state over time (step control) ---

fig, ax = plt.subplots(figsize=(8, 5))

for state in EV_df.columns:
    ax.plot(SFO.time, EV_df[state], label=state)

ax.set_yscale("log")
ax.set_xlabel("Time [h]")
ax.set_ylabel("Minimum error variance (log scale)")
ax.set_title("Sliding Fisher observability - step control at 24 h")
ax.legend()
fig.tight_layout()
plt.show()
