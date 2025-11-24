# --- Path setup and imports ---

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.abspath(os.getcwd())
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import bd
from bd import model, controls, simulator, observability
from bd.measurements import MEASUREMENT_NAMES

# For plots
plt.rcParams["figure.figsize"] = (8, 5)

# --- Control motifs to compare in the NOA study ---

control_motifs = {
    "const_0.12": controls.uD_const_012,
    "step_24h":   controls.uD_step_24h,
    "sin_12h":    controls.uD_sin_12h,
}

T_final = 48.0
dt = 0.1
t_grid = np.arange(0.0, T_final + dt, dt)

x0 = model.default_initial_state()

# --- Loop over control motifs: simulate, compute NOA metrics ---

results = []  # will hold dicts (motif, state, EV_mean, EV_median, EV_min)

for motif_name, uD_fun in control_motifs.items():
    print(f"\n=== Processing motif: {motif_name} ===")

    # 1) Simulate
    t_sim, X, U, Y = simulator.simulate_bd_odeint(
        x0=x0,
        t_grid=t_grid,
        uD_fun=uD_fun,
        theta_local=bd.theta,
        noisy=False,
        rng_seed=123,
    )

    # 2) Build PyBounds simulator and empirical observability matrix
    dt_sim = float(t_sim[1] - t_sim[0])
    pb_sim = simulator.make_pybounds_simulator(dt=dt_sim)

    window_size = 10  # you may experiment with this
    SEOM, O_sliding = observability.build_sliding_empirical_observability_matrix(
        simulator=pb_sim,
        t_sim=t_sim,
        X=X,
        U=U,
        window_size=window_size,
        eps=1e-4,
    )

    # 3) Estimate measurement-noise variances and build sliding Fisher
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

    # 4) Compute simple scalar metrics per state (mean / median / min EV)
    EV_mean = EV_df.mean()
    EV_median = EV_df.median()
    EV_min = EV_df.min()

    for state in model.STATE_NAMES:
        results.append(
            dict(
                motif=motif_name,
                state=state,
                EV_mean=float(EV_mean[state]),
                EV_median=float(EV_median[state]),
                EV_min=float(EV_min[state]),
            )
        )

# Convert results to DataFrame
metrics_df = pd.DataFrame(results)
metrics_df.head()

# --- Pivot table for mean error variance (EV_mean) ---

pivot_mean = metrics_df.pivot(index="state", columns="motif", values="EV_mean")
pivot_median = metrics_df.pivot(index="state", columns="motif", values="EV_median")
pivot_min = metrics_df.pivot(index="state", columns="motif", values="EV_min")

print("Mean minimum error variance per state and motif:")
display(pivot_mean)

print("\nMedian minimum error variance per state and motif:")
display(pivot_median)

print("\nMinimum (best) minimum error variance per state and motif:")
display(pivot_min)

# --- Bar plots to compare control motifs ---

states = model.STATE_NAMES
motifs = list(control_motifs.keys())

# We'll plot EV_mean on a log scale to highlight differences.
fig, ax = plt.subplots(figsize=(9, 5))

width = 0.2
x_positions = np.arange(len(states))

for i, motif_name in enumerate(motifs):
    values = [pivot_mean.loc[state, motif_name] for state in states]
    ax.bar(
        x_positions + i * width,
        values,
        width=width,
        label=motif_name,
    )

ax.set_xticks(x_positions + width)
ax.set_xticklabels(states)
ax.set_yscale("log")
ax.set_ylabel("Mean minimum error variance (log scale)")
ax.set_title("Comparison of empirical observability across control motifs")
ax.legend()
fig.tight_layout()
plt.show()
