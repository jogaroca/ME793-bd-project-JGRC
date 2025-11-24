import os, sys

# Assume this notebook lives in `notebooks/`
# Go one level up to the repo root
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

print("Repo root:", repo_root)
print("Python path OK.")
import numpy as np
import matplotlib.pyplot as plt

from src.bd_model import default_initial_state, STATE_NAMES
from src.bd_inputs import u_const_012, u_step_24h, u_sin_12h
from src.bd_simulation import simulate_bd, compute_effluent_productivity
from src.bd_measurements import meas_setA
# Time grid (hours)
dt = 0.01
t_final = 48.0
t_grid = np.arange(0.0, t_final + dt, dt)

# Initial condition
x0 = default_initial_state()
x0
# Choose input motif
u_func = u_const_012  # we will start with the constant motif

t, X, Y = simulate_bd(x0, t_grid, u_func)
print("Simulation shape:", X.shape, "Measurements shape:", Y.shape)
fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
axes = axes.ravel()

for i, name in enumerate(STATE_NAMES):
    axes[i].plot(t, X[:, i])
    axes[i].set_ylabel(name)

axes[-1].set_xlabel("time (h)")
fig.suptitle("Bd chemostat state trajectories (constant u_D = 0.12)", y=1.02)
plt.tight_layout()
plt.show()
# Input trace
u_trace = np.array([float(u_func(x, ti)[0]) for x, ti in zip(X, t)])

# Effluent productivity P_out = u_D * Z
P_out = compute_effluent_productivity(t, X, u_func)

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax[0].plot(t, u_trace)
ax[0].set_ylabel("u_D (h$^{-1}$)")
ax[0].set_title("Dilution rate motif")

ax[1].plot(t, P_out)
ax[1].set_xlabel("time (h)")
ax[1].set_ylabel("P_out = u_D * Z")
ax[1].set_title("Effluent productivity")

plt.tight_layout()
plt.show()
Y_meas, B_true = meas_setA(X, rng_seed=123)

# B (true vs measured)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, B_true, label="B (true)", lw=2)
ax.plot(t, Y_meas[:, 0], ".", ms=2, label="B (measured)")
ax.set_xlabel("time (h)")
ax.set_ylabel("B")
ax.set_title("Total biomass B")
ax.legend()
plt.tight_layout()
plt.show()

# N and W (true vs measured)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

axes[0].plot(t, X[:, 4], lw=2, label="N (true)")
axes[0].plot(t, Y_meas[:, 1], ".", ms=2, label="N (measured)")
axes[0].set_xlabel("time (h)")
axes[0].set_ylabel("N (mM)")
axes[0].legend()
axes[0].set_title("N")

axes[1].plot(t, X[:, 5], lw=2, label="W (true)")
axes[1].plot(t, Y_meas[:, 2], ".", ms=2, label="W (measured)")
axes[1].set_xlabel("time (h)")
axes[1].set_ylabel("W")
axes[1].legend()
axes[1].set_title("W")

plt.tight_layout()
plt.show()
