import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from Utility import bd_chemostat as bd
from Utility import bd_motifs as motifs

# ---------------- USER CONFIG ----------------
MOTIF_NAME = "sin_12h_big"     # const_0p12, step_24h, sin_12h, sin_12h_big
MEAS_OPT   = "h_BNW"           # h_BNW, h_BN, h_B, h_NW, h_N, h_W

T_FINAL = 48.0   # [h]
DT      = 0.05   # [h]
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)

# Colormap settings (igual idea que Lesson 8B)
V_MIN = 1e-5
V_MAX = 1e8
CMAP  = "inferno"
# --------------------------------------------

def colored_line(ax, x, y, c, norm, cmap):
    """Add a colored line to ax using LineCollection; c sets the color per segment."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.asarray(c, float)

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(seg, cmap=cmap, norm=norm)
    lc.set_array(c[:-1])     # one color per segment
    lc.set_linewidth(2.0)

    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ypad = 0.02 * (y.max() - y.min() + 1e-12)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)
    return lc

def main():
    # Paths
    CURRENT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load MEV results produced by Phase 2F
    npz_path = os.path.join(RESULTS_DIR, f"mev_{MOTIF_NAME}_{MEAS_OPT}.npz")
    data = np.load(npz_path, allow_pickle=True)
    t_mid = np.asarray(data["t_mid"], float)
    mev   = np.asarray(data["mev"], float)  # shape: (len(t_mid), 6)

    # Build model f, measurement h, motif D_fun
    f = bd.F().f
    h = bd.H(measurement_option=MEAS_OPT).h

    D_map = {
        "const_0p12": motifs.D_const_0p12,
        "step_24h": motifs.D_step_24h,
        "sin_12h": motifs.D_sin_12h,
        "sin_12h_big": motifs.D_sin_12h_big,
    }
    D_fun = D_map[MOTIF_NAME]

    # Simulate nominal state trajectory
    t_sim, x_nom, u_nom, y_nom = bd.simulate_bd(
        f=f, h=h, tsim_length=T_FINAL, dt=DT, x0=X0_NOMINAL, D=0.0, D_fun=D_fun
    )

    state_names = bd.F().return_state_names()
    n = len(state_names)

    # Global norm so colors are comparable
    norm = LogNorm(vmin=V_MIN, vmax=V_MAX)

    fig, axes = plt.subplots(n, 2, figsize=(13, 2.1*n), sharex="col")

    for i, name in enumerate(state_names):
        axL = axes[i, 0]
        axR = axes[i, 1]

        mev_i = np.maximum(mev[:, i], 1e-30)

        # Interpolate MEV onto t_sim so the state trajectory can be colored
        mev_on_t = np.interp(t_sim, t_mid, mev_i, left=mev_i[0], right=mev_i[-1])

        # Left: state trajectory colored by MEV
        colored_line(axL, t_sim, x_nom[:, i], mev_on_t, norm=norm, cmap=CMAP)
        axL.set_ylabel(f"{name}")
        axL.grid(True, alpha=0.3)

        # Right: MEV curve colored by itself (log y-scale)
        lcR = colored_line(axR, t_mid, mev_i, mev_i, norm=norm, cmap=CMAP)
        axR.set_yscale("log")
        axR.set_ylabel(f"MEV({name})")
        axR.grid(True, which="both", alpha=0.3)

        cbar = fig.colorbar(lcR, ax=axR, pad=0.01)
        cbar.set_label(f"min. EV: {name}")

    axes[-1, 0].set_xlabel("t [h]")
    axes[-1, 1].set_xlabel("t [h]")
    axes[0, 0].set_title("State trajectory (colored by MEV)")
    axes[0, 1].set_title("Minimum error variance (CRLB)")

    fig.suptitle(f"Lesson-8B colored — motif={MOTIF_NAME}, meas={MEAS_OPT}, window=6.0h, step=0.5h")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_png = os.path.join(RESULTS_DIR, f"mev_{MOTIF_NAME}_{MEAS_OPT}_lesson8B_colored.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print("Saved:", out_png)

if __name__ == "__main__":
    main()
