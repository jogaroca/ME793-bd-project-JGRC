import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from Utility import bd_chemostat as bd
from Utility import bd_motifs as motifs

MOTIF_NAME = "sin_12h_big"
MEAS_LIST  = ["h_BNW", "h_N"]   # good vs bad

T_FINAL = 48.0   # [h]
DT      = 0.05   # [h]
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)

# Color scale comparable across figures (Lesson 8B idea)
V_MIN = 1e-5
V_MAX = 1e8
CMAP  = "inferno"

def colored_line(ax, x, y, c, norm, cmap):
    x = np.asarray(x, float); y = np.asarray(y, float); c = np.asarray(c, float)
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=norm)
    lc.set_array(c[:-1])
    lc.set_linewidth(2.0)
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ypad = 0.02 * (y.max() - y.min() + 1e-12)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)
    return lc

def make_figure(meas_opt: str, results_dir: str):
    npz_path = os.path.join(results_dir, f"mev_{MOTIF_NAME}_{meas_opt}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}. Run Phase 2F first.")

    data = np.load(npz_path, allow_pickle=True)
    t_mid = np.asarray(data["t_mid"], float)
    mev   = np.asarray(data["mev"], float)  # (len(t_mid), 6)

    f = bd.F().f
    h = bd.H(measurement_option=meas_opt).h

    D_map = {
        "const_0p12": motifs.D_const_0p12,
        "step_24h": motifs.D_step_24h,
        "sin_12h": motifs.D_sin_12h,
        "sin_12h_big": motifs.D_sin_12h_big,
    }
    D_fun = D_map[MOTIF_NAME]

    t_sim, x_nom, u_nom, y_nom = bd.simulate_bd(
        f=f, h=h, tsim_length=T_FINAL, dt=DT, x0=X0_NOMINAL, D=0.0, D_fun=D_fun
    )

    state_names = bd.F().return_state_names()
    n = len(state_names)
    norm = LogNorm(vmin=V_MIN, vmax=V_MAX)

    fig, axes = plt.subplots(n, 2, figsize=(13, 2.1*n), sharex="col")

    for i, name in enumerate(state_names):
        axL = axes[i, 0]; axR = axes[i, 1]
        mev_i = np.maximum(mev[:, i], 1e-30)

        # color x(t) using MEV interpolated onto simulation grid
        mev_on_t = np.interp(t_sim, t_mid, mev_i, left=mev_i[0], right=mev_i[-1])

        colored_line(axL, t_sim, x_nom[:, i], mev_on_t, norm=norm, cmap=CMAP)
        axL.set_ylabel(f"{name}")
        axL.grid(True, alpha=0.3)

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

    fig.suptitle(f"Lesson-8B colored — motif={MOTIF_NAME}, meas={meas_opt}, window=6.0h, step=0.5h")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_png = os.path.join(results_dir, f"mev_{MOTIF_NAME}_{meas_opt}_lesson8B_colored.png")
    out_pdf = os.path.join(results_dir, f"mev_{MOTIF_NAME}_{meas_opt}_lesson8B_colored.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf

def main():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    results_dir = os.path.join(project_root, "results", "phase2_empirical")
    os.makedirs(results_dir, exist_ok=True)

    for meas in MEAS_LIST:
        png, pdf = make_figure(meas, results_dir)
        print("Saved:", png)
        print("Saved:", pdf)

if __name__ == "__main__":
    main()
