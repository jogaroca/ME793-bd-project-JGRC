import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

# -----------------------------
# Config (Sección Observability)
# -----------------------------
MOTIFS = ["const_0p12", "step_24h", "sin_12h", "sin_12h_big"]
MEAS_OPTIONS = ["h_BNW", "h_BN", "h_B", "h_NW", "h_N", "h_W"]

STATE_NAMES = ("Z", "S1", "S2", "S3", "N", "W")
Z_IDX = 0

EARLY_WIN = (0.0, 6.0)      # [h]
LATE_WIN  = (24.0, 48.0)    # [h]
STAT = "q10"                # "q10" recomendado (más discriminante que median)

# Figuras Lesson8B coloreadas (bueno vs malo)
L8B_MOTIF = "sin_12h_big"
L8B_CASES = ["h_BNW", "h_N"]

# Color scale (comparable entre subplots)
V_MIN = 1e-5
V_MAX = 1e8
CMAP  = "inferno"

# -----------------------------
# Paths
# -----------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (CURRENT_DIR / "..").resolve()
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2_empirical"
OUT_DIR = PROJECT_ROOT / "paper_figures" / "observability"

def _need(path: Path, msg: str):
    if not path.exists():
        raise FileNotFoundError(f"{msg}\nMissing: {path}")

def load_obs_metrics(motif: str, meas: str):
    p = RESULTS_DIR / f"obs_{motif}_{meas}.npz"
    _need(p, "Run Phase_2_Nonlinear_Observability/A_bd_empirical_observability.py first.")
    d = np.load(p)
    # Keys per A script + D script usage
    out = {
        "lambda_min": float(d["lambda_min"]),
        "lambda_max": float(d["lambda_max"]),
        "condition_number": float(d["condition_number"]),
        "trace_val": float(d["trace_val"]),
        "determinant": float(d["determinant"]),
        "var_Z": float(d["var_Z"]),
        "I_Z": float(d["I_Z"]),
    }
    return out

def load_mev_npz(motif: str, meas: str):
    p = RESULTS_DIR / f"mev_{motif}_{meas}.npz"
    _need(p, "Run Phase_2_Nonlinear_Observability/F_bd_min_error_variance_timeseries.py first.")
    d = np.load(p, allow_pickle=True)
    t_sim = np.asarray(d["t_sim"], float)
    x_nom = np.asarray(d["x_nominal"], float)
    t_mid = np.asarray(d["t_mid"], float)
    mev   = np.asarray(d["mev"], float)
    return t_sim, x_nom, t_mid, mev

def stat_in_window(t, y, window, stat="q10"):
    t0, t1 = window
    m = (t >= t0) & (t <= t1)
    if not np.any(m):
        return np.nan
    yy = np.maximum(y[m], 1e-30)
    if stat == "min":
        return float(np.min(yy))
    if stat == "q10":
        return float(np.quantile(yy, 0.10))
    if stat == "median":
        return float(np.median(yy))
    raise ValueError("stat must be min/q10/median")

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

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_barplots_global_metrics():
    # Barplots por motif: log10(lambda_min), log10(cond#)
    for motif in MOTIFS:
        lam = []
        cond = []
        for meas in MEAS_OPTIONS:
            m = load_obs_metrics(motif, meas)
            lam.append(m["lambda_min"])
            cond.append(m["condition_number"])
        lam = np.log10(np.maximum(np.array(lam), 1e-30))
        cond = np.log10(np.maximum(np.array(cond), 1.0))

        x = np.arange(len(MEAS_OPTIONS))

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(x, lam)
        ax.set_xticks(x)
        ax.set_xticklabels(MEAS_OPTIONS, rotation=30, ha="right")
        ax.set_ylabel("log10(lambda_min)")
        ax.set_title(f"Empirical observability: smallest eigenvalue (motif={motif})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"obs_lambda_min_{motif}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(x, cond)
        ax.set_xticks(x)
        ax.set_xticklabels(MEAS_OPTIONS, rotation=30, ha="right")
        ax.set_ylabel("log10(condition number)")
        ax.set_title(f"Empirical observability: condition number (motif={motif})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"obs_condition_{motif}.png", dpi=300)
        plt.close(fig)

def make_heatmaps_obs_Z_metrics():
    # Heatmaps: log10(var_Z) and log10(I_Z) across motif x meas
    varZ = np.zeros((len(MOTIFS), len(MEAS_OPTIONS)))
    Iz   = np.zeros((len(MOTIFS), len(MEAS_OPTIONS)))

    for i, motif in enumerate(MOTIFS):
        for j, meas in enumerate(MEAS_OPTIONS):
            m = load_obs_metrics(motif, meas)
            varZ[i, j] = m["var_Z"]
            Iz[i, j]   = m["I_Z"]

    def heatmap(mat, title, cbar_label, fname):
        fig, ax = plt.subplots(figsize=(9.2, 3.6))
        logm = np.log10(np.maximum(mat, 1e-30))
        im = ax.imshow(logm, aspect="auto")
        ax.set_xticks(np.arange(len(MEAS_OPTIONS)))
        ax.set_xticklabels(MEAS_OPTIONS, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(MOTIFS)))
        ax.set_yticklabels(MOTIFS)
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=300)
        plt.close(fig)

    heatmap(varZ, "Z global proxy (Chernoff): log10(var_Z) from Gramian", "log10(var_Z)", "obs_varZ_heatmap.png")
    heatmap(Iz,   "Z global proxy: log10(I_Z)=log10(sqrt(W_ZZ))",        "log10(I_Z)",   "obs_Iz_heatmap.png")

def make_mevZ_overlays():
    # Overlays MEV(Z) vs time por motif
    for motif in MOTIFS:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for meas in MEAS_OPTIONS:
            _, _, t_mid, mev = load_mev_npz(motif, meas)
            ax.semilogy(t_mid, np.maximum(mev[:, Z_IDX], 1e-30), label=meas)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("t [h]")
        ax.set_ylabel("MEV(Z)")
        ax.set_title(f"MEV(Z) vs time (motif={motif})")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"mevZ_overlay_{motif}.png", dpi=300)
        plt.close(fig)

def make_mevZ_heatmaps():
    early = np.zeros((len(MOTIFS), len(MEAS_OPTIONS)))
    late  = np.zeros((len(MOTIFS), len(MEAS_OPTIONS)))

    for i, motif in enumerate(MOTIFS):
        for j, meas in enumerate(MEAS_OPTIONS):
            _, _, t_mid, mev = load_mev_npz(motif, meas)
            early[i, j] = stat_in_window(t_mid, mev[:, Z_IDX], EARLY_WIN, stat=STAT)
            late[i, j]  = stat_in_window(t_mid, mev[:, Z_IDX], LATE_WIN,  stat=STAT)

    def heatmap(mat, title, fname):
        fig, ax = plt.subplots(figsize=(9.2, 3.6))
        logm = np.log10(np.maximum(mat, 1e-30))
        im = ax.imshow(logm, aspect="auto")
        ax.set_xticks(np.arange(len(MEAS_OPTIONS)))
        ax.set_xticklabels(MEAS_OPTIONS, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(MOTIFS)))
        ax.set_yticklabels(MOTIFS)
        fig.colorbar(im, ax=ax, label=f"log10({STAT} MEV(Z))")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=300)
        plt.close(fig)

    heatmap(early, f"Z MEV proxy: {STAT} MEV(Z) in {EARLY_WIN[0]}–{EARLY_WIN[1]} h", "mevZ_heatmap_early_0to6_q10.png")
    heatmap(late,  f"Z MEV proxy: {STAT} MEV(Z) in {LATE_WIN[0]}–{LATE_WIN[1]} h",  "mevZ_heatmap_late_24to48_q10.png")

def make_colored_lesson8B(motif: str, meas: str):
    t_sim, x_nom, t_mid, mev = load_mev_npz(motif, meas)

    norm = LogNorm(vmin=V_MIN, vmax=V_MAX)

    fig, axes = plt.subplots(6, 2, figsize=(13, 2.1*6), sharex="col")
    for i, name in enumerate(STATE_NAMES):
        axL = axes[i, 0]
        axR = axes[i, 1]

        mev_i = np.maximum(mev[:, i], 1e-30)
        mev_on_t = np.interp(t_sim, t_mid, mev_i, left=mev_i[0], right=mev_i[-1])

        colored_line(axL, t_sim, x_nom[:, i], mev_on_t, norm=norm, cmap=CMAP)
        axL.set_ylabel(name)
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

    # metadata from npz
    npz = np.load(RESULTS_DIR / f"mev_{motif}_{meas}.npz", allow_pickle=True)
    win_h = float(npz["window_hours"])
    step_h = float(npz["step_hours"])

    fig.suptitle(f"Lesson-8B colored — motif={motif}, meas={meas}, window={win_h}h, step={step_h}h")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_png = OUT_DIR / f"mev_{motif}_{meas}_lesson8B_colored.png"
    out_pdf = OUT_DIR / f"mev_{motif}_{meas}_lesson8B_colored.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

def export_summary_csv():
    rows = []
    for motif in MOTIFS:
        for meas in MEAS_OPTIONS:
            m = load_obs_metrics(motif, meas)
            rows.append([
                motif, meas,
                m["lambda_min"], m["lambda_max"], m["condition_number"],
                m["trace_val"], m["determinant"],
                m["var_Z"], m["I_Z"]
            ])
    header = [
        "motif","measurement",
        "lambda_min","lambda_max","condition_number",
        "trace(W)","det(W)",
        "var_Z","I_Z"
    ]
    arr = np.array(rows, dtype=object)

    out_csv = OUT_DIR / "observability_summary.csv"
    with open(out_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for r in arr:
            f.write(",".join(str(x) for x in r) + "\n")

def main():
    ensure_dirs()

    # (1) Global (Gramian) metrics figures + CSV
    make_barplots_global_metrics()
    make_heatmaps_obs_Z_metrics()
    export_summary_csv()

    # (2) MEV-based figures (core for Sección 4.2)
    make_mevZ_overlays()
    make_mevZ_heatmaps()

    # (3) Two Lesson8B-colored figures (good vs bad)
    for meas in L8B_CASES:
        make_colored_lesson8B(L8B_MOTIF, meas)

    print("All observability figures saved to:")
    print(OUT_DIR)

if __name__ == "__main__":
    main()
