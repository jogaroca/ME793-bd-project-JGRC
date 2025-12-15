%%writefile Phase_2_Nonlinear_Observability/G_bd_mev_Z_summary_figures.py
import os
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")

MOTIFS = ["const_0p12", "step_24h", "sin_12h", "sin_12h_big"]
MEAS_OPTIONS = ["h_BNW", "h_BN", "h_B", "h_NW", "h_N", "h_W"]

STATE_NAMES = ("Z", "S1", "S2", "S3", "N", "W")
Z_IDX = 0

# ventanas para “early transient” y “late” (ajusta si quieres)
EARLY_HOURS = (0.0, 12.0)
LATE_HOURS  = (24.0, 48.0)

def _load_mev_npz(motif: str, meas: str):
    path = os.path.join(RESULTS_DIR, f"mev_{motif}_{meas}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero F_bd_min_error_variance_timeseries.py"
        )
    data = np.load(path, allow_pickle=True)
    t_mid = np.asarray(data["t_mid"], dtype=float)
    mev = np.asarray(data["mev"], dtype=float)
    return t_mid, mev

def _summary_stat(t_mid: np.ndarray, mev_z: np.ndarray, hours_window):
    t0, t1 = hours_window
    mask = (t_mid >= t0) & (t_mid <= t1)
    if not np.any(mask):
        return np.nan
    # métrica robusta: mediana (evita outliers)
    return float(np.median(mev_z[mask]))

def make_overlay_plots():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for motif in MOTIFS:
        plt.figure(figsize=(10, 4))
        for meas in MEAS_OPTIONS:
            t_mid, mev = _load_mev_npz(motif, meas)
            mev_z = np.maximum(mev[:, Z_IDX], 1e-30)
            plt.semilogy(t_mid, mev_z, label=meas)
        plt.grid(True, which="both", alpha=0.3)
        plt.xlabel("t [h]")
        plt.ylabel("MEV(Z)")
        plt.title(f"MEV(Z) vs time — motif={motif}")
        plt.legend(ncol=3, fontsize=8)
        out = os.path.join(RESULTS_DIR, f"mevZ_overlay_{motif}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()

def make_heatmaps():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # construir matrices: rows=motif, cols=meas
    early = np.zeros((len(MOTIFS), len(MEAS_OPTIONS)), dtype=float)
    late  = np.zeros((len(MOTIFS), len(MEAS_OPTIONS)), dtype=float)

    for i, motif in enumerate(MOTIFS):
        for j, meas in enumerate(MEAS_OPTIONS):
            t_mid, mev = _load_mev_npz(motif, meas)
            mev_z = np.maximum(mev[:, Z_IDX], 1e-30)
            early[i, j] = _summary_stat(t_mid, mev_z, EARLY_HOURS)
            late[i, j]  = _summary_stat(t_mid, mev_z, LATE_HOURS)

    # plot helper
    def _plot_heatmap(mat, title, outname):
        plt.figure(figsize=(10, 3.6))
        # usamos log10 para visualizar rangos grandes
        logm = np.log10(np.maximum(mat, 1e-30))
        im = plt.imshow(logm, aspect="auto")
        plt.xticks(np.arange(len(MEAS_OPTIONS)), MEAS_OPTIONS, rotation=30, ha="right")
        plt.yticks(np.arange(len(MOTIFS)), MOTIFS)
        plt.colorbar(im, label="log10( median MEV(Z) )")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, outname), dpi=220)
        plt.close()

    _plot_heatmap(
        early,
        f"Z observability proxy: median MEV(Z) in early window {EARLY_HOURS[0]}–{EARLY_HOURS[1]} h",
        "mevZ_heatmap_early.png",
    )
    _plot_heatmap(
        late,
        f"Z observability proxy: median MEV(Z) in late window {LATE_HOURS[0]}–{LATE_HOURS[1]} h",
        "mevZ_heatmap_late.png",
    )

def main():
    make_overlay_plots()
    make_heatmaps()
    print(f"Saved overlay + heatmap figures to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
