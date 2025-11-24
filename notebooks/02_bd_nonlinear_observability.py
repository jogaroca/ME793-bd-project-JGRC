# 02_bd_nonlinear_observability.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Global list of state names (used in plotting functions)
state_names = ["Z", "S1", "S2", "S3", "N", "W"]


def eval_control_motif(u_func, t_vec):
    """Evaluate a control law u_func(x, t) along a time vector using x=None."""
    return np.array([u_func(None, t)[0] for t in t_vec])


def plot_Finv_heatmap(Finv, title="Finv (log10 scale)"):
    """Plot a heatmap of log10(Finv)."""
    plt.figure(figsize=(5, 4))
    img = plt.imshow(np.log10(Finv), origin="upper", cmap="viridis")
    plt.colorbar(img, label="log10(value)")
    plt.xticks(range(len(state_names)), state_names)
    plt.yticks(range(len(state_names)), state_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_diag_Finv_comparison(results_dict):
    """Bar plot of log10(diag(Finv)) for several motifs.

    Parameters
    ----------
    results_dict : dict
        Dictionary {name: (W_o, Finv)} with observability results.
    """
    n_states = len(state_names)
    n_motifs = len(results_dict)

    width = 0.8 / n_motifs
    x = np.arange(n_states)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (name, (_, Finv)) in enumerate(results_dict.items()):
        diag_vals = np.diag(Finv)
        ax.bar(x + k * width, np.log10(diag_vals),
               width=width, label=name)

    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(state_names)
    ax.set_ylabel("log10(min error variance)")
    ax.set_title("Empirical observability comparison")
    ax.legend()
    fig.tight_layout()
    plt.show()

    # If you want to save this figure for the report, uncomment:
    # fig.savefig("bd_observability_barplot.png", dpi=300, bbox_inches="tight")


def main():
    # ------------------------------------------------------------------
    # Make sure we can import the local bd package from src/
    # ------------------------------------------------------------------
    ROOT = (
        os.path.dirname(os.path.dirname(os.getcwd()))
        if "notebooks" in os.getcwd()
        else os.getcwd()
    )
    SRC_PATH = os.path.join(ROOT, "src")
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)

    import bd  # uses src/bd/__init__.py

    print("Project root:", ROOT)
    print("bd module loaded OK")

    # ------------------------------------------------------------------
    # Time grid and initial condition for Bd simulations
    # ------------------------------------------------------------------
    dt = 0.05  # hours
    t_final = 48.0
    tsim = np.arange(0.0, t_final + dt, dt)

    x0 = bd.default_initial_condition()

    print("Number of time steps:", len(tsim))
    print("Initial condition x0:", x0)

    # ------------------------------------------------------------------
    # Visualize the three control motifs used for observability analysis
    # ------------------------------------------------------------------
    u_const = eval_control_motif(bd.u_const_012, tsim)
    u_step = eval_control_motif(bd.u_step_24h, tsim)
    u_sin = eval_control_motif(bd.u_sin_12h, tsim)

    plt.figure(figsize=(8, 4))
    plt.plot(tsim, u_const, label="const_0.12")
    plt.plot(tsim, u_step, label="step_24h")
    plt.plot(tsim, u_sin, label="sin_12h")
    plt.xlabel("time [h]")
    plt.ylabel("u_D [1/h]")
    plt.title("Bd control motifs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Si quieres guardar esta figura para el reporte:
    # plt.savefig("bd_motifs.png", dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Simulate Bd for one motif as a sanity check
    # ------------------------------------------------------------------
    t_sim, Xdf, Udf, Ydf = bd.simulate_bd(bd.u_const_012, x0, tsim)

    print(Xdf.head())
    print(Ydf.head())

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(t_sim, Xdf["Z"], label="Z")
    axes[0].set_ylabel("Z")
    axes[0].legend()

    axes[1].plot(t_sim, Xdf["S1"], label="S1")
    axes[1].plot(t_sim, Xdf["S2"], label="S2")
    axes[1].plot(t_sim, Xdf["S3"], label="S3")
    axes[1].set_ylabel("S_i")
    axes[1].legend()

    axes[2].plot(t_sim, Xdf["N"], label="N")
    axes[2].set_ylabel("N [mM]")
    axes[2].legend()

    axes[3].plot(t_sim, Xdf["W"], label="W")
    axes[3].set_ylabel("W")
    axes[3].set_xlabel("time [h]")
    axes[3].legend()

    fig.tight_layout()
    plt.show()

    # Plot measurements B, N, W
    Ydf.plot(subplots=True, figsize=(8, 6), sharex=True, title=["B", "N", "W"])
    plt.xlabel("time [h]")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Compute the empirical observability Gramian for each control motif
    # ------------------------------------------------------------------
    motifs = {
        "const_0.12": bd.u_const_012,
        "step_24h": bd.u_step_24h,
        "sin_12h": bd.u_sin_12h,
    }

    results = {}

    eps = 1e-3  # state perturbation magnitude
    lam = 1e-6  # Tikhonov regularization

    for name, ufun in motifs.items():
        print(f"Computing empirical Gramian for motif: {name}")
        W_o, Finv, O = bd.empirical_observability_gramian(
            u_func=ufun,
            x0=x0,
            tsim=tsim,
            eps=eps,
            lam=lam,
        )
        results[name] = (W_o, Finv)
        eigvals = np.linalg.eigvalsh(W_o)
        print(f"  min eigenvalue(W_o) = {eigvals.min():.3e}")
        print(f"  max eigenvalue(W_o) = {eigvals.max():.3e}")

    # ------------------------------------------------------------------
    # Plots: heatmaps and bar comparison
    # ------------------------------------------------------------------
    for name, (W_o, Finv) in results.items():
        plot_Finv_heatmap(Finv, title=f"Finv, motif = {name}")

    plot_diag_Finv_comparison(results)

    # ------------------------------------------------------------------
    # Numeric summary table with diag(Finv) per state and motif
    # ------------------------------------------------------------------
    summary_rows = []
    for name, (W_o, Finv) in results.items():
        diag_vals = np.diag(Finv)
        for s_idx, s_name in enumerate(state_names):
            summary_rows.append(
                dict(
                    motif=name,
                    state=s_name,
                    Finv_diag=diag_vals[s_idx],
                    log10_Finv_diag=np.log10(diag_vals[s_idx]),
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df_pivot = summary_df.pivot(
        index="state", columns="motif", values="log10_Finv_diag"
    )

    print("\nlog10(diag(Finv)) per state and motif:")
    print(summary_df_pivot)

    # Optional: save summary to CSV for later use in the report
    # summary_df_pivot.to_csv("bd_observability_summary.csv")
    # ------------------------------------------------------------------
    # OPTIONAL: measurement-subset analysis (for project write-up)
    # ------------------------------------------------------------------
    measurement_subsets = {
        "B":      [0],
        "N":      [1],
        "W":      [2],
        "B+N":    [0, 1],
        "B+W":    [0, 2],
        "N+W":    [1, 2],
        "B+N+W":  [0, 1, 2],
    }

    subset_rows = []
    for subset_name, meas_idx in measurement_subsets.items():
        for motif_name, ufun in motifs.items():
            W_o_sub, Finv_sub, _ = bd.empirical_observability_gramian(
                u_func=ufun,
                x0=x0,
                tsim=tsim,
                eps=eps,
                lam=lam,
                meas_indices=meas_idx,
            )
            diag_vals = np.diag(Finv_sub)
            for s_idx, s_name in enumerate(state_names):
                subset_rows.append(
                    dict(
                        subset=subset_name,
                        motif=motif_name,
                        state=s_name,
                        Finv_diag=diag_vals[s_idx],
                        log10_Finv_diag=np.log10(diag_vals[s_idx]),
                    )
                )

    subset_df = pd.DataFrame(subset_rows)
    print("\nMeasurement-subset observability summary (log10 diag(Finv)):")
    print(subset_df.head())
    # You can pivot this to inspect specific states or subsets, e.g.:
    # subset_pivot = subset_df.pivot_table(
    #     index=["state"],
    #     columns=["subset", "motif"],
    #     values="log10_Finv_diag",
    # )
    # print(subset_pivot)



if __name__ == "__main__":
    main()
