"""
Phase 5 (Future Work)
A_compare_motifs_vs_adaptive.py

Simulate and plot:
  - 4 existing open-loop motifs (Utility/bd_motifs.py)
  - 1 adaptive closed-loop policy (normalized saturation)

Outputs:
  results/phase5/sims/sim_<name>.npz
  results/phase5/figures/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------
# Robust path handling
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase_5_Future_Work.config_phase5 import (
    default_config,
    sims_dir,
    figures_dir,
    sim_file,
)

from Utility import bd_chemostat as bd
from Utility import bd_motifs as motifs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sat(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def adaptive_policy_normalized(cfg, t: float, x: np.ndarray) -> float:
    """
    Normalized saturated policy:
      u_D = sat_[Dmin,Dmax]( D0 + kW*((W-W*)/W*)_+ + kB*((B-B*)/B*) )

    Note: here we use *true* states (closed-loop on x). For an EKF version,
    replace B,W with estimates (Bhat, What).
    """
    Z, S1, S2, S3, N, W = x
    B = Z + S1 + S2 + S3

    B_star = cfg.B_star if cfg.B_star is not None else max(B, cfg.eps)
    W_star = max(cfg.W_star, cfg.eps)

    term_W = max((W - W_star) / W_star, 0.0)
    term_B = (B - B_star) / max(B_star, cfg.eps)

    D = cfg.D0 + cfg.k_W * term_W + cfg.k_B * term_B
    return sat(D, cfg.D_min, cfg.D_max)


def simulate_open_loop(f_obj: bd.F, T_final: float, dt: float, x0: np.ndarray, D_fun: Callable[[float], float]):
    # Reuse existing helper for open-loop motifs
    t, x, u, _ = bd.simulate_bd(
        f=f_obj.f,
        h=lambda *args, **kwargs: np.zeros(3),
        tsim_length=float(T_final),
        dt=float(dt),
        x0=x0,
        D=0.0,
        D_fun=D_fun,
    )
    return np.asarray(t), np.asarray(x), np.asarray(u)


def simulate_closed_loop(f_obj: bd.F, cfg, T_final: float, dt: float, x0: np.ndarray):
    t_eval = np.arange(0.0, T_final + dt, dt)

    def rhs(t, x):
        D = adaptive_policy_normalized(cfg, t, x)
        return f_obj.f(x, [D])

    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(T_final)),
        y0=np.asarray(x0, dtype=float).reshape(6,),
        t_eval=t_eval,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    t = sol.t
    x = sol.y.T  # (T,6)

    # Recompute u(t) on the stored grid (consistent with x(t))
    u = np.array([adaptive_policy_normalized(cfg, float(ti), x[i, :]) for i, ti in enumerate(t)], dtype=float).reshape(-1, 1)
    return t, x, u


def signals_from_x(x: np.ndarray):
    Z, S1, S2, S3, N, W = (x[:, i] for i in range(6))
    B = Z + S1 + S2 + S3
    return {"Z": Z, "S1": S1, "S2": S2, "S3": S3, "N": N, "W": W, "B": B}


def save_sim_npz(path: Path, name: str, t: np.ndarray, x: np.ndarray, u: np.ndarray):
    np.savez(
        path,
        name=str(name),
        t=np.asarray(t, dtype=float),
        x=np.asarray(x, dtype=float),
        u_D=np.asarray(u, dtype=float).reshape(-1, 1),
    )


def plot_overlay(t: np.ndarray, series: Dict[str, np.ndarray], title: str, ylabel: str, outpath: Path):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    for label, y in series.items():
        ax.plot(t, y, label=label)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_states_grid(t: np.ndarray, series_map: Dict[str, Dict[str, np.ndarray]], outpath: Path):
    """
    series_map: {case_name: signals_dict}, where signals_dict has keys Z,S1,S2,S3,N,W,B,u maybe
    Produces a 2x3 grid for states (Z,S1,S2,S3,N,W).
    """
    state_keys = ["Z", "S1", "S2", "S3", "N", "W"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.ravel()

    for k, ax in zip(state_keys, axes):
        for case, sig in series_map.items():
            ax.plot(t, sig[k], label=case)
        ax.set_title(k)
        ax.grid(True, alpha=0.3)

    for ax in axes[3:]:
        ax.set_xlabel("Time [h]")

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, fontsize=10)
    fig.suptitle("Phase 5: States comparison (open-loop motifs vs adaptive policy)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    cfg = default_config()
    ensure_dir(sims_dir())
    ensure_dir(figures_dir())

    f_obj = bd.F()
    x0 = np.asarray(cfg.x0, dtype=float).reshape(6,)

    # -------------------------
    # Simulate open-loop motifs
    # -------------------------
    series: Dict[str, Dict[str, np.ndarray]] = {}
    u_series: Dict[str, np.ndarray] = {}

    for m in cfg.open_loop_motifs:
        if m not in motifs.MOTIFS:
            raise KeyError(f"Motif '{m}' not found. Available: {sorted(list(motifs.MOTIFS.keys()))}")
        t, x, u = simulate_open_loop(f_obj, cfg.T_final, cfg.dt, x0, motifs.MOTIFS[m])
        save_sim_npz(sim_file(m), m, t, x, u)

        sig = signals_from_x(x)
        series[m] = sig
        u_series[m] = u[:, 0]

    # -------------------------
    # Simulate adaptive policy
    # -------------------------
    # Ensure B* has a default if not provided
    if cfg.B_star is None:
        B0 = float(x0[0] + x0[1] + x0[2] + x0[3])
        # cfg is frozen; define a local view
        class _CfgView:
            pass
        cfg_view = _CfgView()
        for k, v in cfg.__dict__.items():
            setattr(cfg_view, k, v)
        cfg_view.B_star = B0
    else:
        cfg_view = cfg

    t_cl, x_cl, u_cl = simulate_closed_loop(f_obj, cfg_view, cfg.T_final, cfg.dt, x0)
    name_cl = "adaptive_normalized"
    save_sim_npz(sim_file(name_cl), name_cl, t_cl, x_cl, u_cl)

    sig_cl = signals_from_x(x_cl)
    series[name_cl] = sig_cl
    u_series[name_cl] = u_cl[:, 0]

    # -------------------------
    # Plots (overlay + grid)
    # -------------------------
    # Use the same t (they should match); if you later change dt per case, load per-case t.
    t_plot = t_cl

    plot_overlay(
        t_plot,
        {k: u_series[k] for k in u_series},
        title="Dilution input comparison",
        ylabel=r"$u_D(t)$ [h$^{-1}$]",
        outpath=figures_dir() / "phase5_uD_overlay.png",
    )

    plot_overlay(
        t_plot,
        {k: series[k]["B"] for k in series},
        title="Total biomass comparison",
        ylabel=r"$B(t)$",
        outpath=figures_dir() / "phase5_B_overlay.png",
    )

    plot_overlay(
        t_plot,
        {k: series[k]["W"] for k in series},
        title="Inhibition / waste comparison",
        ylabel=r"$W(t)$",
        outpath=figures_dir() / "phase5_W_overlay.png",
    )

    plot_overlay(
        t_plot,
        {k: series[k]["N"] for k in series},
        title="Nutrient comparison",
        ylabel=r"$N(t)$ [mM]",
        outpath=figures_dir() / "phase5_N_overlay.png",
    )

    plot_states_grid(
        t_plot,
        series_map={k: series[k] for k in series},
        outpath=figures_dir() / "phase5_states_grid.png",
    )

    print("=" * 80)
    print("Phase 5 complete.")
    print(f"Saved sims    : {sims_dir()}")
    print(f"Saved figures : {figures_dir()}")
    print("Cases         :", list(series.keys()))
    print("=" * 80)


if __name__ == "__main__":
    main()
