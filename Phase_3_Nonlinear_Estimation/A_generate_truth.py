"""
Phase 3 - Milestone 1
A_generate_truth.py

Generates "truth" trajectories for each trajectory motif and saves them
using the Phase 3 truth NPZ contract defined in config_phase3.py.

Outputs:
  results/phase3/truth/truth_<motif>.npz
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict

import numpy as np

# ---------------------------------------------------------------------
# Robust path handling: add project root to sys.path so `import Utility...` works
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase_3_Nonlinear_Estimation.config_phase3 import (
    PHASE3_VERSION,
    default_config,
    truth_dir,
    truth_file,
)

from Utility import bd_chemostat as bd
from Utility import bd_motifs as motifs


def _dummy_h(*args, **kwargs) -> np.ndarray:
    """
    Dummy measurement function for bd.simulate_bd; we do not use y(t) in truth generation.
    """
    return np.zeros(3, dtype=float)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _validate_motifs(motif_names, motif_map: Dict[str, Callable[[float], float]]) -> None:
    missing = [m for m in motif_names if m not in motif_map]
    if missing:
        raise KeyError(
            f"Motif(s) not found in Utility/bd_motifs.MOTIFS: {missing}. "
            f"Available motifs: {sorted(list(motif_map.keys()))}"
        )


def main() -> None:
    cfg = default_config()

    # Ensure output directories exist
    _ensure_dir(truth_dir())

    # Validate motif names
    _validate_motifs(cfg.motifs, motifs.MOTIFS)

    # Bd dynamics object
    f_obj = bd.F()
    f = f_obj.f

    # Standard names (useful metadata)
    state_names = f_obj.f(None, None, return_state_names=True)
    input_names = ("D",)

    # Nominal initial condition
    x0 = np.asarray(cfg.x0_nominal, dtype=float).reshape(6,)

    print("=" * 80)
    print("Phase 3 - Milestone 1: Generating truth trajectories")
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Truth out dir : {truth_dir()}")
    print(f"T_final       : {cfg.T_final} h")
    print(f"dt_truth      : {cfg.dt_truth} h")
    print(f"Motifs        : {cfg.motifs}")
    print("=" * 80)

    for motif_name in cfg.motifs:
        D_fun = motifs.MOTIFS[motif_name]

        print("-" * 80)
        print(f"Simulating motif: {motif_name}")

        # Simulate
        t_vec, x_traj, u_traj, _ = bd.simulate_bd(
            f=f,
            h=_dummy_h,
            tsim_length=float(cfg.T_final),
            dt=float(cfg.dt_truth),
            x0=x0,
            D=0.0,         # ignored because D_fun is provided
            D_fun=D_fun,
        )

        # Enforce shapes
        t_vec = np.asarray(t_vec, dtype=float).reshape(-1)
        x_traj = np.asarray(x_traj, dtype=float)
        u_traj = np.asarray(u_traj, dtype=float).reshape(-1, 1)

        if x_traj.ndim != 2 or x_traj.shape[1] != 6:
            raise ValueError(f"x_true must have shape (T,6). Got {x_traj.shape}")
        if t_vec.size != x_traj.shape[0] or u_traj.shape[0] != x_traj.shape[0]:
            raise ValueError(
                f"Length mismatch: len(t)={t_vec.size}, x_true rows={x_traj.shape[0]}, u_D rows={u_traj.shape[0]}"
            )

        # Convenience signals (often useful later)
        Z, S1, S2, S3, N, W = (x_traj[:, i] for i in range(6))
        B = Z + S1 + S2 + S3

        # Save
        out_path = truth_file(motif_name)
        np.savez(
            out_path,
            # Required contract fields
            t=t_vec,
            dt=float(cfg.dt_truth),
            x_true=x_traj,
            u_D=u_traj,
            motif_name=str(motif_name),
            # Recommended metadata
            T_final=float(cfg.T_final),
            x0=x0,
            state_names=np.array(state_names, dtype=object),
            input_names=np.array(input_names, dtype=object),
            params_bd=asdict(f_obj.params),
            phase3_version=str(PHASE3_VERSION),
            # Optional convenience (safe to keep)
            B=B,
            N=N,
            W=W,
        )

        print(f"Saved: {out_path}")

    print("\nDone. Truth files generated for all motifs.")


if __name__ == "__main__":
    main()
