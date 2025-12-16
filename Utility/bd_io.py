"""
I/O helpers for Phase 3 (estimation) scripts.

Truth file convention:
  results/phase3_estimation/truth_motif_<motif>.npz

Required keys inside NPZ:
  t      : (T,) time vector [h]
  x_true : (T,6) states [Z,S1,S2,S3,N,W]
  u_D    : (T,) dilution input
"""

import os
import numpy as np

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _coerce_1d(a):
    return np.asarray(a).reshape(-1)

def load_truth_or_fail(project_root: str, motif: str):
    truth_path = os.path.join(
        project_root, "results", "phase3_estimation", f"truth_motif_{motif}.npz"
    )
    if not os.path.exists(truth_path):
        raise FileNotFoundError(
            f"Truth file not found for motif='{motif}'. Expected: {truth_path}. "
            "Run Phase_3_Nonlinear_Estimation/A_bd_generate_synthetic_data.py first."
        )

    data = np.load(truth_path)

    # Allow legacy key 'x' if needed
    if "x_true" in data:
        x_true = np.asarray(data["x_true"], dtype=float)
    elif "x" in data:
        x_true = np.asarray(data["x"], dtype=float)
    else:
        raise KeyError(f"{truth_path} missing 'x_true' (or legacy 'x'). Keys: {list(data.keys())}")

    t = _coerce_1d(np.asarray(data["t"], dtype=float))
    u_D = _coerce_1d(np.asarray(data["u_D"], dtype=float))

    if x_true.ndim != 2 or x_true.shape[1] != 6:
        raise ValueError(f"Expected x_true shape (T,6). Got {x_true.shape}")

    if len(t) != x_true.shape[0] or len(u_D) != x_true.shape[0]:
        raise ValueError(
            f"Inconsistent lengths: len(t)={len(t)}, len(u_D)={len(u_D)}, x_true rows={x_true.shape[0]}"
        )

    Z  = x_true[:, 0]
    S1 = x_true[:, 1]
    S2 = x_true[:, 2]
    S3 = x_true[:, 3]
    N  = x_true[:, 4]
    W  = x_true[:, 5]
    B  = Z + S1 + S2 + S3

    return {
        "t": t,
        "x_true": x_true,
        "u_D": u_D,
        "Z": Z, "S1": S1, "S2": S2, "S3": S3,
        "N": N, "W": W, "B": B,
    }
