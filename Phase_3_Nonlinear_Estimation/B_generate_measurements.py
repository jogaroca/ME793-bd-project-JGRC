"""
Phase 3 - Milestone 2
B_generate_measurements.py

Loads truth trajectories and generates noisy measurements for each:
  (motif, measurement_option)

Outputs:
  results/phase3/meas/meas_<motif>_<measopt>.npz
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------
# Robust imports
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Phase_3_Nonlinear_Estimation.config_phase3 import (
    PHASE3_VERSION,
    default_config,
    meas_dir,
    meas_file,
    truth_file,
)
from Utility import bd_chemostat as bd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _case_seed(base_seed: int, motif: str, meas_opt: str) -> int:
    """
    Deterministic per-case seed (stable across runs / machines).
    """
    s = f"{motif}|{meas_opt}".encode("utf-8")
    # crc32 -> 32-bit unsigned; fold into signed-ish positive int range
    return int(base_seed + (zlib.crc32(s) % 2_000_000_000))


def _compute_sigma(
    y_true: np.ndarray,
    meas_names: Tuple[str, ...],
    sigma_abs_map: dict,
    sigma_rel_map: dict,
    sigma_floor_map: dict,
) -> np.ndarray:
    """
    sigma_t[k,j] = sigma_abs[name_j] + sigma_rel[name_j]*abs(y_true[k,j]),
    then floored by sigma_floor[name_j].
    """
    y_true = np.asarray(y_true, dtype=float)
    T, ny = y_true.shape
    sigma_t = np.zeros((T, ny), dtype=float)

    for j, name in enumerate(meas_names):
        sa = float(sigma_abs_map.get(name, 0.0))
        sr = float(sigma_rel_map.get(name, 0.0))
        sf = float(sigma_floor_map.get(name, 0.0))
        sigma = sa + sr * np.abs(y_true[:, j])
        sigma = np.maximum(sigma, sf)
        sigma_t[:, j] = sigma

    return sigma_t


def main() -> None:
    cfg = default_config()
    _ensure_dir(meas_dir())

    print("=" * 80)
    print("Phase 3 - Milestone 2: Generating noisy measurements")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Meas out dir : {meas_dir()}")
    print(f"Motifs       : {cfg.motifs}")
    print(f"Meas opts    : {cfg.meas_opts}")
    print(f"Base seed    : {cfg.seed_measurements}")
    print("=" * 80)

    for motif_name in cfg.motifs:
        truth_path = truth_file(motif_name)
        if not truth_path.exists():
            raise FileNotFoundError(
                f"Missing truth file: {truth_path}. Run A_generate_truth.py first."
            )

        truth = np.load(truth_path, allow_pickle=True)
        t = np.asarray(truth["t"], dtype=float).reshape(-1)
        dt = float(np.asarray(truth["dt"]).reshape(()))
        x_true = np.asarray(truth["x_true"], dtype=float)
        u_D = np.asarray(truth["u_D"], dtype=float).reshape(-1, 1)

        if x_true.ndim != 2 or x_true.shape[1] != 6:
            raise ValueError(f"truth x_true must be (T,6). Got {x_true.shape}")
        if t.size != x_true.shape[0] or u_D.shape[0] != x_true.shape[0]:
            raise ValueError(
                f"Length mismatch in truth_{motif_name}: len(t)={t.size}, "
                f"x_true rows={x_true.shape[0]}, u_D rows={u_D.shape[0]}"
            )

        for meas_opt in cfg.meas_opts:
            H = bd.H(measurement_option=meas_opt)

            # Get measurement names (e.g., ('B','N','W'))
            meas_names = H.h(None, None, return_measurement_names=True)
            meas_names = tuple(str(s) for s in meas_names)
            ny = len(meas_names)

            # Compute y_true
            y_true = np.zeros((t.size, ny), dtype=float)
            for k in range(t.size):
                yk = H.h(x_true[k, :], u_D[k, :], return_measurement_names=False)
                y_true[k, :] = np.asarray(yk, dtype=float).reshape(-1)

            # Noise model
            sigma_t = _compute_sigma(
                y_true=y_true,
                meas_names=meas_names,
                sigma_abs_map=cfg.noise_sigma_abs,
                sigma_rel_map=cfg.noise_sigma_rel,
                sigma_floor_map=cfg.noise_sigma_floor,
            )

            # Deterministic RNG per case
            seed_case = _case_seed(cfg.seed_measurements, motif_name, meas_opt)
            rng = np.random.default_rng(seed_case)

            noise = rng.normal(loc=0.0, scale=sigma_t)
            y_meas = y_true + noise

            # Build constant R from a typical sigma (median over time)
            sigma_typ = np.median(sigma_t, axis=0)
            R = np.diag(sigma_typ ** 2)

            # Save
            out_path = meas_file(motif_name, meas_opt)
            np.savez(
                out_path,
                # Required contract
                t=t,
                dt=dt,
                y_true=y_true,
                y_meas=y_meas,
                R=R,
                meas_opt=str(meas_opt),
                meas_names=np.array(meas_names, dtype=object),
                # Recommended extras
                sigma_t=sigma_t,
                sigma_typ=sigma_typ,
                seed_case=int(seed_case),
                motif_name=str(motif_name),
                phase3_version=str(PHASE3_VERSION),
            )

            print(f"Saved: {out_path}")

    print("\nDone. Measurement files generated for all motifs and measurement options.")


if __name__ == "__main__":
    main()
