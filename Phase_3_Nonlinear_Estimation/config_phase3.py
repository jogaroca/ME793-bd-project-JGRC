"""
Phase 3 (Nonlinear Estimation) - Configuration + Data Contracts

Truth NPZ contract (results/phase3/truth/truth_<motif>.npz)
----------------------------------------------------------
Required keys:
  - t          : (T,)   time vector [h]
  - dt         : float  time step [h]
  - x_true     : (T,6)  states [Z,S1,S2,S3,N,W]
  - u_D        : (T,1)  dilution trajectory D(t) [h^-1]
  - motif_name : str

Recommended metadata keys:
  - T_final        : float
  - x0             : (6,)
  - state_names    : (6,)
  - input_names    : (1,)
  - params_bd      : dict (BdParameters as a dict)
  - phase3_version : str


Measurement NPZ contract (results/phase3/meas/meas_<motif>_<measopt>.npz)
------------------------------------------------------------------------
Required keys:
  - t           : (T,) time vector [h] (aligned with truth)
  - dt          : float
  - y_true      : (T,ny) noiseless measurement trajectory
  - y_meas      : (T,ny) noisy measurement trajectory
  - R           : (ny,ny) constant measurement covariance used by filters
  - meas_opt    : str (e.g., 'h_BNW', 'h_B', ...)
  - meas_names  : (ny,) measurement channel names

Recommended keys:
  - sigma_t     : (T,ny) per-sample noise std used to generate y_meas
  - sigma_typ   : (ny,) typical sigma used to build R
  - seed_case   : int
  - phase3_version : str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PHASE3_VERSION = "v1.0"


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def results_dir() -> Path:
    return get_project_root() / "results" / "phase3"


def truth_dir() -> Path:
    return results_dir() / "truth"


def meas_dir() -> Path:
    return results_dir() / "meas"


def truth_file(motif_name: str) -> Path:
    return truth_dir() / f"truth_{motif_name}.npz"


def meas_file(motif_name: str, meas_opt: str) -> Path:
    safe_meas = meas_opt.replace("/", "_")
    return meas_dir() / f"meas_{motif_name}_{safe_meas}.npz"


@dataclass(frozen=True)
class Phase3Config:
    # --- Truth simulation settings ---
    T_final: float = 48.0     # [h]
    dt_truth: float = 0.05    # [h]
    x0_nominal: Tuple[float, float, float, float, float, float] = (1.0, 0.2, 0.2, 0.2, 5.0, 0.0)

    # --- Motifs to generate (must exist in Utility/bd_motifs.py::MOTIFS) ---
    motifs: List[str] = None

    # --- Measurement sets to generate (must exist in Utility/bd_chemostat.H) ---
    meas_opts: List[str] = None

    # --- Measurement noise model (per-channel): sigma = sigma_abs + sigma_rel*|y_true| ---
    noise_sigma_abs: Dict[str, float] = None
    noise_sigma_rel: Dict[str, float] = None
    noise_sigma_floor: Dict[str, float] = None

    # --- Base seed (case-specific seed derived deterministically) ---
    seed_measurements: int = 12345

    def __post_init__(self):
        if self.motifs is None:
            object.__setattr__(self, "motifs", ["const_0p12", "step_24h", "sin_12h", "sin_12h_big"])

        if self.meas_opts is None:
            object.__setattr__(self, "meas_opts", ["h_BNW", "h_BN", "h_B", "h_NW", "h_N", "h_W"])

        # Sensible defaults (tweak later):
        # - B and W: mostly relative noise
        # - N: mostly absolute noise (mM)
        if self.noise_sigma_abs is None:
            object.__setattr__(self, "noise_sigma_abs", {"B": 0.02, "N": 0.05, "W": 0.01})
        if self.noise_sigma_rel is None:
            object.__setattr__(self, "noise_sigma_rel", {"B": 0.05, "N": 0.00, "W": 0.10})
        if self.noise_sigma_floor is None:
            object.__setattr__(self, "noise_sigma_floor", {"B": 1e-6, "N": 1e-6, "W": 1e-6})


def default_config() -> Phase3Config:
    return Phase3Config()
