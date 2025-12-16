"""
Phase 3 (Nonlinear Estimation) - Configuration + Data Contract (Milestone 1)

This file defines:
  - Project root discovery
  - Standard results directories
  - Truth data contract for saved NPZ files
  - Core simulation settings (T_final, dt_truth, x0, motifs)

Truth NPZ contract (results/phase3/truth/truth_<motif>.npz):

Required keys:
  - t          : (T,)   time vector [h]
  - dt         : float  time step [h]
  - x_true     : (T,6)  states [Z,S1,S2,S3,N,W]
  - u_D        : (T,1)  dilution input trajectory D(t) [h^-1]
  - motif_name : str

Recommended metadata keys:
  - T_final        : float
  - x0             : (6,)
  - state_names    : (6,)
  - input_names    : (1,)
  - params_bd      : dict (BdParameters as a dict)
  - phase3_version : str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


PHASE3_VERSION = "v1.0"


def get_project_root() -> Path:
    """
    Returns the project root as the parent directory of Phase_3_Nonlinear_Estimation/.
    Works regardless of current working directory.
    """
    here = Path(__file__).resolve()
    return here.parents[1]


def results_dir() -> Path:
    return get_project_root() / "results" / "phase3"


def truth_dir() -> Path:
    return results_dir() / "truth"


def truth_file(motif_name: str) -> Path:
    return truth_dir() / f"truth_{motif_name}.npz"


@dataclass(frozen=True)
class Phase3Config:
    # Simulation horizon and save step for "truth" trajectories
    T_final: float = 48.0     # [h]
    dt_truth: float = 0.05    # [h] stored time-step for truth

    # Nominal initial condition: x = [Z, S1, S2, S3, N, W]
    x0_nominal: tuple = (1.0, 0.2, 0.2, 0.2, 5.0, 0.0)

    # Motifs to generate (names must exist in Utility/bd_motifs.py::MOTIFS)
    motifs: List[str] = None

    def __post_init__(self):
        if self.motifs is None:
            object.__setattr__(
                self,
                "motifs",
                ["const_0p12", "step_24h", "sin_12h", "sin_12h_big"],
            )


def default_config() -> Phase3Config:
    return Phase3Config()
