"""
Phase 5 (Future Work) - configuration + output paths

Goal:
  Compare open-loop motifs vs a simple adaptive dilution policy (closed-loop).

Outputs:
  results/phase5/sims/sim_<name>.npz
  results/phase5/figures/*.png
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def results_dir() -> Path:
    return get_project_root() / "results" / "phase5"


def sims_dir() -> Path:
    return results_dir() / "sims"


def figures_dir() -> Path:
    return results_dir() / "figures"


def sim_file(name: str) -> Path:
    safe = name.replace("/", "_")
    return sims_dir() / f"sim_{safe}.npz"


@dataclass(frozen=True)
class Phase5Config:
    # Simulation window
    T_final: float = 48.0
    dt: float = 0.05  # keep consistent with Phase 3 truth dt if you want

    # Initial condition (same default as Utility.bd_chemostat.simulate_bd)
    x0: Tuple[float, float, float, float, float, float] = (1.0, 0.2, 0.2, 0.2, 5.0, 0.0)

    # Open-loop motifs to compare (must exist in Utility/bd_motifs.py::MOTIFS)
    open_loop_motifs: List[str] = None

    # Adaptive policy parameters (normalized + saturation)
    D_min: float = 0.0
    D_max: float = 0.25     # matches sin_12h_big range [0.05, 0.25]
    D0: float = 0.10        # baseline ("minimal renewal")

    # Gains (units: h^-1). Start small; adjust by eyeballing stability/avoid chatter.
    k_B: float = 0.06
    k_W: float = 0.04

    # Setpoints/thresholds (if None, B* defaults to B(t=0); W* must be > 0)
    B_star: Optional[float] = None
    W_star: float = 0.20

    # Numerics
    rtol: float = 1e-6
    atol: float = 1e-9
    eps: float = 1e-12

    def __post_init__(self):
        if self.open_loop_motifs is None:
            object.__setattr__(self, "open_loop_motifs", ["const_0p12", "step_24h", "sin_12h", "sin_12h_big"])


def default_config() -> Phase5Config:
    return Phase5Config()
