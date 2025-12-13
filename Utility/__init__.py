"""Utility

Reusable code for the ME793 Bd chemostat project.

Design intent (mirrors the course repository pattern):
- Keep reusable dynamics/measurement models, motifs, observability, and filters here.
- Keep Phase_*/ scripts thin and focused on experiments and plots.

If you want this to be importable from anywhere, install the repo once:
    pip install -e .
"""

__version__ = "0.1.0"

# Public re-exports (keep this lightweight)
from .bd_chemostat import BdParameters, F, H, simulate_bd  # noqa: F401
from .bd_ekf import BdEKF  # noqa: F401
