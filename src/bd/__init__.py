"""
Bd chemostat model package for the ME793 nonlinear estimation project.

This package bundles the continuous-time dynamics, measurement model,
control-input motifs, simulation utilities and empirical observability
tools for the Bd chemostat example system.

Typical usage in a notebook::

    from bd import (
        model, controls, measurements, simulator, observability
    )

    # Simulate
    x0 = model.default_initial_state()
    t, X, U, Y = simulator.simulate_bd_odeint(
        x0=x0,
        uD_fun=controls.uD_const_012,
    )

    # Build pybounds simulator and empirical observability matrices
    pb_sim = simulator.make_pybounds_simulator(dt=t[1] - t[0])
    SEOM, O_sliding = observability.build_sliding_empirical_observability_matrix(
        pb_sim, t, X, U, window_size=6
    )
"""

from .model import theta, k_m, f, STATE_NAMES, default_initial_state
from .controls import (
    D_MIN,
    D_MAX,
    uD_const_012,
    uD_step_24h,
    uD_sin_12h,
    uD_demo_steps,
    make_u_func,
    u_func_const,
    u_func_steps,
)
from .measurements import (
    h,
    meas_setA,
    MEASUREMENT_NAMES,
    build_noise_covariance_from_trajectory,
)
from .simulator import (
    default_time_grid,
    simulate_bd_odeint,
    make_pybounds_simulator,
)
from .observability import (
    estimate_measurement_noise_variances_from_outputs,
    build_sliding_empirical_observability_matrix,
    build_single_window_fisher,
    build_sliding_fisher_observability,
)

__all__ = [
    # model
    "theta",
    "k_m",
    "f",
    "STATE_NAMES",
    "default_initial_state",
    # controls
    "D_MIN",
    "D_MAX",
    "uD_const_012",
    "uD_step_24h",
    "uD_sin_12h",
    "uD_demo_steps",
    "make_u_func",
    "u_func_const",
    "u_func_steps",
    # measurements
    "h",
    "meas_setA",
    "MEASUREMENT_NAMES",
    "build_noise_covariance_from_trajectory",
    # simulator
    "default_time_grid",
    "simulate_bd_odeint",
    "make_pybounds_simulator",
    # observability
    "estimate_measurement_noise_variances_from_outputs",
    "build_sliding_empirical_observability_matrix",
    "build_single_window_fisher",
    "build_sliding_fisher_observability",
]
