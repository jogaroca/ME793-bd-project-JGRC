# Utility/bd_chemostat.py

from dataclasses import dataclass
from typing import Optional, Sequence, Callable
import numpy as np
from scipy.integrate import odeint


# -------------------------------------------------------------------
# Bd model parameters (as in the Phase 2 PDF)
# -------------------------------------------------------------------
@dataclass
class BdParameters:
    # Core rates [h^-1]
    f0: float = 4.0      # reproduction factor S3 -> Z
    m0: float = 0.10     # encystment Z -> S1
    mu_Z: float = 0.15   # mortality of Z
    mu_S: float = 0.05   # mortality of S_i, i=1,2,3

    # Maturation kinetics k_m(N,W)
    k_max: float = 0.60  # maximum maturation rate
    K_N: float = 0.5     # nutrient half-saturation (mM)
    K_W: float = 1.0     # inhibition scale by W
    h_W: float = 1.0     # inhibition “steepness” (dimensionless)

    # Yield / waste
    Y_m: float = 5e3     # (stage-units / mM)
    sigma_m: float = 0.02
    sigma_Z: float = 0.01
    sigma_S: float = 0.005

    # Nutrient in the influent
    N0: float = 5.0      # mM

# -------------------------------------------------------------------
# k_m(N,W)
# -------------------------------------------------------------------
def maturation_rate_km(N, W, p: BdParameters):
    """Maturation rate k_m(N,W) as defined in the project document."""
    N = max(float(N), 0.0)
    W = max(float(W), 0.0)
    return p.k_max * (N / (p.K_N + N)) * (1.0 / (1.0 + (W / p.K_W) ** p.h_W))

# -------------------------------------------------------------------
# Continuous dynamics F (analogous to planar_drone.F)
# -------------------------------------------------------------------
class F(object):
    """
    Continuous dynamics of the Bd chemostat.

    State (len = 6):
        x = [Z, S1, S2, S3, N, W]

    Input (len = 1):
        u = [D]  (chemostat dilution rate, h^-1)
    """
    def __init__(self, params: Optional[BdParameters] = None):
        self.params = params or BdParameters()

    def f(self, x_vec, u_vec, return_state_names: bool = False):
        # Same as in planar_drone: if called with return_state_names=True,
        # we only return the names.
        if return_state_names:
            return ('Z', 'S1', 'S2', 'S3', 'N', 'W')

        if x_vec is None:
            raise ValueError("x_vec cannot be None when return_state_names=False")

        x = np.asarray(x_vec, dtype=float).reshape(-1)
        if x.size != 6:
            raise ValueError(f"x_vec must have length 6, got {x.size}")

        # Input (dilution D)
        if u_vec is None:
            D = 0.0
        else:
            u = np.asarray(u_vec, dtype=float).reshape(-1)
            if u.size < 1:
                raise ValueError("u_vec must contain at least one element (D)")
            D = float(u[0])

        Z, S1, S2, S3, N, W = x
        p = self.params

        k_m = maturation_rate_km(N, W, p)
        S_sum = S1 + S2 + S3

        # Equations (1) in the PDF, with D = u_D
        dZ  = p.f0 * k_m * S3 - (p.m0 + p.mu_Z + D) * Z
        dS1 = p.m0 * Z       - (k_m + p.mu_S + D) * S1
        dS2 = k_m * S1       - (k_m + p.mu_S + D) * S2
        dS3 = k_m * S2       - (k_m + p.mu_S + D) * S3
        dN  = D * (p.N0 - N) - (1.0 / p.Y_m) * k_m * S_sum
        dW  = -D * W + p.sigma_m * k_m * S_sum \
              + p.sigma_Z * p.mu_Z * Z \
              + p.sigma_S * p.mu_S * S_sum

        return np.array([dZ, dS1, dS2, dS3, dN, dW], dtype=float)

# -------------------------------------------------------------------
# Measurement functions H (analogous to planar_drone.H)
# -------------------------------------------------------------------
class H(object):
    """
    Measurement functions for the Bd chemostat.

    Base measurement model (from the PDF):
        y = [B, N, W]^T
    with B = Z + S1 + S2 + S3 (total biomass).
    """
    def __init__(self, measurement_option: str = 'h_BNW'):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names: bool = False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # --- different sensor combinations ---

    def h_BNW(self, x_vec, u_vec, return_measurement_names: bool = False):
        if return_measurement_names:
            return ('B', 'N', 'W')
        Z, S1, S2, S3, N, W = np.asarray(x_vec, dtype=float).reshape(-1)
        B = Z + S1 + S2 + S3
        return np.array([B, N, W], dtype=float)

    def h_BN(self, x_vec, u_vec, return_measurement_names: bool = False):
        if return_measurement_names:
            return ('B', 'N')
        Z, S1, S2, S3, N, W = np.asarray(x_vec, dtype=float).reshape(-1)
        B = Z + S1 + S2 + S3
        return np.array([B, N], dtype=float)

    def h_B(self, x_vec, u_vec, return_measurement_names: bool = False):
        if return_measurement_names:
            return ('B',)
        Z, S1, S2, S3, N, W = np.asarray(x_vec, dtype=float).reshape(-1)
        B = Z + S1 + S2 + S3
        return np.array([B], dtype=float)

    def h_NW(self, x_vec, u_vec, return_measurement_names: bool = False):
        if return_measurement_names:
            return ('N', 'W')
        Z, S1, S2, S3, N, W = np.asarray(x_vec, dtype=float).reshape(-1)
        return np.array([N, W], dtype=float)

    def h_N(self, x_vec, u_vec, return_measurement_names: bool = False):
        if return_measurement_names:
            return ('N',)
        Z, S1, S2, S3, N, W = np.asarray(x_vec, dtype=float).reshape(-1)
        return np.array([N], dtype=float)

    def h_W(self, x_vec, u_vec, return_measurement_names: bool = False):
        if return_measurement_names:
            return ('W',)
        Z, S1, S2, S3, N, W = np.asarray(x_vec, dtype=float).reshape(-1)
        return np.array([W], dtype=float)

# -------------------------------------------------------------------
# Simple simulator (to test in the dynamics notebook)
# -------------------------------------------------------------------
def simulate_bd(
    f,
    h,
    tsim_length: float = 48.0,
    dt: float = 0.01,
    x0: Optional[Sequence[float]] = None,
    D: float = 0.12,
    D_fun: Optional[Callable[[float], float]] = None,
):
    """
    Simple Bd chemostat simulator using scipy.integrate.odeint.

    Parameters
    ----------
    f : callable
        State dynamics function x_dot = f(x, u).
    h : callable
        Measurement function y = h(x, u).
    tsim_length : float
        Final time [h].
    dt : float
        Time step [h] used for saving trajectories.
    x0 : array_like, shape (6,), optional
        Initial state. If None, a default initial state is used.
    D : float, optional
        Constant dilution rate [h^-1]. Used only if D_fun is None.
    D_fun : callable, optional
        Time-varying dilution rate u_D(t). If provided, it overrides D.

    Returns
    -------
    t_sim : ndarray, shape (T,)
    x_sim : ndarray, shape (T, 6)
    u_sim : ndarray, shape (T, 1)
    y_sim : ndarray, shape (T, n_outputs)
    """
    if x0 is None:
        # Default initial condition
        x0 = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(6,)

    # Time grid
    t_sim = np.arange(0.0, tsim_length + dt, dt)

    # Helper to get D(t)
    def D_of_t(t: float) -> float:
        if D_fun is None:
            return float(D)
        return float(D_fun(float(t)))

    # Input trajectory (for bookkeeping / plotting)
    D_vals = np.array([D_of_t(t) for t in t_sim], dtype=float)
    u_sim = D_vals.reshape(-1, 1)

    # ODE RHS
    def rhs(x, t):
        D_t = D_of_t(t)
        return f(x, [D_t])

    # Integrate
    x_sim = odeint(rhs, x0, t_sim)

    # Build outputs
    y_list = []
    for xi, ui in zip(x_sim, u_sim):
        yi = h(xi, ui)
        y_list.append(np.asarray(yi, dtype=float).reshape(-1))
    y_sim = np.vstack(y_list)

    return t_sim, x_sim, u_sim, y_sim
