# Utility/bd_chemostat.py

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from scipy.integrate import odeint

# -------------------------------------------------------------------
# Parámetros del modelo Bd (tal como en el PDF de la fase 2)
# -------------------------------------------------------------------
@dataclass
class BdParameters:
    # Tasas núcleo [h^-1]
    f0: float = 4.0      # factor de reproducción S3 -> Z
    m0: float = 0.10     # encistamiento Z -> S1
    mu_Z: float = 0.15   # mortalidad de Z
    mu_S: float = 0.05   # mortalidad de S_i, i=1,2,3

    # Cinética de maduración k_m(N,W)
    k_max: float = 0.60  # tasa máxima de maduración
    K_N: float = 0.5     # semisaturación nutriente (mM)
    K_W: float = 1.0     # escala de inhibición por W
    h_W: float = 1.0     # “steepness” de inhibición (adim.)

    # Rendimiento / desperdicio
    Y_m: float = 5e3     # (stage-units / mM)
    sigma_m: float = 0.02
    sigma_Z: float = 0.01
    sigma_S: float = 0.005

    # Nutriente en el influente
    N0: float = 5.0      # mM

# -------------------------------------------------------------------
# k_m(N,W)
# -------------------------------------------------------------------
def maturation_rate_km(N, W, p: BdParameters):
    """Tasa de maduración k_m(N,W) según el documento del proyecto."""
    N = max(float(N), 0.0)
    W = max(float(W), 0.0)
    return p.k_max * (N / (p.K_N + N)) * (1.0 / (1.0 + (W / p.K_W) ** p.h_W))

# -------------------------------------------------------------------
# Dinámica continua F (análoga a planar_drone.F)
# -------------------------------------------------------------------
class F(object):
    """
    Dinámica continua del quimiostato Bd.

    Estado (len = 6):
        x = [Z, S1, S2, S3, N, W]

    Entrada (len = 1):
        u = [D]  (tasa de dilución del quimiostato, h^-1)
    """
    def __init__(self, params: Optional[BdParameters] = None):
        self.params = params or BdParameters()

    def f(self, x_vec, u_vec, return_state_names: bool = False):
        # Igual que en planar_drone: si se llama con return_state_names=True,
        # sólo devolvemos los nombres.
        if return_state_names:
            return ('Z', 'S1', 'S2', 'S3', 'N', 'W')

        if x_vec is None:
            raise ValueError("x_vec no puede ser None cuando return_state_names=False")

        x = np.asarray(x_vec, dtype=float).reshape(-1)
        if x.size != 6:
            raise ValueError(f"x_vec debe tener longitud 6, recibió {x.size}")

        # Entrada (dilución D)
        if u_vec is None:
            D = 0.0
        else:
            u = np.asarray(u_vec, dtype=float).reshape(-1)
            if u.size < 1:
                raise ValueError("u_vec debe contener al menos un elemento (D)")
            D = float(u[0])

        Z, S1, S2, S3, N, W = x
        p = self.params

        k_m = maturation_rate_km(N, W, p)
        S_sum = S1 + S2 + S3

        # Ecuaciones (1) del PDF, con D = u_D
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
# Funciones de medición H (análogas a planar_drone.H)
# -------------------------------------------------------------------
class H(object):
    """
    Funciones de medición para el quimiostato Bd.

    Modelo de medición base (del PDF):
        y = [B, N, W]^T
    con B = Z + S1 + S2 + S3 (biomasa total).
    """
    def __init__(self, measurement_option: str = 'h_BNW'):
        self.measurement_option = measurement_option

    def h(self, x_vec, u_vec, return_measurement_names: bool = False):
        h_func = self.__getattribute__(self.measurement_option)
        return h_func(x_vec, u_vec, return_measurement_names=return_measurement_names)

    # --- distintas combinaciones de sensores ---

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
# Simulador sencillo (para probar en el notebook de dinámica)
# -------------------------------------------------------------------
def simulate_bd(f, h, tsim_length: float = 48.0, dt: float = 0.01,
                x0: Optional[Sequence[float]] = None,
                D: float = 0.12):
    """
    Simulador sencillo del quimiostato Bd usando scipy.integrate.odeint.

    Devuelve:
        t_sim : (T,)
        x_sim : (T, 6)
        u_sim : (T, 1)
        y_sim : (T, n_y)
    """
    if x0 is None:
        # Estados iniciales del setup de simulación en el PDF
        x0 = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(6,)

    t_sim = np.arange(0.0, tsim_length + dt, dt)
    u_sim = np.full((t_sim.size, 1), D, dtype=float)

    def rhs(x, t):
        # f espera (x_vec, u_vec)
        return f(x, [D])

    x_sim = odeint(rhs, x0, t_sim)

    # Construir salidas
    y_list = []
    for xi, ui in zip(x_sim, u_sim):
        yi = h(xi, ui)
        y_list.append(np.asarray(yi, dtype=float).reshape(-1))
    y_sim = np.vstack(y_list)

    return t_sim, x_sim, u_sim, y_sim
