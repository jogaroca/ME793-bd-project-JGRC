import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Añadir ../Utility al path de Python (para importar bd_chemostat)
CURRENT_DIR = os.path.dirname(__file__)
UTILITY_DIR = os.path.join(CURRENT_DIR, "..", "Utility")
sys.path.append(UTILITY_DIR)

import bd_chemostat as bd


def main():
    # Definir dinámica y medición
    f = bd.F().f
    h = bd.H("h_BNW").h  # y = [B, N, W]

    # Simular
    t_sim, x_sim, u_sim, y_sim = bd.simulate_bd(
        f,
        h,
        tsim_length=48.0,
        dt=0.01,
        D=0.12,
    )

    B = y_sim[:, 0]
    N = y_sim[:, 1]
    W = y_sim[:, 2]

    # Graficar
    fig, ax = plt.subplots()
    ax.plot(t_sim, B, label="B (biomasa total)")
    ax.plot(t_sim, N, label="N (nutriente)")
    ax.plot(t_sim, W, label="W (waste)")
    ax.set_xlabel("t [h]")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
