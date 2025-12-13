import numpy as np
import matplotlib.pyplot as plt

from Utility import bd_chemostat as bd


def main():
    # Define dynamics and measurements
    f = bd.F().f
    h = bd.H("h_BNW").h  # y = [B, N, W]

    # Simulate
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

    # Plot
    fig, ax = plt.subplots()
    ax.plot(t_sim, B, label="B (total biomass)")
    ax.plot(t_sim, N, label="N (nutrient)")
    ax.plot(t_sim, W, label="W (waste)")
    ax.set_xlabel("t [h]")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()