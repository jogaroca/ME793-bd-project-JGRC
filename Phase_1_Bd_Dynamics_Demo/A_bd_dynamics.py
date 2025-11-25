import sys
sys.path.append('../Utility')  # para poder importar bd_chemostat

import bd_chemostat
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Definir f y h (igual estilo que planar_drone)
    f = bd_chemostat.F().f
    h = bd_chemostat.H('h_BNW').h   # y = [B, N, W]
    
    t_sim, x_sim, u_sim, y_sim = bd_chemostat.simulate_bd(
        f, h,
        tsim_length=48.0,
        dt=0.01,
        D=0.12
    )
    
    # Separar salidas
    B = y_sim[:, 0]
    N = y_sim[:, 1]
    W = y_sim[:, 2]
    
    plt.figure()
    plt.plot(t_sim, B, label='B (biomasa total)')
    plt.plot(t_sim, N, label='N (nutriente)')
    plt.plot(t_sim, W, label='W (waste)')
    plt.xlabel('t [h]')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
