import os
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2_empirical")

MOTIFS = ["const_0p12", "step_24h", "sin_12h"]
MEAS_OPTIONS = ["h_BNW", "h_BN", "h_B", "h_NW", "h_N", "h_W"]

def load_metrics(motif, meas):
    fname = f"obs_{motif}_{meas}.npz"
    path = os.path.join(RESULTS_DIR, fname)
    data = np.load(path)
    var_Z = float(data["var_Z"])
    I_Z = float(data["I_Z"])
    return var_Z, I_Z

def main():
    print("Z-specific observability summary (Chernoff inverse):")
    table_varZ = []

    for motif in MOTIFS:
        print("=" * 72)
        print(f"Motif: {motif}")
        print("  meas     log10(var_Z)    var_Z         I_Z")
        for meas in MEAS_OPTIONS:
            var_Z, I_Z = load_metrics(motif, meas)
            table_varZ.append((motif, meas, var_Z, I_Z))
            print(f"  {meas:6s}  {np.log10(var_Z):8.3f}  {var_Z:10.3e}  {I_Z:10.3e}")

    # Example bar plot: log10(var_Z) for each meas, per motif
    for motif in MOTIFS:
        vals = []
        labels = []
        for meas in MEAS_OPTIONS:
            var_Z, _ = load_metrics(motif, meas)
            vals.append(np.log10(var_Z))
            labels.append(meas)
        plt.figure()
        plt.bar(labels, vals)
        plt.ylabel("log10(var_Z)")
        plt.title(f"Z minimum error variance vs measurement set\n(motif={motif})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
