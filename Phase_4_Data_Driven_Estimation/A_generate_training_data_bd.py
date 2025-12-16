
"""
Phase 4 - Data-driven estimation (Option 1, simulated data)
A_generate_training_data_bd.py

Generates training/testing datasets from the true Bd chemostat simulator.

Outputs:
  data/phase4/train/bd_train.npz
  data/phase4/test/bd_test.npz

Saved keys (NPZ):
  t, dt
  X        : (Ntraj, T, 6)   true states
  U        : (Ntraj, T, 1)   input (dilution)
  Y_true   : (Ntraj, T, ny)  noiseless measurements
  Y_meas   : (Ntraj, T, ny)  noisy measurements
  R        : (ny, ny)        constant measurement covariance used (typical)
  motif    : (Ntraj,)        motif name per trajectory
  x0       : (Ntraj, 6)      initial conditions
  state_names, meas_names, meas_opt
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import asdict
import numpy as np

# ---------------------------------------------------------------------
# Robust path handling
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility import bd_chemostat as bd
from Utility import bd_motifs as motifs


# -------------------------
# USER SETTINGS (MVP)
# -------------------------
T_FINAL = 48.0
DT = 0.05

MEAS_OPT = "h_BNW"  # keep same as Phase 3 for consistency
MOTIF_NAMES = ["const_0p12", "step_24h", "sin_12h", "sin_12h_big"]

N_TRAIN_PER_MOTIF = 40
N_TEST_PER_MOTIF  = 10

SEED = 12345

# Initial condition distribution (relative lognormal noise around nominal)
X0_NOMINAL = np.array([1.0, 0.2, 0.2, 0.2, 5.0, 0.0], dtype=float)
X0_LOGN_SIGMA = 0.35  # 0.2–0.5 is reasonable; higher => more diversity
W0_MAX = 0.20         # small initial waste variability

# Measurement noise model per channel: sigma = abs + rel*|y|, with floor
NOISE_ABS = {"B": 0.02, "N": 0.05, "W": 0.01}
NOISE_REL = {"B": 0.05, "N": 0.00, "W": 0.10}
NOISE_FLOOR = {"B": 1e-6, "N": 1e-6, "W": 1e-6}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sample_x0(rng: np.random.Generator, x0_nominal: np.ndarray, N0: float) -> np.ndarray:
    """
    Sample positive initial conditions around nominal with lognormal noise.
    """
    x0 = np.array(x0_nominal, dtype=float).copy()

    # lognormal factor for positive states
    fac = np.exp(X0_LOGN_SIGMA * rng.standard_normal(size=6))
    x0 = x0 * fac

    # Nutrient: keep within a sensible range near influent N0
    x0[4] = rng.uniform(0.6 * N0, 1.0 * N0)

    # Waste: small initial waste
    x0[5] = rng.uniform(0.0, W0_MAX)

    # Avoid exact zeros/negatives
    x0 = np.maximum(x0, 1e-10)
    return x0


def add_measurement_noise(
    rng: np.random.Generator,
    y_true: np.ndarray,
    meas_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y_meas, R_typical) where R_typical is a constant diag covariance
    built from typical sigma over the trajectory.
    """
    T, ny = y_true.shape
    sigma_t = np.zeros((T, ny), dtype=float)

    for j, name in enumerate(meas_names):
        abs_s = float(NOISE_ABS.get(name, 0.01))
        rel_s = float(NOISE_REL.get(name, 0.00))
        flo_s = float(NOISE_FLOOR.get(name, 1e-6))
        sigma_t[:, j] = np.maximum(abs_s + rel_s * np.abs(y_true[:, j]), flo_s)

    noise = rng.normal(loc=0.0, scale=sigma_t)
    y_meas = y_true + noise

    sigma_typ = np.median(sigma_t, axis=0)
    R = np.diag(sigma_typ ** 2)
    return y_meas, R


def simulate_one(f, h, D_fun, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, X, U, Y = bd.simulate_bd(
        f=f,
        h=h,
        tsim_length=float(T_FINAL),
        dt=float(DT),
        x0=x0,
        D=0.0,
        D_fun=D_fun,
    )
    return np.asarray(t), np.asarray(X), np.asarray(U), np.asarray(Y)


def main() -> None:
    rng = np.random.default_rng(SEED)

    out_train = PROJECT_ROOT / "data" / "phase4" / "train"
    out_test  = PROJECT_ROOT / "data" / "phase4" / "test"
    ensure_dir(out_train)
    ensure_dir(out_test)

    # Validate motifs
    for m in MOTIF_NAMES:
        if m not in motifs.MOTIFS:
            raise KeyError(f"Motif '{m}' not found. Available: {sorted(list(motifs.MOTIFS.keys()))}")

    # True model + measurement function
    f_obj = bd.F()
    h_obj = bd.H(measurement_option=MEAS_OPT)
    f = f_obj.f
    h = h_obj.h

    state_names = list(f_obj.f(None, None, return_state_names=True))
    meas_names = list(h_obj.h(None, None, return_measurement_names=True))

    # Helper to generate split
    def build_split(n_per_motif: int):
        X_list, U_list, Yt_list, Ym_list, motif_list, x0_list = [], [], [], [], [], []
        R_ref = None

        for motif_name in MOTIF_NAMES:
            D_fun = motifs.MOTIFS[motif_name]
            for _ in range(n_per_motif):
                x0 = sample_x0(rng, X0_NOMINAL, f_obj.params.N0)
                t, X, U, Y_true = simulate_one(f, h, D_fun, x0)

                Y_meas, R = add_measurement_noise(rng, Y_true, meas_names)
                if R_ref is None:
                    R_ref = R

                X_list.append(X)
                U_list.append(U)
                Yt_list.append(Y_true)
                Ym_list.append(Y_meas)
                motif_list.append(motif_name)
                x0_list.append(x0)

        X_arr = np.stack(X_list, axis=0)
        U_arr = np.stack(U_list, axis=0)
        Yt_arr = np.stack(Yt_list, axis=0)
        Ym_arr = np.stack(Ym_list, axis=0)
        motif_arr = np.array(motif_list, dtype=object)
        x0_arr = np.stack(x0_list, axis=0)

        return t, X_arr, U_arr, Yt_arr, Ym_arr, R_ref, motif_arr, x0_arr

    # Build datasets
    t, Xtr, Utr, Ytr_t, Ytr_m, Rtr, motif_tr, x0_tr = build_split(N_TRAIN_PER_MOTIF)
    t2, Xte, Ute, Yte_t, Yte_m, Rte, motif_te, x0_te = build_split(N_TEST_PER_MOTIF)

    # Save
    train_path = out_train / "bd_train.npz"
    test_path  = out_test / "bd_test.npz"

    np.savez_compressed(
        train_path,
        t=t, dt=float(DT),
        X=Xtr, U=Utr, Y_true=Ytr_t, Y_meas=Ytr_m,
        R=Rtr,
        motif=motif_tr, x0=x0_tr,
        state_names=np.array(state_names, dtype=object),
        meas_names=np.array(meas_names, dtype=object),
        meas_opt=str(MEAS_OPT),
        params_bd=asdict(f_obj.params),
        seed=int(SEED),
    )

    np.savez_compressed(
        test_path,
        t=t2, dt=float(DT),
        X=Xte, U=Ute, Y_true=Yte_t, Y_meas=Yte_m,
        R=Rte,
        motif=motif_te, x0=x0_te,
        state_names=np.array(state_names, dtype=object),
        meas_names=np.array(meas_names, dtype=object),
        meas_opt=str(MEAS_OPT),
        params_bd=asdict(f_obj.params),
        seed=int(SEED),
    )

    print("=" * 80)
    print("Phase 4 - Training data generated")
    print(f"Train: {train_path}  X={Xtr.shape} U={Utr.shape} Y={Ytr_m.shape}")
    print(f"Test : {test_path}   X={Xte.shape} U={Ute.shape} Y={Yte_m.shape}")
    print(f"meas_opt={MEAS_OPT}, motifs={MOTIF_NAMES}, dt={DT}, T_final={T_FINAL}")
    print("=" * 80)


if __name__ == "__main__":
    main()
