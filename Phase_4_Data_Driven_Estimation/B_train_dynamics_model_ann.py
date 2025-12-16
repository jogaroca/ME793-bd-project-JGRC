"""
Phase 4 - Data-driven estimation (Option 1, simulated data)
B_train_dynamics_model_ann.py

Trains a discrete-time ANN dynamics model:
    x_{k+1} = f_theta(x_k, u_k)

Reads:
  data/phase4/train/bd_train.npz
  data/phase4/test/bd_test.npz

Writes:
  models/phase4/dynamics_ann/model.keras
  models/phase4/dynamics_ann/scaler.npz
  results/phase4/metrics/ann_one_step_rmse.csv
  results/phase4/figures/ann_training_loss.png
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import csv
import numpy as np

# --- robust paths ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_PATH = PROJECT_ROOT / "data" / "phase4" / "train" / "bd_train.npz"
TEST_PATH  = PROJECT_ROOT / "data" / "phase4" / "test"  / "bd_test.npz"

MODEL_DIR  = PROJECT_ROOT / "models" / "phase4" / "dynamics_ann"
METRICS_DIR = PROJECT_ROOT / "results" / "phase4" / "metrics"
FIG_DIR     = PROJECT_ROOT / "results" / "phase4" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- training settings (MVP) ---
SEED = 12345
EPOCHS = 100
BATCH_SIZE = 1024
VAL_FRAC = 0.10
LR = 1e-3

# --- TF import after path setup ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(SEED)
tf.random.set_seed(SEED)


def flatten_one_step_pairs(X: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised pairs (x_k,u_k) -> x_{k+1}.
    X: (Ntraj, T, nx)
    U: (Ntraj, T, nu)
    Returns:
      IN : (Ntraj*(T-1), nx+nu)
      OUT: (Ntraj*(T-1), nx)
    """
    Xk = X[:, :-1, :]
    Xkp1 = X[:, 1:, :]
    Uk = U[:, :-1, :]

    Ntraj, Tm1, nx = Xk.shape
    nu = Uk.shape[-1]

    IN = np.concatenate([Xk, Uk], axis=-1).reshape(Ntraj * Tm1, nx + nu)
    OUT = Xkp1.reshape(Ntraj * Tm1, nx)
    return IN, OUT


def standardize_fit(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean, std


def standardize_apply(A: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (A - mean) / std


def unstandardize(Az: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return Az * std + mean


def build_model(n_in: int, n_out: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(n_in,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_out, activation="linear"),
    ])
    opt = keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss="mse")
    return model


def main() -> None:
    # Load data
    train = np.load(TRAIN_PATH, allow_pickle=True)
    test  = np.load(TEST_PATH, allow_pickle=True)

    Xtr, Utr = train["X"], train["U"]
    Xte, Ute = test["X"], test["U"]

    state_names = train["state_names"].tolist()

    INtr, OUTtr = flatten_one_step_pairs(Xtr, Utr)
    INte, OUTte = flatten_one_step_pairs(Xte, Ute)

    # Fit scalers on TRAIN only
    in_mean, in_std = standardize_fit(INtr)
    out_mean, out_std = standardize_fit(OUTtr)

    INtr_z = standardize_apply(INtr, in_mean, in_std)
    OUTtr_z = standardize_apply(OUTtr, out_mean, out_std)
    INte_z = standardize_apply(INte, in_mean, in_std)

    # Train/val split
    n = INtr_z.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(VAL_FRAC * n)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    x_train, y_train = INtr_z[tr_idx], OUTtr_z[tr_idx]
    x_val,   y_val   = INtr_z[val_idx], OUTtr_z[val_idx]

    # Build + train
    model = build_model(n_in=INtr_z.shape[1], n_out=OUTtr_z.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks,
    )

    # Evaluate on TEST (one-step)
    OUTte_pred_z = model.predict(INte_z, batch_size=4096, verbose=0)
    OUTte_pred = unstandardize(OUTte_pred_z, out_mean, out_std)

    err = OUTte_pred - OUTte
    rmse_by_state = np.sqrt(np.mean(err**2, axis=0))

    # Save model + scaler
    model_path = MODEL_DIR / "model.keras"
    scaler_path = MODEL_DIR / "scaler.npz"
    model.save(model_path)

    np.savez_compressed(
        scaler_path,
        in_mean=in_mean, in_std=in_std,
        out_mean=out_mean, out_std=out_std,
        state_names=np.array(state_names, dtype=object),
    )

    # Save metrics CSV
    metrics_path = METRICS_DIR / "ann_one_step_rmse.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "rmse_one_step"])
        for s, v in zip(state_names, rmse_by_state):
            w.writerow([s, float(v)])

    # Plot training loss
    import matplotlib.pyplot as plt
    fig_path = FIG_DIR / "ann_training_loss.png"
    plt.figure()
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("=" * 80)
    print("ANN training complete")
    print(f"Saved model : {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figure : {fig_path}")
    print("RMSE by state (one-step, test):")
    for s, v in zip(state_names, rmse_by_state):
        print(f"  {s:>4s} : {v:.6g}")
    print("=" * 80)


if __name__ == "__main__":
    main()
