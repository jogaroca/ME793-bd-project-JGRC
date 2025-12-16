"""
Utility/bd_dynamics_ann.py

Loads the trained ANN discrete dynamics model and its scalers, and provides:
  x_{k+1} = f_dd(x_k, u_k)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras


class BdDynamicsANN:
    def __init__(self, model_path: Path, scaler_path: Path):
        self.model = keras.models.load_model(model_path)

        sc = np.load(scaler_path, allow_pickle=True)
        self.in_mean = sc["in_mean"].astype(np.float32)
        self.in_std  = sc["in_std"].astype(np.float32)
        self.out_mean = sc["out_mean"].astype(np.float32)
        self.out_std  = sc["out_std"].astype(np.float32)

    def _standardize_in(self, XU: np.ndarray) -> np.ndarray:
        return (XU - self.in_mean) / self.in_std

    def _unstandardize_out(self, Xn: np.ndarray) -> np.ndarray:
        return Xn * self.out_std + self.out_mean

    def predict_xnext_batch(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        X: (B,6), U: (B,1) -> Xnext: (B,6)
        """
        X = np.asarray(X, dtype=np.float32)
        U = np.asarray(U, dtype=np.float32)
        XU = np.concatenate([X, U], axis=1)  # (B,7)

        XU_z = self._standardize_in(XU)
        Xnext_z = self.model(XU_z, training=False).numpy()
        Xnext = self._unstandardize_out(Xnext_z)
        return np.asarray(Xnext, dtype=np.float64)

    def f_dd(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        x: (6,), u: (1,) -> x_next: (6,)
        """
        x = np.asarray(x, dtype=np.float32).reshape(1, 6)
        u = np.asarray(u, dtype=np.float32).reshape(1, 1)
        return self.predict_xnext_batch(x, u).reshape(6,)
