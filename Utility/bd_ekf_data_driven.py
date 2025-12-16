"""
Utility/bd_ekf_data_driven.py

EKF using a learned DISCRETE dynamics model:
  x_{k+1} = f_dd(x_k, u_k)
Jacobian A_k = ∂f_dd/∂x estimated by finite differences (batch-evaluated).
Measurement model is the same as BdEKF (BNW, BN, ...), with constant H.
"""

from __future__ import annotations
import numpy as np


class BdEKFDataDriven:
    def __init__(self, f_dd_model, Q: np.ndarray, R: np.ndarray, measurement_option: str = "h_BNW"):
        self.model = f_dd_model  # must expose predict_xnext_batch(X,U)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.meas_opt = measurement_option

    def _linearize_f_fd(self, x: np.ndarray, u: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Finite-difference linearization of the DISCRETE map x_next = f(x,u).
        Returns A = ∂f/∂x (6x6).
        Uses one batched forward pass of size (1+6).
        """
        x = np.asarray(x, dtype=float).reshape(6,)
        u = np.asarray(u, dtype=float).reshape(1,)

        X0 = x.reshape(1, 6)
        U0 = u.reshape(1, 1)

        # Build batch: base + 6 perturbed states
        Xb = np.repeat(X0, repeats=7, axis=0)
        Ub = np.repeat(U0, repeats=7, axis=0)

        for j in range(6):
            Xb[j + 1, j] += eps

        Xnext_b = self.model.predict_xnext_batch(Xb, Ub)  # (7,6)
        xnext0 = Xnext_b[0, :]
        A = np.zeros((6, 6), dtype=float)
        for j in range(6):
            A[:, j] = (Xnext_b[j + 1, :] - xnext0) / eps

        return A

    def predict(self, x_hat: np.ndarray, P: np.ndarray, u: np.ndarray, eps_fd: float = 1e-4):
        """
        x_pred = f_dd(x_hat, u)
        P_pred = A P A^T + Q
        """
        x_hat = np.asarray(x_hat, dtype=float).reshape(6,)
        u = np.asarray(u, dtype=float).reshape(1,)

        x_pred = self.model.predict_xnext_batch(x_hat.reshape(1, 6), u.reshape(1, 1)).reshape(6,)
        A = self._linearize_f_fd(x_hat, u, eps=eps_fd)

        P = np.asarray(P, dtype=float)
        P_pred = A @ P @ A.T + self.Q
        return x_pred, P_pred

    def _measurement_and_jacobian(self, x: np.ndarray):
        x = np.asarray(x, dtype=float).reshape(6,)
        Z, S1, S2, S3, N, W = x
        B = Z + S1 + S2 + S3

        if self.meas_opt == "h_BNW":
            y = np.array([B, N, W], dtype=float)
            H = np.array([[1,1,1,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1]], dtype=float)
        elif self.meas_opt == "h_BN":
            y = np.array([B, N], dtype=float)
            H = np.array([[1,1,1,1,0,0],
                          [0,0,0,0,1,0]], dtype=float)
        elif self.meas_opt == "h_B":
            y = np.array([B], dtype=float)
            H = np.array([[1,1,1,1,0,0]], dtype=float)
        elif self.meas_opt == "h_NW":
            y = np.array([N, W], dtype=float)
            H = np.array([[0,0,0,0,1,0],
                          [0,0,0,0,0,1]], dtype=float)
        elif self.meas_opt == "h_N":
            y = np.array([N], dtype=float)
            H = np.array([[0,0,0,0,1,0]], dtype=float)
        elif self.meas_opt == "h_W":
            y = np.array([W], dtype=float)
            H = np.array([[0,0,0,0,0,1]], dtype=float)
        else:
            raise ValueError(f"Unknown measurement_option: {self.meas_opt}")

        return y, H

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, y_meas: np.ndarray):
        x_pred = np.asarray(x_pred, dtype=float).reshape(6,)
        P_pred = np.asarray(P_pred, dtype=float)
        y_meas = np.asarray(y_meas, dtype=float).reshape(-1)

        y_pred, H = self._measurement_and_jacobian(x_pred)

        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innovation = y_meas - y_pred
        x_upd = x_pred + K @ innovation
        P_upd = (np.eye(6) - K @ H) @ P_pred
        return x_upd, P_upd
