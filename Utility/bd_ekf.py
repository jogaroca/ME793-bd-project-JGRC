"""
Extended Kalman Filter (EKF) for the Bd chemostat model.

- Uses the continuous dynamics f(x,u) from bd_chemostat.F
- Discretizes with Euler: x_{k+1} = x_k + dt * f(x_k, u_k)
- Linearizes f numerically (d f/dx) to obtain the matrix A_k
- Implements different measurement functions (BNW, BN, B, NW, N, W)
  and their Jacobians H(x) (here all are linear in x, so H is constant).
"""

from __future__ import annotations

import numpy as np

import bd_chemostat as bd


class BdEKF:
    def __init__(self, Q: np.ndarray, R: np.ndarray, measurement_option: str = "h_BNW"):
        """
        Parameters
        ----------
        Q : (6,6) ndarray
            Process noise covariance (discrete).
        R : (ny,ny) ndarray
            Measurement noise covariance.
        measurement_option : str
            One of: 'h_BNW', 'h_BN', 'h_B', 'h_NW', 'h_N', 'h_W'.
        """
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.meas_opt = measurement_option

        self.f_obj = bd.F()
        self.f = self.f_obj.f

    # ------------------------------------------------------------------
    # Discrete-time prediction: x_{k+1} = x_k + dt * f(x_k, u_k)
    # ------------------------------------------------------------------
    def _linearize_f_fd(self, x: np.ndarray, u: np.ndarray, dt: float,
                        eps: float = 1e-6) -> np.ndarray:
        """
        Finite-difference linearization of f around (x,u).

        Returns the discrete-time Jacobian:
            A_d = I + dt * (∂f/∂x).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        n = x.size

        f0 = self.f(x, u)
        f0 = np.asarray(f0, dtype=float).reshape(-1)

        A = np.zeros((n, n), dtype=float)

        for j in range(n):
            dx = np.zeros(n, dtype=float)
            dx[j] = eps
            f_pert = self.f(x + dx, u)
            f_pert = np.asarray(f_pert, dtype=float).reshape(-1)
            A[:, j] = (f_pert - f0) / eps

        A_d = np.eye(n) + dt * A
        return A_d

    def predict(self, x_hat: np.ndarray, P: np.ndarray,
                u: np.ndarray, dt: float):
        """
        EKF prediction step.

        x_hat^- = x_hat + dt f(x_hat, u)
        P^-     = A_d P A_d^T + Q
        """
        x_hat = np.asarray(x_hat, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)

        # Nonlinear propagation
        f_val = self.f(x_hat, u)
        f_val = np.asarray(f_val, dtype=float).reshape(-1)
        x_pred = x_hat + dt * f_val

        # Linearization
        A_d = self._linearize_f_fd(x_hat, u, dt)

        P = np.asarray(P, dtype=float)
        P_pred = A_d @ P @ A_d.T + self.Q

        return x_pred, P_pred

    # ------------------------------------------------------------------
    # Measurement model y = h(x), H = ∂h/∂x
    # ------------------------------------------------------------------
    def _measurement_and_jacobian(self, x: np.ndarray):
        """
        Returns (y, H) for the current measurement_option.

        x = [Z, S1, S2, S3, N, W]
        B = Z + S1 + S2 + S3
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        Z, S1, S2, S3, N, W = x
        B = Z + S1 + S2 + S3

        if self.meas_opt == "h_BNW":
            y = np.array([B, N, W], dtype=float)
            H = np.array([
                [1, 1, 1, 1, 0, 0],  # dB/dx
                [0, 0, 0, 0, 1, 0],  # dN/dx
                [0, 0, 0, 0, 0, 1],  # dW/dx
            ], dtype=float)

        elif self.meas_opt == "h_BN":
            y = np.array([B, N], dtype=float)
            H = np.array([
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ], dtype=float)

        elif self.meas_opt == "h_B":
            y = np.array([B], dtype=float)
            H = np.array([[1, 1, 1, 1, 0, 0]], dtype=float)

        elif self.meas_opt == "h_NW":
            y = np.array([N, W], dtype=float)
            H = np.array([
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ], dtype=float)

        elif self.meas_opt == "h_N":
            y = np.array([N], dtype=float)
            H = np.array([[0, 0, 0, 0, 1, 0]], dtype=float)

        elif self.meas_opt == "h_W":
            y = np.array([W], dtype=float)
            H = np.array([[0, 0, 0, 0, 0, 1]], dtype=float)

        else:
            raise ValueError(f"Unknown measurement_option: {self.meas_opt}")

        return y, H

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray,
               y_meas: np.ndarray):
        """
        EKF update step.

        K = P^- H^T (H P^- H^T + R)^{-1}
        x^+ = x^- + K (y_meas - y_pred)
        P^+ = (I - K H) P^-
        """
        x_pred = np.asarray(x_pred, dtype=float).reshape(-1)
        P_pred = np.asarray(P_pred, dtype=float)
        y_meas = np.asarray(y_meas, dtype=float).reshape(-1)

        y_pred, H = self._measurement_and_jacobian(x_pred)

        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innovation = y_meas - y_pred
        x_upd = x_pred + K @ innovation

        I = np.eye(x_pred.size)
        P_upd = (I - K @ H) @ P_pred

        return x_upd, P_upd
