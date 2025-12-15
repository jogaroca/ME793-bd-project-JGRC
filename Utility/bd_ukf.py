"""
Unscented Kalman Filter (UKF) for the Bd chemostat model.

State:
    x = [Z, S1, S2, S3, N, W]^T

Dynamics (discrete Euler):
    x_{k+1} = x_k + dt * f(x_k, u_k)

Measurements (same options as EKF):
    B = Z + S1 + S2 + S3

    h_BNW: y = [B, N, W]
    h_BN : y = [B, N]
    h_B  : y = [B]
    h_NW : y = [N, W]
    h_N  : y = [N]
    h_W  : y = [W]
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from Utility import bd_chemostat as bd


@dataclass
class UKFParams:
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0


class BdUKF:
    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        measurement_option: str = "h_BNW",
        ukf_params: UKFParams | None = None,
    ):
        """
        Parameters
        ----------
        Q : (6,6) ndarray
            Discrete process noise covariance.
        R : (ny,ny) ndarray
            Measurement noise covariance.
        measurement_option : str
            One of: 'h_BNW', 'h_BN', 'h_B', 'h_NW', 'h_N', 'h_W'.
        ukf_params : UKFParams
            Unscented transform parameters (alpha, beta, kappa).
        """
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.meas_opt = str(measurement_option)

        self.nx = 6
        self.params = ukf_params if ukf_params is not None else UKFParams()

        # UKF scaling
        self._lambda = (self.params.alpha ** 2) * (self.nx + self.params.kappa) - self.nx
        self._c = self.nx + self._lambda
        self._gamma = np.sqrt(self._c)

        # Weights
        self.Wm = np.full(2 * self.nx + 1, 1.0 / (2.0 * self._c), dtype=float)
        self.Wc = np.full(2 * self.nx + 1, 1.0 / (2.0 * self._c), dtype=float)
        self.Wm[0] = self._lambda / self._c
        self.Wc[0] = self.Wm[0] + (1.0 - self.params.alpha ** 2 + self.params.beta)

        # Dynamics
        self.f_obj = bd.F()
        self.f = self.f_obj.f

    # ------------------------------------------------------------------
    # Measurement function (consistent with BdEKF)
    # ------------------------------------------------------------------
    def _measurement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        Z, S1, S2, S3, N, W = x
        B = Z + S1 + S2 + S3

        if self.meas_opt == "h_BNW":
            return np.array([B, N, W], dtype=float)
        if self.meas_opt == "h_BN":
            return np.array([B, N], dtype=float)
        if self.meas_opt == "h_B":
            return np.array([B], dtype=float)
        if self.meas_opt == "h_NW":
            return np.array([N, W], dtype=float)
        if self.meas_opt == "h_N":
            return np.array([N], dtype=float)
        if self.meas_opt == "h_W":
            return np.array([W], dtype=float)

        raise ValueError(f"Unknown measurement_option: {self.meas_opt}")

    # ------------------------------------------------------------------
    # Sigma points
    # ------------------------------------------------------------------
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Returns sigma points X of shape (2*nx+1, nx).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float)

        # Symmetrize
        P = 0.5 * (P + P.T)

        # Robust Cholesky with jitter
        jitter = 0.0
        for k in range(10):
            try:
                S = np.linalg.cholesky(P + jitter * np.eye(self.nx))
                break
            except np.linalg.LinAlgError:
                jitter = 1e-12 if k == 0 else jitter * 10.0
        else:
            raise np.linalg.LinAlgError("Cholesky failed in UKF sigma point generation (P not PD).")

        X = np.zeros((2 * self.nx + 1, self.nx), dtype=float)
        X[0, :] = x

        for i in range(self.nx):
            col = self._gamma * S[:, i]
            X[1 + i, :] = x + col
            X[1 + self.nx + i, :] = x - col

        return X

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, x_hat: np.ndarray, P: np.ndarray, u: np.ndarray, dt: float):
        """
        UKF prediction step using Euler discretization.
        """
        x_hat = np.asarray(x_hat, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)

        X = self._sigma_points(x_hat, P)

        # Propagate sigma points through dynamics
        Xp = np.zeros_like(X)
        for i in range(X.shape[0]):
            f_val = self.f(X[i, :], u)
            f_val = np.asarray(f_val, dtype=float).reshape(-1)
            Xp[i, :] = X[i, :] + dt * f_val

        # Predicted mean
        x_pred = np.sum(self.Wm[:, None] * Xp, axis=0)

        # Predicted covariance
        P_pred = np.zeros((self.nx, self.nx), dtype=float)
        for i in range(Xp.shape[0]):
            dx = (Xp[i, :] - x_pred).reshape(-1, 1)
            P_pred += self.Wc[i] * (dx @ dx.T)

        P_pred += self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        return x_pred, P_pred

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, y_meas: np.ndarray):
        """
        UKF update step.
        """
        x_pred = np.asarray(x_pred, dtype=float).reshape(-1)
        P_pred = np.asarray(P_pred, dtype=float)
        y_meas = np.asarray(y_meas, dtype=float).reshape(-1)

        # Sigma points from predicted state
        X = self._sigma_points(x_pred, P_pred)

        # Transform to measurement space
        ny = y_meas.size
        Y = np.zeros((X.shape[0], ny), dtype=float)
        for i in range(X.shape[0]):
            Y[i, :] = self._measurement(X[i, :])

        # Predicted measurement mean
        y_pred = np.sum(self.Wm[:, None] * Y, axis=0)

        # Innovation covariance S and cross covariance Pxy
        S = np.zeros((ny, ny), dtype=float)
        Pxy = np.zeros((self.nx, ny), dtype=float)

        for i in range(X.shape[0]):
            dy = (Y[i, :] - y_pred).reshape(-1, 1)
            dx = (X[i, :] - x_pred).reshape(-1, 1)
            S += self.Wc[i] * (dy @ dy.T)
            Pxy += self.Wc[i] * (dx @ dy.T)

        S += self.R
        S = 0.5 * (S + S.T)

        # Kalman gain
        K = Pxy @ np.linalg.inv(S)

        # Update
        innovation = (y_meas - y_pred).reshape(-1, 1)
        x_upd = x_pred + (K @ innovation).reshape(-1)

        P_upd = P_pred - K @ S @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        return x_upd, P_upd
