"""Extended Kalman Filter for orbit determination.

Predict: propagate the state through the nonlinear dynamics and the covariance
through the linearized STM.  Update: range/range-rate with Joseph-form covariance.
"""
import numpy as np
from ..propagators.propagator import ForceModel, propagate_with_stm


def process_noise(dt, q_tilde):
    """6x6 process noise from continuous white-noise acceleration.

        Q = q~ [[ dt^3/3 I , dt^2/2 I ],
                [ dt^2/2 I ,  dt   I ]]
    """
    I3 = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = (dt**3 / 3) * I3
    Q[:3, 3:] = (dt**2 / 2) * I3
    Q[3:, :3] = (dt**2 / 2) * I3
    Q[3:, 3:] = dt * I3
    return q_tilde * Q


class EKF:
    def __init__(self, force: ForceModel, R, q_tilde):
        self.force = force
        self.R = np.asarray(R, float)     # measurement noise cov
        self.q_tilde = q_tilde            # process-noise spectral density

    def predict(self, x, P, dt):
        """Propagate mean (nonlinear) and covariance (STM) over dt."""
        x_next, Phi = propagate_with_stm(x, dt, self.force)
        Q = process_noise(dt, self.q_tilde)
        P_next = Phi @ P @ Phi.T + Q
        P_next = 0.5 * (P_next + P_next.T)   # keep symmetric
        return x_next, P_next

    def update(self, x, P, z, station, t):
        """Measurement update with innovation, gain, Joseph-form covariance."""
        h = station.measure(x, t)
        H = station.jacobian(x, t)
        y = z - h                                  # innovation
        S = H @ P @ H.T + self.R                   # innovation covariance
        K = P @ H.T @ np.linalg.inv(S)             # Kalman gain
        x_new = x + K @ y
        I = np.eye(6)
        # Joseph form (numerically stable)
        P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ self.R @ K.T
        P_new = 0.5 * (P_new + P_new.T)
        nis = float(y @ np.linalg.inv(S) @ y)      # normalized innovation squared
        return x_new, P_new, y, nis
