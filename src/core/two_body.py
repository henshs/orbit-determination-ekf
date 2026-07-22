"""Two-body dynamics: the equation of motion and the gravity-gradient Jacobian.

EOM and gravity gradient tensor G.
"""
import numpy as np
from ..constants import MU_EARTH


def two_body_accel(r, mu=MU_EARTH):
    """Central-gravity acceleration:  a = -mu r / |r|^3."""
    r = np.asarray(r, float)
    rn = np.linalg.norm(r)
    return -mu * r / rn**3


def gravity_gradient(r, mu=MU_EARTH):
    """3x3 gravity-gradient tensor  G = d(accel)/dr.

        G_ij = mu/r^5 (3 r_i r_j - r^2 delta_ij)
    """
    r = np.asarray(r, float)
    rn = np.linalg.norm(r)
    I3 = np.eye(3)
    return (mu / rn**5) * (3.0 * np.outer(r, r) - rn**2 * I3)


def dynamics_jacobian(r, mu=MU_EARTH):
    """6x6 continuous dynamics Jacobian A = [[0, I],[G, 0]]."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = gravity_gradient(r, mu)
    return A
