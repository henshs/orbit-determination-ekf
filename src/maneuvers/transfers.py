"""Impulsive transfers and plane changes."""
import numpy as np
from ..constants import MU_EARTH


def hohmann(r1, r2, mu=MU_EARTH):
    """Two-burn Hohmann transfer between coplanar circular orbits.

    Returns dict(dv1, dv2, dv_total, t_transfer).
    """
    at = 0.5 * (r1 + r2)
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    vp = np.sqrt(mu * (2 / r1 - 1 / at))   # transfer speed at r1
    va = np.sqrt(mu * (2 / r2 - 1 / at))   # transfer speed at r2
    dv1 = abs(vp - v1)
    dv2 = abs(v2 - va)
    t = np.pi * np.sqrt(at**3 / mu)
    return dict(dv1=dv1, dv2=dv2, dv_total=dv1 + dv2, t_transfer=t, a_transfer=at)


def bielliptic(r1, r2, rb, mu=MU_EARTH):
    """Three-burn bi-elliptic transfer via intermediate radius rb."""
    a1 = 0.5 * (r1 + rb)
    a2 = 0.5 * (r2 + rb)
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    # burn 1: r1 circular -> ellipse1 periapsis
    dv1 = abs(np.sqrt(mu * (2 / r1 - 1 / a1)) - v1)
    # burn 2: at rb, ellipse1 apoapsis -> ellipse2 apoapsis
    dv2 = abs(np.sqrt(mu * (2 / rb - 1 / a2)) - np.sqrt(mu * (2 / rb - 1 / a1)))
    # burn 3: at r2, ellipse2 periapsis -> circular
    dv3 = abs(v2 - np.sqrt(mu * (2 / r2 - 1 / a2)))
    t = np.pi * (np.sqrt(a1**3 / mu) + np.sqrt(a2**3 / mu))
    return dict(dv1=dv1, dv2=dv2, dv3=dv3, dv_total=dv1 + dv2 + dv3, t_transfer=t)


def plane_change(v, di_rad):
    """Pure plane-change dv:  dv = 2 v sin(di/2)."""
    return 2 * v * np.sin(di_rad / 2)


def combined_maneuver(vi, vf, di_rad):
    """Combined speed + plane change (law of cosines)."""
    return np.sqrt(vi**2 + vf**2 - 2 * vi * vf * np.cos(di_rad))


def low_thrust_spiral(r1, r2, mu=MU_EARTH):
    """Edelbaum in-plane spiral dv = |v_c1 - v_c2|."""
    return abs(np.sqrt(mu / r1) - np.sqrt(mu / r2))
