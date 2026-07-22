"""Conversions between Cartesian state and Keplerian elements.

Elements tuple/dict: (a, e, i, Omega, omega, nu)  with angles in radians.
"""
import numpy as np
from ..constants import MU_EARTH
from .kepler import solve_kepler, true_from_eccentric, eccentric_from_true
from .frames import perifocal_to_eci


def elements_to_state(a, e, i, Omega, omega, nu, mu=MU_EARTH):
    """Keplerian elements -> (r, v) in ECI."""
    p = a * (1 - e**2)
    r_pf = np.array([p * np.cos(nu) / (1 + e * np.cos(nu)),
                     p * np.sin(nu) / (1 + e * np.cos(nu)),
                     0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    R = perifocal_to_eci(Omega, i, omega)
    return R @ r_pf, R @ v_pf


def state_to_elements(r, v, mu=MU_EARTH):
    """(r, v) -> Keplerian elements  (extraction, with quadrant checks)."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)

    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0, 0, 1], h)          # node vector
    nn = np.linalg.norm(n)

    # eccentricity vector
    e_vec = (np.cross(v, h) / mu) - r / rn
    e = np.linalg.norm(e_vec)

    energy = vn**2 / 2 - mu / rn
    a = -mu / (2 * energy)

    i = np.arccos(np.clip(h[2] / hn, -1, 1))

    # RAAN
    if nn > 1e-12:
        Omega = np.arccos(np.clip(n[0] / nn, -1, 1))
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0

    # argument of perigee
    if nn > 1e-12 and e > 1e-12:
        omega = np.arccos(np.clip(np.dot(n, e_vec) / (nn * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0

    # true anomaly
    if e > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * rn), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = np.arccos(np.clip(np.dot(n, r) / (nn * rn), -1, 1)) if nn > 1e-12 else 0.0

    return dict(a=a, e=e, i=i, Omega=Omega, omega=omega, nu=nu)


def period(a, mu=MU_EARTH):
    return 2 * np.pi * np.sqrt(a**3 / mu)


def mean_motion(a, mu=MU_EARTH):
    return np.sqrt(mu / a**3)
