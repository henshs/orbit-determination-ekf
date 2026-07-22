"""Atmospheric drag with an exponential density model."""
import numpy as np
from ..constants import R_EARTH, OMEGA_EARTH

# Simple piecewise-exponential atmosphere: (base alt km, density kg/m^3, scale height km)
_ATM = [
    (0,   1.225,     8.44),
    (200, 2.5e-10,   45.0),
    (300, 1.9e-11,   52.5),
    (400, 2.1e-12,   58.5),
    (500, 5.0e-13,   63.8),
    (600, 1.6e-13,   71.0),
    (700, 5.0e-14,   79.0),
    (800, 5.0e-15,   88.0),
]


def density(alt_km):
    """Exponential atmospheric density [kg/m^3] at altitude alt_km."""
    base = _ATM[0]
    for entry in _ATM:
        if alt_km >= entry[0]:
            base = entry
        else:
            break
    h0, rho0, H = base
    return rho0 * np.exp(-(alt_km - h0) / H)


def drag_accel(r, v, beta, Re=R_EARTH, omega_e=OMEGA_EARTH):
    """Drag acceleration [km/s^2] opposing velocity relative to rotating atmosphere.

    beta = m / (Cd A)  ballistic coefficient [kg/m^2].
    Uses velocity relative to the co-rotating atmosphere.
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    alt = np.linalg.norm(r) - Re
    rho = density(alt)                       # kg/m^3
    # atmosphere co-rotates with Earth
    v_atm = np.cross([0, 0, omega_e], r)     # km/s
    v_rel = v - v_atm
    vrel_n = np.linalg.norm(v_rel)
    # a = -0.5 rho vrel^2 / beta * vhat ; convert: rho[kg/m^3], vrel[km/s]->m/s
    vrel_ms = vrel_n * 1000.0
    a_ms2 = -0.5 * rho * vrel_ms**2 / beta   # m/s^2 magnitude along -vhat
    a_kms2 = a_ms2 / 1000.0
    if vrel_n == 0:
        return np.zeros(3)
    return a_kms2 * (v_rel / vrel_n)
