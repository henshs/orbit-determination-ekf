"""Physical constants and Earth parameters (SI-ish: km, s, rad)."""
import numpy as np

# Gravitational parameters [km^3 / s^2]
MU_EARTH = 398600.4418
MU_SUN   = 1.32712440018e11
MU_MOON  = 4902.800066

# Earth shape / rotation
R_EARTH   = 6378.137          # equatorial radius [km]
J2        = 1.08262668e-3     # second zonal harmonic [-]
OMEGA_EARTH = 7.2921159e-5    # sidereal rotation rate [rad/s]

# Misc
G0 = 9.80665e-3               # standard gravity [km/s^2] (for Isp in seconds)
DEG = np.pi / 180.0
RAD = 180.0 / np.pi
