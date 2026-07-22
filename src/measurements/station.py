"""Ground station model, range/range-rate measurements and Jacobians."""
import numpy as np
from ..constants import R_EARTH, OMEGA_EARTH


class GroundStation:
    def __init__(self, lat_deg, lon_deg, alt_km=0.0, theta_g0=0.0,
                 elevation_mask_deg=10.0):
        self.lat = np.radians(lat_deg)
        self.lon = np.radians(lon_deg)
        self.alt = alt_km
        self.theta_g0 = theta_g0
        self.mask = np.radians(elevation_mask_deg)

    def sidereal(self, t):
        """Local sidereal time."""
        return self.theta_g0 + OMEGA_EARTH * t + self.lon

    def state_eci(self, t):
        """Station position & velocity in ECI at time t."""
        theta = self.sidereal(t)
        R = R_EARTH + self.alt
        cphi = np.cos(self.lat)
        r = R * np.array([cphi * np.cos(theta),
                          cphi * np.sin(theta),
                          np.sin(self.lat)])
        v = np.cross([0, 0, OMEGA_EARTH], r)
        return r, v

    def enu_basis(self, t):
        """East, North, Up unit vectors in ECI."""
        theta = self.sidereal(t)
        E = np.array([-np.sin(theta), np.cos(theta), 0.0])
        U = np.array([np.cos(self.lat) * np.cos(theta),
                      np.cos(self.lat) * np.sin(theta),
                      np.sin(self.lat)])
        N = np.cross(U, E)
        return E, N, U

    def elevation(self, r_sc, t):
        r_stn, _ = self.state_eci(t)
        rho = r_sc - r_stn
        _, _, U = self.enu_basis(t)
        return np.arcsin(np.clip(np.dot(rho, U) / np.linalg.norm(rho), -1, 1))

    def visible(self, r_sc, t):
        return self.elevation(r_sc, t) > self.mask

    # ---------------- measurement function h(x) ----------------
    def measure(self, x, t):
        """Return (range, range_rate) for state x=[r,v] at time t."""
        r_stn, v_stn = self.state_eci(t)
        rho = x[:3] - r_stn
        rho_dot = x[3:6] - v_stn
        rng = np.linalg.norm(rho)
        rng_rate = np.dot(rho, rho_dot) / rng
        return np.array([rng, rng_rate])

    def jacobian(self, x, t):
        """2x6 measurement Jacobian H = d[range, range_rate]/dx."""
        r_stn, v_stn = self.state_eci(t)
        rho = x[:3] - r_stn
        rho_dot = x[3:6] - v_stn
        rng = np.linalg.norm(rho)
        rhat = rho / rng
        rng_rate = np.dot(rhat, rho_dot)

        H = np.zeros((2, 6))
        # range row
        H[0, :3] = rhat
        # range-rate row
        H[1, :3] = (rho_dot - rng_rate * rhat) / rng
        H[1, 3:6] = rhat
        return H
