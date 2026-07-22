"""Capstone: full EKF orbit determination.

Pipeline:
  1. Generate a truth orbit (two-body + J2) from known elements.
  2. Simulate noisy range/range-rate from a ground station, only when visible.
  3. Run the EKF from a deliberately wrong initial guess.
  4. Plot position-error convergence with 3-sigma envelope, and NIS consistency.

Run:  python3 examples/ekf_orbit_determination.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import MU_EARTH
from src.core.elements import elements_to_state, period
from src.propagators.propagator import ForceModel, propagate
from src.measurements.station import GroundStation
from src.filters.ekf import EKF
from src.filters.consistency import nis, chi2_bounds

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. TRUTH
a, e, i = 7000.0, 0.01, np.radians(51.6)
Om, om, nu = np.radians(30), np.radians(40), np.radians(0)
r0, v0 = elements_to_state(a, e, i, Om, om, nu)
x_true0 = np.concatenate([r0, v0])

truth_force = ForceModel(use_j2=True, use_drag=False)
T = period(a)
t_end = 6 * T                              # 6 orbits -> several tracking passes
dt = 30.0                                  # measurement cadence [s]
times = np.arange(0, t_end, dt)
_, X_true = propagate(x_true0, (0, t_end), truth_force, t_eval=times)

# ---------------------------------------------------------------- 2. MEASUREMENTS
# A small ground-station network greatly improves visibility & observability.
stations = [
    GroundStation(lat_deg=10.0,  lon_deg=30.0,  elevation_mask_deg=5.0),   # equatorial
    GroundStation(lat_deg=48.0,  lon_deg=-75.0, elevation_mask_deg=5.0),   # mid-north
    GroundStation(lat_deg=-33.0, lon_deg=151.0, elevation_mask_deg=5.0),   # southern
    GroundStation(lat_deg=28.0,  lon_deg=-16.0, elevation_mask_deg=5.0),   # Canary-like
]
sigma_rho, sigma_rr = 0.02, 1e-4           # 20 m, 0.1 m/s  (km, km/s)
R = np.diag([sigma_rho**2, sigma_rr**2])

# meas[k] = list of (station, z) visible at step k
meas = {}
n_meas = 0
for k, t in enumerate(times):
    hits = []
    for st in stations:
        if st.visible(X_true[k][:3], t):
            z = st.measure(X_true[k], t) + rng.multivariate_normal([0, 0], R)
            hits.append((st, z))
            n_meas += 1
    if hits:
        meas[k] = hits
print(f"{n_meas} measurements from {len(stations)} stations over {len(times)} steps "
      f"({100*len(meas)/len(times):.0f}% of steps have a pass)")

# ---------------------------------------------------------------- 3. EKF
filter_force = ForceModel(use_j2=True, use_drag=False)   # filter also models J2
q_tilde = 5e-12                            # process-noise spectral density [km^2/s^3]
ekf = EKF(filter_force, R, q_tilde)

# a-priori guess ~1 km / 1 m/s off (typical for OD refinement)
x_est = x_true0 + np.concatenate([rng.normal(0, 1, 3), rng.normal(0, 1e-3, 3)])
P = np.diag([4., 4., 4., 1e-4, 1e-4, 1e-4])   # (2 km)^2, (10 m/s)^2

pos_err, sig3, nis_hist, nis_times = [], [], [], []

x, t_prev = x_est.copy(), times[0]
for k, t in enumerate(times):
    if k > 0:
        x, P = ekf.predict(x, P, t - t_prev)
        t_prev = t
    if k in meas:
        for st, z in meas[k]:                      # sequential update per station
            x, P, y, S_nis = ekf.update(x, P, z, st, t)
            nis_hist.append(S_nis)
            nis_times.append(t / 3600.0)
    err = np.linalg.norm(x[:3] - X_true[k][:3]) * 1000.0    # m
    pos_err.append(err)
    sig3.append(3 * np.sqrt(np.trace(P[:3, :3])) * 1000.0)  # m

pos_err = np.array(pos_err)
sig3 = np.array(sig3)
hours = times / 3600.0

print(f"final position error: {pos_err[-1]:.1f} m")
print(f"mean NIS: {np.mean(nis_hist):.2f}  (expected 2.0)")

# ---------------------------------------------------------------- 4. PLOTS
os.makedirs("examples/figures", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5))

ax1.plot(hours, pos_err, color="#13315c", lw=1.4, label="position error")
ax1.plot(hours, sig3, color="#b8860b", lw=1.0, ls="--", label="3σ envelope")
# mark visible passes
vis_mask = np.array([k in meas for k in range(len(times))])
ax1.fill_between(hours, 0, pos_err.max()*1.05, where=vis_mask, color="#eef4fb",
                 step="mid", zorder=0, label="station visible")
ax1.set_yscale("log")
ax1.set_xlabel("time [hours]"); ax1.set_ylabel("position error [m]")
ax1.set_title("EKF convergence — error collapses during tracking passes")
ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.3)

lo, hi = chi2_bounds(dof=2, N=1)
ax2.plot(nis_times, nis_hist, ".", color="#13315c", ms=4, label="NIS")
ax2.axhline(2.0, color="#b8860b", lw=1.2, label="expected (m=2)")
ax2.axhline(lo, color="#888", ls=":", lw=0.8)
ax2.axhline(hi, color="#888", ls=":", lw=0.8, label="95% χ² band")
ax2.set_ylim(0, max(12, hi*1.3))
ax2.set_xlabel("time [hours]"); ax2.set_ylabel("NIS")
ax2.set_title(f"Innovation consistency — mean NIS = {np.mean(nis_hist):.2f} (target 2.0)")
ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
out = "examples/figures/ekf_orbit_determination.png"
plt.savefig(out, dpi=130)
print("saved", out)
