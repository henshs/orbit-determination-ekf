"""Demo: J2 secular RAAN drift — numerical propagation vs. analytical formula (Topic 16)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import MU_EARTH, R_EARTH, J2, RAD
from src.core.elements import elements_to_state, state_to_elements, period, mean_motion
from src.propagators.propagator import ForceModel, propagate

a, e, i = 7000.0, 0.001, np.radians(55.0)
r, v = elements_to_state(a, e, i, np.radians(20), np.radians(0), np.radians(0))

# analytical secular RAAN drift [rad/s] -> deg/day
n = mean_motion(a)
dOm_dt = -1.5 * n * J2 * (R_EARTH / (a * (1 - e**2)))**2 * np.cos(i)
dOm_analytic = dOm_dt * RAD * 86400.0

# numerical: propagate 10 orbits, measure RAAN change
force = ForceModel(use_j2=True)
T = period(a)
t, X = propagate(np.concatenate([r, v]), (0, 10 * T), force,
                 t_eval=np.linspace(0, 10 * T, 400))
Om0 = state_to_elements(X[0][:3], X[0][3:])['Omega']
Om1 = state_to_elements(X[-1][:3], X[-1][3:])['Omega']
dOm = (Om1 - Om0 + np.pi) % (2 * np.pi) - np.pi
dOm_numeric = dOm * RAD / (10 * T / 86400.0)

print(f"J2 RAAN drift for a={a} km, i=55 deg:")
print(f"  analytical : {dOm_analytic:+.3f} deg/day")
print(f"  numerical  : {dOm_numeric:+.3f} deg/day")
print(f"  difference : {abs(dOm_analytic - dOm_numeric):.4f} deg/day")
