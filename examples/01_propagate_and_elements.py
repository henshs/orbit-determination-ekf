"""Quick demo: state <-> elements round-trip and a one-orbit propagation."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.elements import elements_to_state, state_to_elements, period
from src.propagators.propagator import ForceModel, propagate
from src.constants import RAD

a, e, i = 7000.0, 0.05, np.radians(51.6)
Om, om, nu = np.radians(30), np.radians(40), np.radians(10)
r, v = elements_to_state(a, e, i, Om, om, nu)
print("Initial state:")
print("  r =", np.round(r, 3), "km")
print("  v =", np.round(v, 5), "km/s")

el = state_to_elements(r, v)
print("Recovered elements:")
print(f"  a={el['a']:.3f} km  e={el['e']:.5f}  i={el['i']*RAD:.3f} deg  "
      f"Om={el['Omega']*RAD:.3f}  om={el['omega']*RAD:.3f}  nu={el['nu']*RAD:.3f}")

T = period(a)
force = ForceModel(use_j2=True)
t, X = propagate(np.concatenate([r, v]), (0, T), force, t_eval=np.linspace(0, T, 5))
print(f"\nOne orbital period = {T/60:.2f} min. Sampled positions |r| [km]:")
for tk, xk in zip(t, X):
    print(f"  t={tk/60:6.2f} min  |r|={np.linalg.norm(xk[:3]):.2f}")
