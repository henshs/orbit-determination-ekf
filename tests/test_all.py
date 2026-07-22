"""Unit tests. Run with:  pytest -q   (from the repo root)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import MU_EARTH, R_EARTH
from src.core.kepler import solve_kepler, mean_from_eccentric, true_from_mean
from src.core.elements import (elements_to_state, state_to_elements,
                               period, mean_motion)
from src.core.two_body import gravity_gradient, two_body_accel
from src.perturbations.j2 import j2_accel
from src.propagators.propagator import ForceModel, propagate
from src.maneuvers.transfers import hohmann
from src.measurements.station import GroundStation


def test_kepler_roundtrip():
    for e in [0.0, 0.1, 0.5, 0.9]:
        for M in np.linspace(0, 2 * np.pi, 20):
            E = solve_kepler(M, e)
            assert abs(mean_from_eccentric(E, e) - np.mod(M, 2*np.pi)) < 1e-10


def test_elements_state_roundtrip():
    a, e, i = 7000.0, 0.05, np.radians(51.6)
    Om, om, nu = np.radians(40), np.radians(60), np.radians(120)
    r, v = elements_to_state(a, e, i, Om, om, nu)
    el = state_to_elements(r, v)
    assert abs(el['a'] - a) < 1e-6
    assert abs(el['e'] - e) < 1e-9
    assert abs(el['i'] - i) < 1e-9
    assert abs(el['Omega'] - Om) < 1e-9
    assert abs(el['omega'] - om) < 1e-9
    assert abs(el['nu'] - nu) < 1e-9


def test_gravity_gradient_matches_finite_difference():
    r = np.array([7000.0, 1200.0, -800.0])
    G = gravity_gradient(r)
    eps = 1e-3
    Gnum = np.zeros((3, 3))
    for j in range(3):
        dr = np.zeros(3); dr[j] = eps
        Gnum[:, j] = (two_body_accel(r + dr) - two_body_accel(r - dr)) / (2 * eps)
    assert np.max(np.abs(G - Gnum)) < 1e-6


def test_hohmann_leo_to_geo():
    res = hohmann(6678.0, 42164.0)
    # Known textbook value ~3.9 km/s
    assert abs(res['dv_total'] - 3.90) < 0.1


def test_energy_conserved_two_body():
    a = 7500.0
    r, v = elements_to_state(a, 0.1, np.radians(30), 0, 0, 0)
    force = ForceModel(use_j2=False, use_drag=False)
    x0 = np.concatenate([r, v])
    T = period(a)
    t, X = propagate(x0, (0, T), force, t_eval=np.linspace(0, T, 50))
    energies = [np.linalg.norm(x[3:])**2/2 - MU_EARTH/np.linalg.norm(x[:3]) for x in X]
    assert (max(energies) - min(energies)) < 1e-6


def test_j2_raan_drift_sign():
    """Prograde orbit (i<90) should give negative (westward) RAAN drift."""
    a, e, i = 7000.0, 0.001, np.radians(55)
    r, v = elements_to_state(a, e, i, np.radians(10), np.radians(0), np.radians(0))
    force = ForceModel(use_j2=True, use_drag=False)
    x0 = np.concatenate([r, v])
    T = period(a)
    t, X = propagate(x0, (0, 5*T), force, t_eval=np.linspace(0, 5*T, 200))
    Om0 = state_to_elements(X[0][:3], X[0][3:])['Omega']
    Om1 = state_to_elements(X[-1][:3], X[-1][3:])['Omega']
    # unwrap
    dOm = (Om1 - Om0 + np.pi) % (2*np.pi) - np.pi
    assert dOm < 0     # westward


def test_measurement_jacobian_finite_difference():
    stn = GroundStation(40.0, -75.0)
    x = np.array([7000.0, 500.0, 1000.0, 1.0, 7.0, 2.0])
    t = 1234.0
    H = stn.jacobian(x, t)
    eps = 1e-4
    Hnum = np.zeros((2, 6))
    for j in range(6):
        dx = np.zeros(6); dx[j] = eps
        Hnum[:, j] = (stn.measure(x+dx, t) - stn.measure(x-dx, t)) / (2*eps)
    assert np.max(np.abs(H - Hnum)) < 1e-5


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for tfn in tests:
        try:
            tfn(); print(f"PASS  {tfn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL  {tfn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
