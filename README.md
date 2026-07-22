# Orbital Mechanics & EKF Orbit Determination

A from-scratch Python implementation of orbital dynamics, perturbations, impulsive
maneuvers, and an **Extended Kalman Filter** that recovers a spacecraft's orbit from
noisy ground-station tracking. Every module is built directly from first-principles
derivations (two-body mechanics through Gauss's variational equations and the EKF).

The centerpiece is a full orbit-determination pipeline with rigorous **filter
consistency validation** (NEES / NIS / innovation whiteness) — the piece that turns
"a filter that runs" into "a filter you can prove works."

## Highlights

- Two-body + **J2 oblateness** + **atmospheric drag** force model
- State ↔ Keplerian element conversion, Kepler-equation solver, frame rotations
- Impulsive **Hohmann / bi-elliptic / plane-change** transfers and low-thrust spiral
- Ground-station **range / range-rate** measurements with analytic Jacobians
- **EKF** with variational state-transition matrix and **Joseph-form** covariance update
- **NEES / NIS / whiteness** consistency diagnostics with χ² confidence bounds
- Unit tests including finite-difference checks of every Jacobian

## Layout

```
src/
  constants.py                 physical constants (mu, Re, J2, omega_earth, ...)
  core/
    two_body.py                EOM  r'' = -mu r/r^3  and gravity-gradient tensor G
    kepler.py                  Kepler's equation + anomaly conversions
    elements.py                state <-> (a,e,i,Omega,omega,nu)
    frames.py                  perifocal <-> ECI rotations
  perturbations/
    j2.py                      J2 acceleration vector
    drag.py                    exponential-atmosphere drag
  propagators/
    propagator.py              DOP853 integrator + STM (Phi_dot = A Phi)
  maneuvers/
    transfers.py               Hohmann, bi-elliptic, plane change, low-thrust
  measurements/
    station.py                 ground station, range/range-rate, Jacobians
  filters/
    ekf.py                     EKF predict/update, process noise
    consistency.py             NEES, NIS, autocorrelation, chi2 bounds
examples/
  01_propagate_and_elements.py
  02_maneuvers.py
  03_j2_raan_drift.py          numerical drift vs. analytical formula
  ekf_orbit_determination.py   CAPSTONE: truth -> measurements -> EKF -> validation
tests/
  test_all.py
```

## Quick start

```bash
pip install -r requirements.txt

python examples/ekf_orbit_determination.py     # runs the full OD demo + plots
python tests/test_all.py                        # or:  pytest -q
```

## What the capstone shows

`examples/ekf_orbit_determination.py` generates a truth orbit (two-body + J2),
simulates noisy range/range-rate from a 4-station network (only when each station
sees the satellite above its elevation mask), and runs the EKF from a ~1 km a-priori
error. Output (`examples/figures/`):

- **Convergence plot** — position error collapses from tens of km to ~100 m during
  tracking passes; the 3σ envelope grows between passes and shrinks during them.
- **NIS plot** — normalized innovation squared scatters around its expected value
  (m = 2) inside the 95% χ² band, confirming the filter is statistically consistent.

Typical run: ~130 measurements over 6 orbits, final position error ~100 m,
mean NIS ≈ 1.7–2.0.

## Validation

`tests/test_all.py` checks:
- Kepler equation round-trip (M → E → M)
- state ↔ elements round-trip (exact to 1e-9)
- gravity-gradient tensor G vs. finite differences
- measurement Jacobian H vs. finite differences
- energy conservation under two-body propagation
- J2 RAAN-drift sign (westward for prograde orbits)
- Hohmann LEO→GEO Δv against the textbook value

## Units

km, seconds, radians throughout. Gravitational parameters in km³/s².

## Acknowledgments

The derivations, code, and validation in this project were developed by the author
in collaboration with **Claude** (Anthropic), used as a technical sounding board for
working through the underlying mathematics and structuring the implementation. All
results were independently checked (unit tests, finite-difference Jacobian
verification, and numerical-vs-analytical cross-checks).
