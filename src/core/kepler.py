"""Kepler's equation and anomaly conversions."""
import numpy as np


def solve_kepler(M, e, tol=1e-12, maxit=50):
    """Solve M = E - e sin E for the eccentric anomaly E via Newton's method."""
    M = np.mod(M, 2 * np.pi)
    E = M if e < 0.8 else np.pi          # good starting guess
    for _ in range(maxit):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def true_from_eccentric(E, e):
    """True anomaly from eccentric anomaly."""
    return 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                            np.sqrt(1 - e) * np.cos(E / 2))


def eccentric_from_true(nu, e):
    """Eccentric anomaly from true anomaly."""
    return 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                            np.sqrt(1 + e) * np.cos(nu / 2))


def mean_from_eccentric(E, e):
    """Mean anomaly from eccentric anomaly (Kepler's equation)."""
    return E - e * np.sin(E)


def true_from_mean(M, e):
    """Convenience: M -> E -> nu."""
    return true_from_eccentric(solve_kepler(M, e), e)
