"""Filter consistency diagnostics: NEES, NIS, whiteness."""
import numpy as np
from scipy.stats import chi2


def nees(x_true, x_est, P):
    """Normalized estimation error squared (needs truth). Expected value = dim(x)."""
    e = x_true - x_est
    return float(e @ np.linalg.inv(P) @ e)


def nis(y, S):
    """Normalized innovation squared. Expected value = dim(y)."""
    return float(y @ np.linalg.inv(S) @ y)


def chi2_bounds(dof, N=1, alpha=0.05):
    """Two-sided (1-alpha) confidence band for the average of N chi-square(dof) samples.

    Returns (lower, upper) already divided by N so it compares to the mean.
    """
    lo = chi2.ppf(alpha / 2, dof * N) / N
    hi = chi2.ppf(1 - alpha / 2, dof * N) / N
    return lo, hi


def autocorrelation(residuals, max_lag=None):
    """Normalized autocorrelation r(tau) of a scalar residual sequence."""
    y = np.asarray(residuals, float)
    y = y - 0.0                          # innovations are zero-mean by construction
    n = len(y)
    if max_lag is None:
        max_lag = min(n // 2, 30)
    denom = np.sum(y * y)
    r = np.array([np.sum(y[:n - k] * y[k:]) / denom for k in range(max_lag + 1)])
    return r


def whiteness_band(N, alpha=0.05):
    """+/- confidence bound for white-noise autocorrelation: ~1.96/sqrt(N)."""
    from scipy.stats import norm
    return norm.ppf(1 - alpha / 2) / np.sqrt(N)
