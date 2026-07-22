"""Orbit propagator: assembles the force model and integrates.

Also propagates the state-transition matrix Phi via the variational equation
Phi_dot = A Phi, used by the EKF covariance prediction.
"""
import numpy as np
from scipy.integrate import solve_ivp

from ..constants import MU_EARTH
from ..core.two_body import two_body_accel, dynamics_jacobian
from ..perturbations.j2 import j2_accel
from ..perturbations.drag import drag_accel


class ForceModel:
    """Configurable acceleration model."""
    def __init__(self, mu=MU_EARTH, use_j2=False, use_drag=False, beta=50.0):
        self.mu = mu
        self.use_j2 = use_j2
        self.use_drag = use_drag
        self.beta = beta

    def accel(self, r, v):
        a = two_body_accel(r, self.mu)
        if self.use_j2:
            a = a + j2_accel(r, self.mu)
        if self.use_drag:
            a = a + drag_accel(r, v, self.beta)
        return a

    def deriv(self, t, x):
        r, v = x[:3], x[3:6]
        return np.concatenate([v, self.accel(r, v)])


def propagate(x0, t_span, force: ForceModel, t_eval=None, rtol=1e-10, atol=1e-12):
    """Integrate the state over t_span. Returns (t, X) with X shape (len(t), 6)."""
    sol = solve_ivp(force.deriv, t_span, np.asarray(x0, float),
                    t_eval=t_eval, rtol=rtol, atol=atol, method="DOP853")
    return sol.t, sol.y.T


def propagate_with_stm(x0, dt, force: ForceModel, rtol=1e-10, atol=1e-12):
    """Propagate state AND 6x6 STM over a single step dt (for the EKF predict).

    Integrates the augmented system [x; vec(Phi)] with Phi(0)=I, Phi_dot=A(x)Phi.
    Returns (x_next, Phi).
    """
    n = 6
    y0 = np.concatenate([np.asarray(x0, float), np.eye(n).ravel()])

    def aug(t, y):
        x = y[:n]
        Phi = y[n:].reshape(n, n)
        dx = force.deriv(t, x)
        A = dynamics_jacobian(x[:3], force.mu)   # two-body gradient (dominant term)
        dPhi = A @ Phi
        return np.concatenate([dx, dPhi.ravel()])

    sol = solve_ivp(aug, (0, dt), y0, rtol=rtol, atol=atol, method="DOP853")
    yf = sol.y[:, -1]
    return yf[:n], yf[n:].reshape(n, n)
