"""J2 oblateness perturbation acceleration."""
import numpy as np
from ..constants import MU_EARTH, R_EARTH, J2


def j2_accel(r, mu=MU_EARTH, Re=R_EARTH, j2=J2):
    """J2 acceleration vector in ECI.

        a_x = -1.5 mu J2 Re^2 / r^5 * x (1 - 5 z^2/r^2)
        a_y = -1.5 mu J2 Re^2 / r^5 * y (1 - 5 z^2/r^2)
        a_z = -1.5 mu J2 Re^2 / r^5 * z (3 - 5 z^2/r^2)
    """
    x, y, z = r
    rn = np.linalg.norm(r)
    k = -1.5 * mu * j2 * Re**2 / rn**5
    zr2 = 5.0 * z**2 / rn**2
    return np.array([k * x * (1 - zr2),
                     k * y * (1 - zr2),
                     k * z * (3 - zr2)])
