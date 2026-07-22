"""Coordinate-frame rotations."""
import numpy as np


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def perifocal_to_eci(Omega, i, omega):
    """Rotation matrix PQW -> ECI:  R = Rz(Om) Rx(i) Rz(om).

    (Standard 3-1-3 sequence; equivalent to the Rz(-Om)Rx(-i)Rz(-om) form
    applied to inertial->perifocal, transposed.)
    """
    return Rz(Omega) @ Rx(i) @ Rz(omega)
