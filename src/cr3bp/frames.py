"""
Reference frame transforms for NRHO / LLO mission analysis.

Three frame families
---------------------
- **CR3BP rotating frame**: Earth-Moon barycentric, x-axis toward Moon, z up.
  Used for NRHO propagation and coast arcs.
- **Moon-centred inertial (MCI)**: Moon at origin, fixed axes.
  Used for LLO geometry.
- **LVLH**: Local-vertical local-horizontal, tied to a spacecraft state.
  Used for powered-flight reasoning and burn vectors.

All quantities in dimensionless CR3BP units unless SI is explicitly noted.
"""

from __future__ import annotations
import numpy as np
from .cr3bp_dynamics import OMEGA0, OMEGA0_VEC, P_M_VEC


def rotation_z(theta: float) -> np.ndarray:
    """Right-handed 3×3 rotation matrix for a +theta rotation about z."""
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0],
                     [st,  ct, 0.0],
                     [0.0, 0.0, 1.0]])


def cr3bp_to_mci(
    r_rot: np.ndarray,
    v_rot: np.ndarray,
    t_nd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CR3BP rotating-frame barycentric state → Moon-centred inertial (MCI).

    Parameters
    ----------
    r_rot, v_rot : rotating-frame barycentric position/velocity [CR3BP].
    t_nd         : CR3BP epoch [dimensionless time].
    """
    theta  = float(OMEGA0) * float(t_nd)
    r_mc   = np.asarray(r_rot, dtype=float) - P_M_VEC
    rot    = rotation_z(theta)
    r_mci  = rot @ r_mc
    v_mci  = rot @ (np.asarray(v_rot, dtype=float) + np.cross(OMEGA0_VEC, r_mc))
    return r_mci, v_mci


def mci_to_cr3bp(
    r_mci: np.ndarray,
    v_mci: np.ndarray,
    t_nd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moon-centred inertial (MCI) → CR3BP rotating frame (barycentric).

    Parameters
    ----------
    r_mci, v_mci : MCI position/velocity [CR3BP].
    t_nd         : CR3BP epoch [dimensionless time].
    """
    theta   = float(OMEGA0) * float(t_nd)
    Rz_neg  = rotation_z(-theta)
    r_mc    = Rz_neg @ np.asarray(r_mci, dtype=float)
    v_rot   = Rz_neg @ np.asarray(v_mci, dtype=float) - np.cross(OMEGA0_VEC, r_mc)
    r_rot   = r_mc + P_M_VEC
    return r_rot, v_rot


def lvlh_basis_from_mci(
    r_mci: np.ndarray,
    v_mci: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return LVLH orthonormal basis vectors expressed in MCI coordinates.

    Returns
    -------
    (r_hat, theta_hat, h_hat)
        Radial-outward, along-track, orbit-normal unit vectors.
    """
    r_vec = np.asarray(r_mci, dtype=float)
    v_vec = np.asarray(v_mci, dtype=float)

    r_norm = np.linalg.norm(r_vec)
    if r_norm <= 0.0:
        raise ValueError("LVLH basis requires non-zero position vector.")

    h_vec  = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm <= 0.0:
        raise ValueError("LVLH basis requires non-collinear r and v.")

    r_hat     = r_vec / r_norm
    h_hat     = h_vec / h_norm
    theta_hat = np.cross(h_hat, r_hat)
    theta_hat /= np.linalg.norm(theta_hat)
    return r_hat, theta_hat, h_hat
