"""
Circular LLO state in the CR3BP rotating frame with analytical gradients.

Used to build the LLO terminal constraint and its Jacobian for gradient-based
optimisation of NRHO ↔ LLO transfers.
"""

from __future__ import annotations
import numpy as np
from .cr3bp_dynamics import MU_M, OMEGA0, OMEGA0_VEC, P_M_VEC, R_MOON
from .frames import mci_to_cr3bp, rotation_z


def llo_state(
    x: np.ndarray,
    h_llo_nd: float,
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Circular LLO Cartesian state in the CR3BP rotating frame, plus gradients.

    Orbit parameterisation
    ----------------------
    x[0] = TA        departure epoch [CR3BP time]
    x[1] = u         argument of latitude [rad]
    x[2] = inc       inclination [rad]
    x[3] = raan      RAAN [rad]  (inertial)
    x[4] = offset    additional epoch offset [CR3BP time]

    Parameters
    ----------
    x        : (≥5,) parameter vector (only first 5 elements used).
    h_llo_nd : LLO altitude above Moon surface [CR3BP length].

    Returns
    -------
    (r_rot, v_rot,
     dr_du, dr_dinc, dr_draan,
     dv_du, dv_dinc, dv_draan)
    All in the CR3BP rotating frame [CR3BP units].
    """
    u, inc, raan = float(x[1]), float(x[2]), float(x[3])

    cu, su = np.cos(u), np.sin(u)
    ci, si = np.cos(inc), np.sin(inc)
    cR, sR = np.cos(raan), np.sin(raan)

    r_LLO  = R_MOON + h_llo_nd
    v_circ = np.sqrt(MU_M / r_LLO)

    # Position and velocity unit vectors in MCI
    r_unit = np.array([cR*cu - sR*su*ci,  sR*cu + cR*su*ci,  su*si])
    v_unit = np.array([-cR*su - sR*cu*ci, -sR*su + cR*cu*ci, cu*si])

    r_iner = r_LLO  * r_unit
    v_iner = v_circ * v_unit

    # Analytical gradients in MCI
    dr_du    = r_LLO  * v_unit
    dv_du    = v_circ * (-r_unit)

    dr_dinc  = r_LLO  * np.array([ sR*su*si, -cR*su*si,  su*ci])
    dv_dinc  = v_circ * np.array([ sR*cu*si, -cR*cu*si,  cu*ci])

    dr_draan = r_LLO  * np.array([-sR*cu - cR*su*ci,  cR*cu - sR*su*ci, 0.0])
    dv_draan = v_circ * np.array([ sR*su - cR*cu*ci, -cR*su - sR*cu*ci, 0.0])

    # Transform to rotating frame
    t_epoch   = float(x[0]) + float(x[4])
    Rz_neg    = rotation_z(-OMEGA0 * t_epoch)
    r_rot, v_rot = mci_to_cr3bp(r_iner, v_iner, t_epoch)

    # Rotate gradient vectors (position: pure rotation; velocity: add -Ω×r_mc)
    r_mc = r_rot - P_M_VEC

    def _rot_grad_pos(dg_mci):
        return Rz_neg @ dg_mci

    def _rot_grad_vel(dg_mci, dpos_mci):
        return Rz_neg @ dg_mci - np.cross(OMEGA0_VEC, Rz_neg @ dpos_mci)

    return (
        r_rot, v_rot,
        _rot_grad_pos(dr_du),    _rot_grad_pos(dr_dinc),    _rot_grad_pos(dr_draan),
        _rot_grad_vel(dv_du, dr_du),
        _rot_grad_vel(dv_dinc, dr_dinc),
        _rot_grad_vel(dv_draan, dr_draan),
    )


def llo_circular_speed_ms(h_llo_m: float) -> float:
    """Circular orbital speed [m/s] at LLO altitude h_llo_m [m]."""
    from .cr3bp_dynamics import MU_MOON_SI, R_MOON_M
    r = R_MOON_M + float(h_llo_m)
    return float(np.sqrt(MU_MOON_SI / r))
