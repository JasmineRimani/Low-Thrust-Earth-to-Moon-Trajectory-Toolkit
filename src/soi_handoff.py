"""
Sphere-of-Influence (SOI) handoff utilities.

Bridges the two propagation worlds:

  low-thrust propagator  (ECI, two-body Moon-centred, SI units)
      ↓  soi_to_cr3bp()
  CR3BP    (Earth-Moon rotating frame, dimensionless)

and back:

  CR3BP
      ↓  cr3bp_to_soi()
  low-thrust propagator  (ECI SI)

The conversion accounts for:
- Unit normalisation (SI → CR3BP dimensionless)
- Frame rotation from Moon-centred inertial (MCI) to the Earth-Moon rotating frame
- Barycentric offset: the rotating-frame origin is the Earth-Moon barycentre,
  not the Moon centre

Usage
-----
At the end of ``propagate_moon_phase`` the spacecraft is somewhere inside the
Moon SOI in a Moon-centred ECI frame.  Feed the final (r_eci, v_eci) pair into
``soi_to_cr3bp`` to get a CR3BP state ready for ``nrho_to_llo``.

At the end of ``nrho_to_llo`` the spacecraft is in LLO in CR3BP coordinates.
Feed into ``cr3bp_llo_to_coe`` to get classical orbital elements for the
descent / ascent analysis.
"""

from __future__ import annotations
import numpy as np

from .cr3bp.cr3bp_dynamics import (
    A_SCALE, T_SCALE, V_SCALE,
    P_M_VEC, OMEGA0_VEC, OMEGA0,
    R_MOON_M, MU_MOON_SI,
)
from .cr3bp.frames import mci_to_cr3bp, cr3bp_to_mci
from .orbital_elements import eci2coe


def soi_to_cr3bp(
    r_moon_eci_m: np.ndarray,
    v_moon_eci_ms: np.ndarray,
    t_epoch_nd: float = 0.0,
) -> np.ndarray:
    """
    Convert a Moon-centred ECI state (SI) to a CR3BP rotating-frame state.

    Parameters
    ----------
    r_moon_eci_m  : (3,) Moon-centred ECI position [m].
    v_moon_eci_ms : (3,) Moon-centred ECI velocity [m/s].
    t_epoch_nd    : CR3BP epoch at which this state is defined [dimensionless].
                    Defines the orientation of the rotating frame.
                    Use 0.0 if the epoch is arbitrary / not yet fixed.

    Returns
    -------
    state_cr3bp : (6,) CR3BP rotating-frame *barycentric* state [dimensionless].
    """
    # Normalise to CR3BP units (MCI, Moon-centred)
    r_mci_nd = np.asarray(r_moon_eci_m,  dtype=float) / A_SCALE
    v_mci_nd = np.asarray(v_moon_eci_ms, dtype=float) / V_SCALE

    # Transform to rotating frame (barycentric)
    r_rot, v_rot = mci_to_cr3bp(r_mci_nd, v_mci_nd, t_epoch_nd)
    return np.concatenate([r_rot, v_rot])


def cr3bp_to_mci_si(
    state_cr3bp: np.ndarray,
    t_epoch_nd: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a CR3BP rotating-frame state back to Moon-centred ECI (SI).

    Parameters
    ----------
    state_cr3bp : (6,) CR3BP barycentric state [dimensionless].
    t_epoch_nd  : CR3BP epoch [dimensionless].

    Returns
    -------
    r_mci_m  : (3,) Moon-centred inertial position [m].
    v_mci_ms : (3,) Moon-centred inertial velocity [m/s].
    """
    r_rot = state_cr3bp[:3]
    v_rot = state_cr3bp[3:]
    r_mci_nd, v_mci_nd = cr3bp_to_mci(r_rot, v_rot, t_epoch_nd)
    return r_mci_nd * A_SCALE, v_mci_nd * V_SCALE


def cr3bp_llo_to_coe(
    state_cr3bp: np.ndarray,
    t_epoch_nd: float = 0.0,
) -> np.ndarray:
    """
    Convert a CR3BP LLO state to Moon-centred classical orbital elements.

    Parameters
    ----------
    state_cr3bp : (6,) CR3BP rotating-frame state at LLO [dimensionless].
    t_epoch_nd  : CR3BP epoch [dimensionless].

    Returns
    -------
    coe : (6,) [a_m, e, i_rad, omega_rad, raan_rad, nu_rad]
          Semi-major axis in metres, angles in radians.
    """
    r_m, v_ms = cr3bp_to_mci_si(state_cr3bp, t_epoch_nd)
    return eci2coe(MU_MOON_SI, r_m, v_ms)


def magneto_soi_exit_to_cr3bp(
    propagation_result,
    t_epoch_nd: float = 0.0,
) -> np.ndarray:
    """
    Extract the SOI-exit state from a ``PropagationResult`` and
    convert it to a CR3BP state.

    The Moon-phase propagator works in Moon-centred ECI (SI).
    This helper takes the *last* point of the propagation (= SOI boundary
    or target orbit) and hands it off to the CR3BP layer.

    Parameters
    ----------
    propagation_result : ``src.propagator.PropagationResult``
        Result of ``propagate_moon_phase``.
    t_epoch_nd : float
        CR3BP epoch to assign to this state [dimensionless].

    Returns
    -------
    state_cr3bp : (6,) CR3BP state [dimensionless].
    """
    r_m  = propagation_result.r_eci[-1]   # last ECI position [m]
    v_ms = propagation_result.v_eci[-1]   # last ECI velocity [m/s]
    return soi_to_cr3bp(r_m, v_ms, t_epoch_nd)
