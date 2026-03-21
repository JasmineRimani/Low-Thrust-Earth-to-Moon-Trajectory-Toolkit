"""
Lyapunov feedback control law for low-thrust orbit transfers.

The thrust direction is computed as a weighted superposition of the
analytically optimal directions for modifying each orbital element.

References
----------
Ruggiero, A., Pergola, P., Marcuccio, S., Andrenucci, M. (2011):
  "Low-Thrust Maneuvers for the Efficient Correction of Orbital Elements".
Petropoulos, A.E. (2004): "Simple Control Laws for Low-Thrust Orbit Transfers"
  (Q-law foundation).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ControlWeights:
    """
    Enable/disable flags for each orbital element.

    Set a weight to 0 to exclude that element from the control law.
    Tunable by the scipy.optimize wrapper in ``optimise.py``.
    """
    ka: float = 1.0   # semi-major axis
    ke: float = 0.0   # eccentricity
    ki: float = 0.0   # inclination
    kw: float = 1.0   # argument of perigee
    kraan: float = 0.0  # RAAN

    def as_array(self) -> np.ndarray:
        return np.array([self.ka, self.ke, self.ki, self.kw, self.kraan])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "ControlWeights":
        return cls(*x)


def _safe(val: float) -> float:
    """Replace NaN/Inf with zero."""
    return 0.0 if not np.isfinite(val) else val


def _element_weights(
    coe_current: np.ndarray,
    coe_initial: np.ndarray,
    coe_target: np.ndarray,
    enable: ControlWeights,
) -> tuple[float, float, float, float, float]:
    """
    Proportional remaining-error weights for each orbital element.

    Returns (ka, ke, ki, kw, kraan) scaled by enable flags.
    """
    _, e_c, i_c, w_c, raan_c, _ = coe_current
    a0, e0, i0, w0, raan0, _    = coe_initial
    af, ef, incf, wf, raanf, _  = coe_target

    def safe_weight(target, current, initial, flag):
        denom = abs(target - initial)
        if denom < 1e-12 or flag == 0.0:
            return 0.0
        return float(np.clip((target - current) / denom, -1.0, 1.0))

    ka   = safe_weight(af,    coe_current[0], a0,    enable.ka)
    ke   = safe_weight(ef,    e_c,            e0,    enable.ke)
    ki   = safe_weight(incf,  i_c,            i0,    enable.ki)
    kw   = safe_weight(wf,    w_c,            w0,    enable.kw)
    kraan= safe_weight(raanf, raan_c,         raan0, enable.kraan)
    return ka, ke, ki, kw, kraan


def thrust_direction_lvlh(
    coe_current: np.ndarray,
    coe_initial: np.ndarray,
    coe_target: np.ndarray,
    enable: ControlWeights,
) -> tuple[float, float, float]:
    """
    Compute the unit thrust vector in the LVLH frame.

    Based on the analytically optimal thrust angles for each element
    (Gauss variational equations) combined via the Lyapunov weighting.

    Parameters
    ----------
    coe_current : (6,) current classical orbital elements [m, -, rad…].
    coe_initial : (6,) initial orbital elements (for weight normalisation).
    coe_target  : (6,) target orbital elements.
    enable      : ControlWeights  element enable flags / scale factors.

    Returns
    -------
    ur, ut, un : float  radial, along-track, cross-track unit vector [-].
    """
    a_c, e_c, i_c, w_c, raan_c, nu_c = coe_current
    ka, ke, ki, kw, kraan = _element_weights(
        coe_current, coe_initial, coe_target, enable
    )

    # Eccentric anomaly
    E = np.arctan2(
        np.sqrt(max(1.0 - e_c**2, 0.0)) * np.sin(nu_c),
        1.0 + e_c * np.cos(nu_c),
    ) % (2.0 * np.pi)

    # ---- Optimal thrust angles for each element (radial, in-plane, out-of-plane)

    # Semi-major axis  →  tangential burn at periapsis/apoapsis
    alpha_a = _safe(np.arctan2(e_c * np.sin(nu_c), 1.0 + e_c * np.cos(nu_c)))
    beta_a  = 0.0

    # Eccentricity
    alpha_e = _safe(np.arctan2(np.sin(nu_c), np.cos(nu_c) + np.cos(E)))
    beta_e  = 0.0

    # Argument of perigee
    tan_nu  = np.tan(nu_c) if abs(np.cos(nu_c)) > 1e-9 else np.inf
    if not np.isfinite(tan_nu) or abs(tan_nu) < 1e-12:
        tangential_term = np.copysign(1e12, 1.0 + e_c * np.cos(nu_c))
    else:
        tangential_term = (1.0 + e_c * np.cos(nu_c)) / tan_nu
    alpha_w = _safe(np.arctan2(
        tangential_term,
        2.0 + e_c * np.cos(nu_c),
    ))
    denom_w = ((np.sin(alpha_w - nu_c) * (1.0 + e_c * np.cos(nu_c))
                - np.cos(alpha_w) * np.sin(nu_c)))
    tan_i   = (1.0 / np.tan(i_c)) if abs(np.sin(i_c)) > 1e-9 else 1e12
    beta_w  = _safe(np.arctan2(
        e_c * tan_i * np.sin(w_c + nu_c),
        denom_w if abs(denom_w) > 1e-12 else 1e-12,
    ))

    # Inclination  →  normal burn at argument-of-latitude = 90°
    alpha_i = 0.0
    beta_i  = np.sign(np.cos(w_c + nu_c)) * 0.5 * np.pi
    beta_i  = _safe(beta_i)

    # RAAN  →  normal burn at argument-of-latitude = 0° or 180°
    alpha_raan = 0.0
    beta_raan  = np.sign(np.sin(w_c + nu_c)) * 0.5 * np.pi
    beta_raan  = _safe(beta_raan)

    # ---- Superposition in RTN
    ur = (enable.ka * ka * np.cos(beta_a) * np.sin(alpha_a)
          + enable.ke * ke * np.cos(beta_e) * np.sin(alpha_e)
          + enable.kw * kw * np.cos(beta_w) * np.sin(alpha_w))

    ut = (enable.ka * ka * np.cos(beta_a) * np.cos(alpha_a)
          + enable.ke * ke * np.cos(beta_e) * np.cos(alpha_e)
          + enable.kw * kw * np.cos(beta_w) * np.cos(alpha_w))

    un = (enable.ka * ka * np.sin(beta_a)
          + enable.ke * ke * np.sin(beta_e)
          + enable.kw * kw * np.sin(beta_w)
          + enable.ki * ki * np.sin(beta_i)
          + enable.kraan * kraan * np.sin(beta_raan))

    # Normalise
    q = np.array([ur, ut, un])
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0
    q /= norm
    return float(q[0]), float(q[1]), float(q[2])


def maneuver_efficiency(
    coe: np.ndarray,
    mu: float,
    enable: ControlWeights,
) -> tuple[float, float, float, float, float]:
    """
    Orbit-position-dependent maneuver efficiency factors η_a, η_e, η_i, η_ω, η_Ω.

    Used in the Moon-phase propagator to weight the thrust more accurately.
    These are the Gauss variational equation sensitivities evaluated at the
    current orbit position.

    Returns
    -------
    eta_a, eta_e, eta_i, eta_w, eta_raan : float
    """
    a, e, i, omega, raan, nu = coe
    e_eff = float(np.clip(e, 0.0, 0.999))
    denom = max(1.0 + e_eff * np.cos(nu), 1e-8)

    # Semi-major axis efficiency: proportional to radial velocity fraction
    v_circ  = np.sqrt(abs(mu / a)) if a > 0 else 0.0
    factor  = np.sqrt(max((a / mu) * (1.0 - e_eff) / (1.0 + e_eff), 0.0)) if a > 0 else 0.0
    eta_a   = enable.ka * v_circ * factor

    # Eccentricity efficiency
    eta_e   = enable.ke * 2.0 * (1.0 + 2.0 * e_eff * np.cos(nu) + np.cos(nu)**2) / denom

    # Inclination efficiency (maximised when thrust is at 90° latitude arg)
    cos_arg = np.cos(omega + nu)
    eta_i   = enable.ki * (abs(cos_arg) / denom) * (
                  np.sqrt(max(1.0 - e_eff**2 * np.sin(omega)**2, 0.0))
                  - e_eff * abs(np.cos(omega)))

    # RAAN efficiency
    sin_arg = np.sin(omega + nu)
    eta_raan= enable.kraan * (abs(sin_arg) / denom) * (
                  np.sqrt(max(1.0 - e_eff**2 * np.cos(omega)**2, 0.0))
                  - e_eff * abs(np.sin(omega)))

    # Argument of perigee efficiency (approximation)
    eta_w   = enable.kw * 0.25 * (1.0 + np.sin(nu)**2) / denom

    return eta_a, eta_e, eta_i, eta_w, eta_raan
