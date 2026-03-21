"""
Modified Equinoctial Element (MEE) equations of motion.

The state vector is  y = [p, f, g, h, k, L, m]
where the 7th component m is spacecraft mass [kg].

Perturbations included
----------------------
* Thrust (Lyapunov guidance law)
* J2 oblateness of central body
* Third-body gravity (two perturbers)
* Solar radiation pressure
* Aerodynamic drag (Earth phase only, altitude < 1500 km)

All perturbations are expressed in the LVLH frame and then inserted into
the Gauss variational equations for MEE (Vallado 2013, Algorithm 9).
"""

from __future__ import annotations
import numpy as np
from .orbital_elements import mee2eci, mee2coe, lvlh_rotation
from .perturbations import (
    j2_acceleration,
    third_body_acceleration,
    srp_acceleration,
    drag_acceleration,
    eclipse_conical,
)
from .control import ControlWeights, thrust_direction_lvlh, maneuver_efficiency


def _gauss_mee(
    mee: np.ndarray,
    mu: float,
    dr: float,
    dt: float,
    dn: float,
) -> np.ndarray:
    """
    Gauss variational equations for Modified Equinoctial Elements.

    Parameters
    ----------
    mee     : (6,) MEE state [p, f, g, h, k, L].
    mu      : gravitational parameter [m³/s²].
    dr, dt, dn : radial, along-track, cross-track perturbing accelerations [m/s²].

    Returns
    -------
    dmee_dt : (6,) time derivatives.
    """
    p, f, g, h, k, L = mee

    sinL = np.sin(L)
    cosL = np.cos(L)
    w    = 1.0 + f * cosL + g * sinL
    s2   = 1.0 + h**2 + k**2
    sq   = np.sqrt(p / mu)

    dp = (2.0 * p / w) * sq * dt

    df = (sq * (sinL * dr
                + ((w + 1.0) * cosL + f) * dt / w
                - (h * sinL - k * cosL) * g * dn / w))

    dg = (sq * (-cosL * dr
                + ((w + 1.0) * sinL + g) * dt / w
                + (h * sinL - k * cosL) * f * dn / w))

    dh = sq * (s2**2 * cosL * dn) / (2.0 * w)

    dk = sq * (s2**2 * sinL * dn) / (2.0 * w)

    dL = (np.sqrt(mu * p) * (w / p)**2
          + (1.0 / w) * sq * (h * sinL - k * cosL) * dn)

    return np.array([dp, df, dg, dh, dk, dL])


# ---------------------------------------------------------------------------
# Earth-phase ODE  (GTO → Moon SOI)
# ---------------------------------------------------------------------------

def meeeqm_earth(
    t: float,
    y: np.ndarray,
    *,
    mu: float,
    mu_moon: float,
    mu_sun: float,
    j2: float,
    r_body: float,
    coe_initial: np.ndarray,
    coe_target: np.ndarray,
    enable: ControlWeights,
    n_thrusters: int,
    thrust_per_thruster: float,
    isp: float,
    g0: float,
    Cd: float,
    S_sl: float,
    S_sp: float,
    c_r: float,
    enable_eclipse: bool,
    smart_mode: bool,
    get_third_body: callable,   # fn(t) → (r_moon, r_sun) in ECI [m]
) -> np.ndarray:
    """
    Time derivative of [p, f, g, h, k, L, mass] for the Earth-centred phase.

    Parameters
    ----------
    t          : current time [s] (relative to mission epoch).
    y          : (7,) state vector [p, f, g, h, k, L, mass].
    (keyword)  : physical parameters (see signature).

    Returns
    -------
    ydot : (7,) state derivative.
    """
    mee  = y[:6]
    mass = max(y[6], 1.0)   # guard against numeric underflow

    # --- State ---
    r_eci, v_eci = mee2eci(mu, mee)
    coe          = mee2coe(mee)
    Q            = lvlh_rotation(r_eci, v_eci)

    # --- Third-body positions ---
    r_moon_eci, r_sun_eci = get_third_body(t)

    # --- Eclipse ---
    in_eclipse = False
    if enable_eclipse:
        in_eclipse = eclipse_conical(r_eci, r_sun_eci)

    # --- Thrust ---
    total_thrust = n_thrusters * thrust_per_thruster
    if in_eclipse:
        thrust_acc = 0.0
        mass_dot   = 0.0
    else:
        # Smart-mode: coasting near apoapsis when efficiency is low
        if smart_mode:
            v_mag   = np.linalg.norm(v_eci)
            factor  = np.sqrt(coe[0] / mu * (1.0 - coe[1]) / (1.0 + coe[1]))
            eta_eff = v_mag * factor
            r_peri  = coe[0] * (1.0 - coe[1])
            if coe[0] < 200_000e3 and r_peri >= 20_000e3 and eta_eff < 0.6:
                total_thrust = 0.0

        thrust_acc = total_thrust / mass if total_thrust > 0.0 else 0.0
        mass_dot   = -total_thrust / (isp * g0) if total_thrust > 0.0 else 0.0

    ur, ut, un = thrust_direction_lvlh(coe, coe_initial, coe_target, enable)

    # --- Perturbations ---
    dr_j2, dt_j2, dn_j2 = j2_acceleration(r_eci, mu, j2, r_body, Q)

    dr_moon, dt_moon, dn_moon = third_body_acceleration(r_eci, r_moon_eci, mu_moon, Q)
    dr_sun,  dt_sun,  dn_sun  = third_body_acceleration(r_eci, r_sun_eci,  mu_sun,  Q)

    dr_srp, dt_srp, dn_srp = srp_acceleration(
        r_eci, r_sun_eci, mass, S_sp, S_sl, c_r, Q, in_eclipse
    )

    dr_drag, dt_drag, dn_drag = drag_acceleration(r_eci, v_eci, mass, Cd, S_sl + S_sp, Q)

    # Total perturbation
    dr = thrust_acc * ur + dr_j2 + dr_moon + dr_sun + dr_srp + dr_drag
    dt_ = thrust_acc * ut + dt_j2 + dt_moon + dt_sun + dt_srp + dt_drag
    dn = thrust_acc * un + dn_j2 + dn_moon + dn_sun + dn_srp + dn_drag

    dmee = _gauss_mee(mee, mu, dr, dt_, dn)
    return np.append(dmee, mass_dot)


# ---------------------------------------------------------------------------
# Moon-phase ODE  (Moon SOI → NHRO)
# ---------------------------------------------------------------------------

def meeeqm_moon(
    t: float,
    y: np.ndarray,
    *,
    mu: float,          # Moon gravitational parameter
    mu_earth: float,
    mu_sun: float,
    j2: float,
    r_body: float,
    coe_initial: np.ndarray,
    coe_target: np.ndarray,
    enable: ControlWeights,
    n_thrusters: int,
    thrust_per_thruster: float,
    isp: float,
    g0: float,
    S_sp: float,
    c_r: float,
    get_third_body: callable,  # fn(t) → (r_earth, r_sun) in Moon-centred ECI [m]
) -> np.ndarray:
    """
    Time derivative of [p, f, g, h, k, L, mass] for the Moon-centred phase.

    Uses the maneuver efficiency weighting (from ``meeeqm_moon.m``) in
    addition to the basic Lyapunov weights.
    """
    mee  = y[:6]
    mass = max(y[6], 1.0)

    r_eci, v_eci = mee2eci(mu, mee)
    coe          = mee2coe(mee)
    Q            = lvlh_rotation(r_eci, v_eci)

    r_earth_eci, r_sun_eci = get_third_body(t)

    # --- Thrust ---
    total_thrust = n_thrusters * thrust_per_thruster
    thrust_acc   = total_thrust / mass
    mass_dot     = -total_thrust / (isp * g0)

    ur, ut, un = thrust_direction_lvlh(coe, coe_initial, coe_target, enable)

    # Apply maneuver efficiency weighting
    eta_a, eta_e, eta_i, eta_w, eta_raan = maneuver_efficiency(coe, mu, enable)
    ur = eta_a * ur
    ut = eta_a * ut
    # (efficiency modulation — normalise after combining)
    q = np.array([ur, ut, un])
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    ur, ut, un = q

    # --- Perturbations (J2 + third body; no drag in lunar orbit) ---
    dr_j2, dt_j2, dn_j2 = j2_acceleration(r_eci, mu, j2, r_body, Q)

    dr_earth, dt_earth, dn_earth = third_body_acceleration(r_eci, r_earth_eci, mu_earth, Q)
    dr_sun,   dt_sun,   dn_sun   = third_body_acceleration(r_eci, r_sun_eci,   mu_sun,   Q)

    dr_srp, dt_srp, dn_srp = srp_acceleration(
        r_eci, r_sun_eci, mass, S_sp, 0.0, c_r, Q, False
    )

    dr  = thrust_acc * ur + dr_j2 + dr_earth + dr_sun + dr_srp
    dt_ = thrust_acc * ut + dt_j2 + dt_earth + dt_sun + dt_srp
    dn  = thrust_acc * un + dn_j2 + dn_earth + dn_sun + dn_srp

    dmee = _gauss_mee(mee, mu, dr, dt_, dn)
    return np.append(dmee, mass_dot)
