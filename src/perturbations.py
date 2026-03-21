"""
Orbital perturbation accelerations in the LVLH frame.

Each function returns (delta_r, delta_t, delta_n) — radial, along-track,
and cross-track acceleration components [m/s²].

Fixes vs. original MATLAB
--------------------------
* Third-body: removed the element-wise ``abs()`` that destroyed direction info.
* Solar radiation pressure: solar angle now computed from geometry, not
  hard-coded to π/2.
* Eclipse: full 3-D conical shadow model (umbra + penumbra).
"""

from __future__ import annotations
import numpy as np
from .constants import (
    J2_EARTH, R_EARTH, MU_EARTH,
    J2_MOON, R_MOON,
    OMEGA_EARTH,
    P_SOLAR, R_SUN,
)
from .orbital_elements import lvlh_rotation


# ---------------------------------------------------------------------------
# J2 gravity
# ---------------------------------------------------------------------------

def j2_acceleration(
    r_eci: np.ndarray,
    mu: float,
    j2: float,
    r_body: float,
    Q: np.ndarray,
) -> tuple[float, float, float]:
    """
    J2 oblateness perturbation.

    Parameters
    ----------
    r_eci   : (3,) ECI position [m].
    mu      : gravitational parameter [m³/s²].
    j2      : J2 coefficient [-].
    r_body  : equatorial radius of central body [m].
    Q       : (3,3) LVLH←ECI rotation matrix.

    Returns
    -------
    delta_r, delta_t, delta_n : float  LVLH components [m/s²].
    """
    r  = np.linalg.norm(r_eci)
    k  = -1.5 * mu * j2 * r_body**2 / r**4
    zr = r_eci[2] / r

    g_eci = k * np.array([
        (1.0 - 5.0 * zr**2) * (r_eci[0] / r),
        (1.0 - 5.0 * zr**2) * (r_eci[1] / r),
        (3.0 - 5.0 * zr**2) * (r_eci[2] / r),
    ])
    g_lvlh = Q @ g_eci
    return g_lvlh[0], g_lvlh[1], g_lvlh[2]


# ---------------------------------------------------------------------------
# Third-body perturbation  (Battin formulation)
# ---------------------------------------------------------------------------

def third_body_acceleration(
    r_eci: np.ndarray,
    s_vec: np.ndarray,
    mu_3: float,
    Q: np.ndarray,
) -> tuple[float, float, float]:
    """
    Third-body perturbation using the Battin q-function.

    Avoids the near-cancellation issue in the direct formulation.

    Parameters
    ----------
    r_eci : (3,) ECI position of spacecraft [m].
    s_vec : (3,) ECI position of perturbing body *from central body* [m].
             **Pass the actual vector, not abs().**
    mu_3  : gravitational parameter of perturbing body [m³/s²].
    Q     : (3,3) LVLH←ECI rotation matrix.

    Returns
    -------
    delta_r, delta_t, delta_n : float  LVLH components [m/s²].
    """
    d = r_eci - s_vec                              # sc → perturbing body
    q = np.dot(r_eci, r_eci - 2.0 * s_vec) / np.dot(s_vec, s_vec)
    F = q * (3.0 + 3.0 * q + q**2) / (1.0 + (1.0 + q)**1.5)

    a_eci = -(mu_3 / np.linalg.norm(d)**3) * (r_eci + F * s_vec)
    a_lvlh = Q @ a_eci
    return a_lvlh[0], a_lvlh[1], a_lvlh[2]


# ---------------------------------------------------------------------------
# Solar radiation pressure
# ---------------------------------------------------------------------------

def srp_acceleration(
    r_eci: np.ndarray,
    r_sun_eci: np.ndarray,   # Sun position from central body [m]
    mass: float,
    S_sp: float,
    S_lat: float,
    c_r: float,
    Q: np.ndarray,
    in_eclipse: bool = False,
) -> tuple[float, float, float]:
    """
    Solar radiation pressure acceleration.

    The solar incidence angle is computed from the actual geometry
    (fixes the hardcoded π/2 in the original MATLAB).

    Parameters
    ----------
    r_eci     : (3,) spacecraft ECI position [m].
    r_sun_eci : (3,) Sun ECI position from central body [m].
                     **Pass the signed vector, not abs().**
    mass      : spacecraft mass [kg].
    S_sp      : solar-panel projected area [m²].
    S_lat     : lateral body area [m²].
    c_r       : reflectivity coefficient [-].
    Q         : (3,3) LVLH←ECI rotation.
    in_eclipse: bool  If True, SRP is zero.

    Returns
    -------
    delta_r, delta_t, delta_n : float  LVLH components [m/s²].
    """
    if in_eclipse or mass <= 0.0:
        return 0.0, 0.0, 0.0

    # Unit vector from spacecraft towards Sun
    sun_from_sc = r_sun_eci - r_eci
    sun_hat     = sun_from_sc / np.linalg.norm(sun_from_sc)

    # Cosine of incidence angle (angle between SC-Sun vector and SC position)
    cos_theta   = np.clip(np.dot(r_eci / np.linalg.norm(r_eci), sun_hat), -1.0, 1.0)
    sin_theta   = np.sqrt(max(0.0, 1.0 - cos_theta**2))

    S_eff = S_sp + S_lat * sin_theta               # effective illuminated area
    F_sr  = P_SOLAR * S_eff * c_r                  # SRP force [N]
    a_sr  = F_sr / mass                             # acceleration magnitude [m/s²]

    # Direction: pushes spacecraft away from Sun
    a_eci   = -a_sr * sun_hat
    a_lvlh  = Q @ a_eci
    return a_lvlh[0], a_lvlh[1], a_lvlh[2]


# ---------------------------------------------------------------------------
# Aerodynamic drag  (significant only in LEO, < 1500 km)
# ---------------------------------------------------------------------------

# Tabulated atmosphere model from Vallado/Mostaza-Prieto et al.
_ALT_KM  = np.array([100, 200, 300, 400, 500, 600, 700, 800, 920, 1000, 1500], dtype=float) * 1e3
_RHO_TAB = np.array([4.974e-6, 2.557e-10, 1.708e-11, 2.249e-12,
                      3.916e-13, 8.07e-14, 2.043e-14, 7.069e-15,
                      2.210e-15, 1.150e-15, 2.3e-16])


def drag_acceleration(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    mass: float,
    C_d: float,
    S_drag: float,
    Q: np.ndarray,
    omega_body: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Aerodynamic drag (relevant only below ~1500 km altitude).

    Parameters
    ----------
    r_eci      : (3,) ECI position [m].
    v_eci      : (3,) ECI velocity [m/s].
    mass       : spacecraft mass [kg].
    C_d        : drag coefficient [-].
    S_drag     : drag reference area [m²].
    Q          : (3,3) LVLH←ECI rotation.
    omega_body : (3,) angular velocity vector of atmosphere [rad/s].
                  Defaults to Earth rotation axis.

    Returns
    -------
    delta_r, delta_t, delta_n : float  LVLH components [m/s²].
    """
    altitude = np.linalg.norm(r_eci) - R_EARTH
    if altitude > 1_500_000.0 or mass <= 0.0:
        return 0.0, 0.0, 0.0

    # Atmospheric density by interpolation
    rho = float(np.interp(altitude, _ALT_KM, _RHO_TAB, left=0.0, right=0.0))
    if rho == 0.0 or not np.isfinite(rho):
        return 0.0, 0.0, 0.0

    if omega_body is None:
        omega_body = np.array([0.0, 0.0, OMEGA_EARTH])

    v_rel = v_eci - np.cross(omega_body, r_eci)
    B     = C_d * S_drag / mass                        # ballistic coefficient [m²/kg]
    a_drag_eci = -0.5 * B * rho * np.linalg.norm(v_rel) * v_rel

    a_lvlh = Q @ a_drag_eci
    return a_lvlh[0], a_lvlh[1], a_lvlh[2]


# ---------------------------------------------------------------------------
# Eclipse detection  (3-D conical penumbra model)
# ---------------------------------------------------------------------------

def eclipse_conical(
    r_eci: np.ndarray,
    r_sun_eci: np.ndarray,
    r_body: float = R_EARTH,
) -> bool:
    """
    Conical shadow model (umbra + penumbra).

    Tests whether the spacecraft is inside the penumbral cone cast by a
    spherical central body (Earth or Moon) against the Sun.

    Reference
    ---------
    Ortiz-Gomez & Rickman, "Method for the Calculation of Spacecraft
    Umbra and Penumbra Shadow Terminator Points".

    Parameters
    ----------
    r_eci     : (3,) spacecraft ECI position [m].
    r_sun_eci : (3,) Sun position from central body [m].
    r_body    : equatorial radius of occulting body [m].

    Returns
    -------
    bool  True if spacecraft is in shadow (penumbra or umbra).
    """
    s_norm = np.linalg.norm(r_sun_eci)

    # Penumbra half-angle
    alpha_p = np.arcsin((r_body + R_SUN) / s_norm)
    # Umbra half-angle
    alpha_u = np.arcsin((R_SUN - r_body) / max(s_norm - r_body, 1.0))

    s_hat = r_sun_eci / s_norm             # unit vector towards Sun

    # Project spacecraft onto the Sun–body axis
    proj  = np.dot(r_eci, s_hat)          # signed distance along axis

    # Only in shadow if spacecraft is on the anti-Sun side
    if proj > 0.0:
        return False

    # Perpendicular distance from Sun–body axis
    perp = np.linalg.norm(r_eci - proj * s_hat)

    # Penumbra cone radius at spacecraft location
    dist_from_body = abs(proj)
    r_penumbra = r_body + dist_from_body * np.tan(alpha_p)

    return perp <= r_penumbra
