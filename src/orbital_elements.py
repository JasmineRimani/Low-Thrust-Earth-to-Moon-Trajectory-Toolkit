"""
Orbital element conversions.

Supported element sets
----------------------
* Classical Orbital Elements (COE):
    [a, e, i, omega, raan, nu]   [m, -, rad, rad, rad, rad]
* Modified Equinoctial Elements (MEE):
    [p, f, g, h, k, L]           [m, -, -, -, -, rad]
* ECI Cartesian:
    r [m, 3-vector], v [m/s, 3-vector]

References
----------
Vallado, D.A. (2013) "Fundamentals of Astrodynamics and Applications", 4th ed.
Curtis, H.D. (2014) "Orbital Mechanics for Engineering Students", 3rd ed.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# COE  <-->  ECI
# ---------------------------------------------------------------------------

def coe2eci(mu: float, coe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical orbital elements → ECI position/velocity.

    Parameters
    ----------
    mu  : float       Gravitational parameter [m³/s²].
    coe : (6,) array  [a, e, i, omega, raan, nu]  [m, -, rad, rad, rad, rad].

    Returns
    -------
    r : (3,) array   ECI position [m].
    v : (3,) array   ECI velocity [m/s].
    """
    a, e, inc, omega, raan, nu = coe

    slr = a * (1.0 - e**2)              # semi-latus rectum
    rm  = slr / (1.0 + e * np.cos(nu)) # radius

    arglat   = omega + nu
    sarglat  = np.sin(arglat)
    carglat  = np.cos(arglat)

    c4 = np.sqrt(mu / slr)
    c5 = e * np.cos(omega) + carglat
    c6 = e * np.sin(omega) + sarglat

    sinc = np.sin(inc);  cinc = np.cos(inc)
    sraan = np.sin(raan); craan = np.cos(raan)

    r = np.array([
        rm * (craan * carglat - sraan * cinc * sarglat),
        rm * (sraan * carglat + cinc * sarglat * craan),
        rm *  sinc * sarglat,
    ])
    v = np.array([
        -c4 * (craan * c6 + sraan * cinc * c5),
        -c4 * (sraan * c6 - craan * cinc * c5),
         c4 *  c5 * sinc,
    ])
    return r, v


def eci2coe(mu: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    ECI position/velocity → classical orbital elements.

    Parameters
    ----------
    mu : float       Gravitational parameter [m³/s²].
    r  : (3,) array  ECI position [m].
    v  : (3,) array  ECI velocity [m/s].

    Returns
    -------
    coe : (6,) array  [a, e, i, omega, raan, nu].
    """
    eps = 1.0e-10
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)
    vr   = np.dot(r, v) / rmag

    H    = np.cross(r, v)
    hmag = np.linalg.norm(H)

    inc  = np.arccos(np.clip(H[2] / hmag, -1.0, 1.0))

    N    = np.cross([0.0, 0.0, 1.0], H)
    nmag = np.linalg.norm(N)

    if nmag > eps:
        raan = np.arccos(np.clip(N[0] / nmag, -1.0, 1.0))
        if N[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    E_vec = (1.0 / mu) * ((vmag**2 - mu / rmag) * r - rmag * vr * v)
    e     = np.linalg.norm(E_vec)

    if nmag > eps:
        if e > eps:
            omega = np.arccos(np.clip(np.dot(N, E_vec) / (nmag * e), -1.0, 1.0))
            if E_vec[2] < 0.0:
                omega = 2.0 * np.pi - omega
        else:
            omega = 0.0
    else:
        omega = 0.0

    if e > eps:
        nu = np.arccos(np.clip(np.dot(E_vec, r) / (e * rmag), -1.0, 1.0))
        if vr < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        cp = np.cross(N, r)
        if cp[2] >= 0.0:
            nu = np.arccos(np.clip(np.dot(N, r) / (nmag * rmag), -1.0, 1.0))
        else:
            nu = 2.0 * np.pi - np.arccos(np.clip(np.dot(N, r) / (nmag * rmag), -1.0, 1.0))

    a = hmag**2 / (mu * (1.0 - e**2))
    return np.array([a, e, inc, omega, raan, nu])


# ---------------------------------------------------------------------------
# MEE  <-->  COE
# ---------------------------------------------------------------------------

def coe2mee(coe: np.ndarray) -> np.ndarray:
    """
    Classical orbital elements → modified equinoctial elements.

    Parameters
    ----------
    coe : (6,) array  [a, e, i, omega, raan, nu].

    Returns
    -------
    mee : (6,) array  [p, f, g, h, k, L].
    """
    a, e, inc, omega, raan, nu = coe
    p    = a * (1.0 - e**2)
    f    = e * np.cos(omega + raan)
    g    = e * np.sin(omega + raan)
    h    = np.tan(inc / 2.0) * np.cos(raan)
    k    = np.tan(inc / 2.0) * np.sin(raan)
    L    = (raan + omega + nu) % (2.0 * np.pi)
    return np.array([p, f, g, h, k, L])


def mee2coe(mee: np.ndarray) -> np.ndarray:
    """
    Modified equinoctial elements → classical orbital elements.

    Parameters
    ----------
    mee : (6,) array  [p, f, g, h, k, L].

    Returns
    -------
    coe : (6,) array  [a, e, i, omega, raan, nu].
    """
    p, f, g, h, k, L = mee
    a     = p / (1.0 - f**2 - g**2)
    e     = np.sqrt(f**2 + g**2)
    inc   = np.arctan2(2.0 * np.sqrt(h**2 + k**2), 1.0 - h**2 - k**2) % (2.0 * np.pi)
    omega = (np.arctan2(g, f) - np.arctan2(k, h)) % (2.0 * np.pi)
    raan  = np.arctan2(k, h) % (2.0 * np.pi)
    nu    = (L - np.arctan2(g, f)) % (2.0 * np.pi)
    return np.array([a, e, inc, omega, raan, nu])


# ---------------------------------------------------------------------------
# MEE  <-->  ECI
# ---------------------------------------------------------------------------

def mee2eci(mu: float, mee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified equinoctial elements → ECI position/velocity.

    Parameters
    ----------
    mu  : float       Gravitational parameter [m³/s²].
    mee : (6,) array  [p, f, g, h, k, L].

    Returns
    -------
    r : (3,) array   ECI position [m].
    v : (3,) array   ECI velocity [m/s].

    Reference: Vallado (2013), Algorithm 9.
    """
    p, f, g, h, k, L = mee

    mu_p    = np.sqrt(mu / p)
    cosL    = np.cos(L)
    sinL    = np.sin(L)
    w       = 1.0 + f * cosL + g * sinL
    radius  = p / w
    alpha2  = h**2 - k**2
    s2      = 1.0 + h**2 + k**2

    r = np.array([
        radius * (cosL + alpha2 * cosL + 2.0 * h * k * sinL) / s2,
        radius * (sinL - alpha2 * sinL + 2.0 * h * k * cosL) / s2,
        2.0 * radius * (h * sinL - k * cosL) / s2,
    ])

    v = np.array([
        -mu_p * (sinL + alpha2 * sinL - 2.0*h*k*cosL + g
                 - 2.0*f*h*k + alpha2*g) / s2,
        -mu_p * (-cosL + alpha2*cosL + 2.0*h*k*sinL - f
                 + 2.0*g*h*k + alpha2*f) / s2,
         2.0 * mu_p * (h*cosL + k*sinL + f*h + g*k) / s2,
    ])
    return r, v


def eci2mee(mu: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ECI position/velocity → modified equinoctial elements."""
    return coe2mee(eci2coe(mu, r, v))


# ---------------------------------------------------------------------------
# Rotation matrix  LVLH ← ECI
# ---------------------------------------------------------------------------

def lvlh_rotation(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotation matrix Q such that  vec_LVLH = Q @ vec_ECI.

    LVLH axes:  x̂ = radial,  ŷ = along-track,  ẑ = cross-track.

    Parameters
    ----------
    r, v : (3,) arrays  ECI position and velocity.

    Returns
    -------
    Q : (3, 3) array
    """
    r_hat = r / np.linalg.norm(r)
    h_vec = np.cross(r, v)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    return np.vstack([r_hat, t_hat, n_hat])
