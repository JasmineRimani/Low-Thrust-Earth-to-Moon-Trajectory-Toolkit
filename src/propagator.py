"""
Low-thrust trajectory propagator.

Uses ``scipy.integrate.solve_ivp`` with the DOP853 (Dormand-Prince 8th order)
method — replacing the custom RKF7(8) in the MATLAB original with a properly
adaptive integrator where mass is a 7th state variable (fixes the fixed-outer-
step mass update issue in the original).

Third-body positions
--------------------
Without a JPL SPICE kernel available, mock third-body functions using
circular/elliptic Keplerian propagation are provided.  If ``spiceypy`` is
installed and kernels are loaded, pass ``get_third_body`` directly to the
propagators.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

from .constants import (
    MU_EARTH, MU_MOON, MU_SUN, R_EARTH, R_MOON,
    J2_EARTH, J2_MOON, G, OMEGA_MOON,
    R_SOI_MOON,
)
from .orbital_elements import coe2mee, mee2coe, mee2eci, eci2coe
from .control import ControlWeights
from .equations_of_motion import meeeqm_earth, meeeqm_moon


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PropagationResult:
    """Time-history of the propagated trajectory."""
    t:            np.ndarray          # time [s]
    mee:          np.ndarray          # (N, 6) MEE state
    mass:         np.ndarray          # (N,) mass [kg]
    coe:          np.ndarray          # (N, 6) classical elements
    r_eci:        np.ndarray          # (N, 3) ECI position [m]
    v_eci:        np.ndarray          # (N, 3) ECI velocity [m/s]
    delta_v:      float = 0.0         # total ΔV [m/s]
    m_prop:       float = 0.0         # propellant consumed [kg]
    t_transfer:   float = 0.0         # transfer time [s]
    converged:    bool  = True


# ---------------------------------------------------------------------------
# Mock third-body ephemeris  (Keplerian circles — no SPICE required)
# ---------------------------------------------------------------------------

def make_keplerian_third_body(
    mu_central: float,
    r_orbit: float,
    inclination: float = 0.0,
    phase0: float = 0.0,
    omega_body: float | None = None,
) -> callable:
    """
    Return a function  f(t) → position_vector [m]  for a body in a circular
    Keplerian orbit around the central body.

    Parameters
    ----------
    mu_central  : gravitational parameter of central body [m³/s²].
    r_orbit     : orbital radius [m].
    inclination : orbital inclination [rad].
    phase0      : initial phase angle [rad].
    omega_body  : angular velocity [rad/s]; computed from Kepler if None.
    """
    if omega_body is None:
        omega_body = np.sqrt(mu_central / r_orbit**3)

    si = np.sin(inclination)
    ci = np.cos(inclination)

    def position(t: float) -> np.ndarray:
        phi = phase0 + omega_body * t
        return r_orbit * np.array([np.cos(phi), np.sin(phi) * ci, np.sin(phi) * si])

    return position


def make_earth_phase_third_body(t0_phase: float = 0.0):
    """
    Return get_third_body(t) → (r_moon_eci, r_sun_eci) for Earth-centred phase.

    Uses circular-orbit approximations for both Moon and Sun.
    Replace with SPICE-based function if higher accuracy is needed.
    """
    moon_pos = make_keplerian_third_body(
        mu_central=MU_EARTH,
        r_orbit=384_400_000.0,
        inclination=np.radians(5.145),
        phase0=t0_phase,
        omega_body=OMEGA_MOON,
    )
    sun_pos = make_keplerian_third_body(
        mu_central=MU_SUN,
        r_orbit=1.496e11,
        inclination=np.radians(23.45),
        phase0=t0_phase * 0.0,
        omega_body=np.sqrt(MU_SUN / (1.496e11)**3),
    )

    def get_third_body(t: float):
        return moon_pos(t), sun_pos(t)

    return get_third_body


def make_moon_phase_third_body(t0_phase: float = 0.0):
    """
    Return get_third_body(t) → (r_earth_moon_eci, r_sun_moon_eci) for Moon phase.

    Positions of Earth and Sun as seen from the Moon.
    """
    # Earth as seen from Moon (reverse of Moon-from-Earth)
    earth_from_moon = make_keplerian_third_body(
        mu_central=MU_MOON,
        r_orbit=384_400_000.0,
        inclination=np.radians(5.145),
        phase0=t0_phase + np.pi,
        omega_body=OMEGA_MOON,
    )
    sun_from_moon = make_keplerian_third_body(
        mu_central=MU_MOON,
        r_orbit=1.496e11,
        inclination=np.radians(23.45),
        phase0=0.0,
        omega_body=np.sqrt(MU_SUN / (1.496e11)**3),
    )

    def get_third_body(t: float):
        return earth_from_moon(t), sun_from_moon(t)

    return get_third_body


# ---------------------------------------------------------------------------
# Earth-phase propagator
# ---------------------------------------------------------------------------

def propagate_earth_phase(
    coe_initial: np.ndarray,
    coe_target: np.ndarray,
    mass0: float,
    n_thrusters: int,
    thrust_per_thruster: float,
    isp: float,
    enable: ControlWeights,
    *,
    Cd: float = 2.2,
    S_sl: float = 0.0,
    S_sp: float = 0.0,
    c_r: float = 1.8,
    enable_eclipse: bool = True,
    smart_mode: bool = True,
    max_days: float = 800.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    get_third_body: callable | None = None,
    soi_multiplier: float = 1.0,
) -> PropagationResult:
    """
    Propagate GTO → Moon SOI using the Lyapunov low-thrust guidance.

    Parameters
    ----------
    coe_initial   : (6,) initial classical orbital elements [m, -, rad…].
    coe_target    : (6,) target classical orbital elements (Moon mean orbit).
    mass0         : initial spacecraft mass [kg].
    n_thrusters   : number of electric thrusters.
    thrust_per_thruster : thrust per thruster [N].
    isp           : specific impulse [s].
    enable        : ControlWeights  which elements to control.
    (keywords)    : optional physical + numerical parameters.

    Returns
    -------
    PropagationResult
    """
    if get_third_body is None:
        get_third_body = make_earth_phase_third_body()

    mee0 = coe2mee(coe_initial)
    y0   = np.append(mee0, mass0)
    t_max = max_days * 86400.0

    # SOI exit event: spacecraft reaches Moon SOI when |r_sc - r_moon| < SOI_radius
    def soi_event(t, y):
        r_eci, _ = mee2eci(MU_EARTH, y[:6])
        r_moon, _ = get_third_body(t)
        return np.linalg.norm(r_eci - r_moon) - R_SOI_MOON * soi_multiplier
    soi_event.terminal  = True
    soi_event.direction = -1

    # Mass depletion guard
    def mass_event(t, y):
        return y[6] - 1.0
    mass_event.terminal  = True
    mass_event.direction = -1

    def ode(t, y):
        return meeeqm_earth(
            t, y,
            mu=MU_EARTH, mu_moon=MU_MOON, mu_sun=MU_SUN,
            j2=J2_EARTH, r_body=R_EARTH,
            coe_initial=coe_initial, coe_target=coe_target,
            enable=enable,
            n_thrusters=n_thrusters,
            thrust_per_thruster=thrust_per_thruster,
            isp=isp, g0=G,
            Cd=Cd, S_sl=S_sl, S_sp=S_sp, c_r=c_r,
            enable_eclipse=enable_eclipse,
            smart_mode=smart_mode,
            get_third_body=get_third_body,
        )

    sol = solve_ivp(
        ode,
        [0.0, t_max],
        y0,
        method="DOP853",
        events=[soi_event, mass_event],
        rtol=rtol,
        atol=atol,
        dense_output=False,
        max_step=3600.0,        # max 1-hour step
    )

    return _build_result(sol, MU_EARTH, coe_initial, n_thrusters,
                         thrust_per_thruster, isp)


# ---------------------------------------------------------------------------
# Moon-phase propagator
# ---------------------------------------------------------------------------

def propagate_moon_phase(
    coe_initial_moon: np.ndarray,
    coe_target_moon: np.ndarray,
    mass0: float,
    n_thrusters: int,
    thrust_per_thruster: float,
    isp: float,
    enable: ControlWeights,
    *,
    S_sp: float = 0.0,
    c_r: float = 1.8,
    max_days: float = 300.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    get_third_body: callable | None = None,
) -> PropagationResult:
    """
    Propagate Moon SOI entry → target lunar orbit (e.g. NHRO).
    """
    if get_third_body is None:
        get_third_body = make_moon_phase_third_body()

    mee0 = coe2mee(coe_initial_moon)
    y0   = np.append(mee0, mass0)
    t_max = max_days * 86400.0

    # Target reached when semi-major axis converges to target within 1%
    a_target = coe_target_moon[0]
    def target_event(t, y):
        coe = mee2coe(y[:6])
        return coe[0] - a_target
    target_event.terminal  = True
    target_event.direction = 1 if coe_initial_moon[0] < a_target else -1

    def mass_event(t, y):
        return y[6] - 1.0
    mass_event.terminal  = True
    mass_event.direction = -1

    def ode(t, y):
        return meeeqm_moon(
            t, y,
            mu=MU_MOON, mu_earth=MU_EARTH, mu_sun=MU_SUN,
            j2=J2_MOON, r_body=R_MOON,
            coe_initial=coe_initial_moon, coe_target=coe_target_moon,
            enable=enable,
            n_thrusters=n_thrusters,
            thrust_per_thruster=thrust_per_thruster,
            isp=isp, g0=G,
            S_sp=S_sp, c_r=c_r,
            get_third_body=get_third_body,
        )

    sol = solve_ivp(
        ode,
        [0.0, t_max],
        y0,
        method="DOP853",
        events=[target_event, mass_event],
        rtol=rtol,
        atol=atol,
        dense_output=False,
        max_step=600.0,
    )

    return _build_result(sol, MU_MOON, coe_initial_moon, n_thrusters,
                         thrust_per_thruster, isp)


# ---------------------------------------------------------------------------
# Helper: build PropagationResult from solve_ivp solution
# ---------------------------------------------------------------------------

def _build_result(
    sol,
    mu: float,
    coe_initial: np.ndarray,
    n_thrusters: int,
    thrust_per_thruster: float,
    isp: float,
) -> PropagationResult:
    t    = sol.t
    mee  = sol.y[:6, :].T
    mass = sol.y[6, :]

    N      = len(t)
    coe_h  = np.zeros((N, 6))
    r_h    = np.zeros((N, 3))
    v_h    = np.zeros((N, 3))

    for j in range(N):
        coe_h[j] = mee2coe(mee[j])
        r_h[j], v_h[j] = mee2eci(mu, mee[j])

    m_prop   = mass[0] - mass[-1]
    # Tsiolkovsky ΔV (from propellant actually consumed)
    dv = isp * G * np.log(mass[0] / max(mass[-1], 1.0)) if mass[-1] > 1.0 else 0.0

    return PropagationResult(
        t=t, mee=mee, mass=mass,
        coe=coe_h, r_eci=r_h, v_eci=v_h,
        delta_v=dv,
        m_prop=m_prop,
        t_transfer=t[-1] if len(t) > 0 else 0.0,
        converged=sol.success or len(sol.t_events[0]) > 0,
    )
