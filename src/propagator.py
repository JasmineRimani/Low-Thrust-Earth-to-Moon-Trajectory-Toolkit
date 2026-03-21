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
from dataclasses import dataclass
from scipy.integrate import solve_ivp

from .constants import (
    MU_EARTH, MU_MOON, MU_SUN, R_EARTH, R_MOON,
    J2_EARTH, J2_MOON, G, OMEGA_MOON,
    R_SOI_MOON,
)
from .orbital_elements import coe2mee, mee2coe, mee2eci
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
    target_reached: bool = False      # target event reached
    stop_reason:   str   = "completed"


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

    return _build_result(
        sol,
        MU_EARTH,
        coe_initial,
        n_thrusters,
        thrust_per_thruster,
        isp,
        event_names=["moon_soi_reached", "mass_depleted"],
        target_event_index=0,
    )


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

    target_rp = coe_target_moon[0] * (1.0 - coe_target_moon[1])
    target_ra = coe_target_moon[0] * (1.0 + coe_target_moon[1])
    target_inc = coe_target_moon[2]

    def target_event(t, y):
        coe = mee2coe(y[:6])
        a, e, inc = coe[0], coe[1], coe[2]
        if a <= 0.0 or e >= 0.999:
            return 1.0

        rp = a * (1.0 - e)
        ra = a * (1.0 + e)
        inc_err = abs(((inc - target_inc + np.pi) % (2.0 * np.pi)) - np.pi)

        # Use periapsis/apoapsis closeness rather than semi-major axis alone so
        # the stop event corresponds to an actual near-target lunar orbit.
        rp_err = abs(rp - target_rp) / max(target_rp, 1.0)
        ra_err = abs(ra - target_ra) / max(target_ra, 1.0)
        inc_metric = inc_err / np.radians(5.0)
        return max(rp_err / 0.03, ra_err / 0.03, inc_metric) - 1.0
    target_event.terminal  = True
    target_event.direction = -1

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

    return _build_result(
        sol,
        MU_MOON,
        coe_initial_moon,
        n_thrusters,
        thrust_per_thruster,
        isp,
        event_names=["target_orbit_reached", "mass_depleted"],
        target_event_index=0,
    )


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
    *,
    event_names: list[str] | None = None,
    target_event_index: int = 0,
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

    if event_names is None:
        event_names = []

    target_reached = (
        target_event_index < len(sol.t_events)
        and len(sol.t_events[target_event_index]) > 0
    )
    stop_reason = "integration_failed"
    if target_reached:
        stop_reason = event_names[target_event_index] if target_event_index < len(event_names) else "target_reached"
    elif any(len(events) > 0 for events in sol.t_events):
        for index, events in enumerate(sol.t_events):
            if len(events) > 0:
                stop_reason = event_names[index] if index < len(event_names) else f"event_{index}"
                break
    elif sol.status == 0:
        stop_reason = "time_limit_reached"

    return PropagationResult(
        t=t, mee=mee, mass=mass,
        coe=coe_h, r_eci=r_h, v_eci=v_h,
        delta_v=dv,
        m_prop=m_prop,
        t_transfer=t[-1] if len(t) > 0 else 0.0,
        converged=target_reached,
        target_reached=target_reached,
        stop_reason=stop_reason,
    )
