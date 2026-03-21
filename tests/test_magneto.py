"""
Unit tests for the core low-thrust mission-analysis library.
Run with:  pytest tests/test_magneto.py -v
"""

import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.constants import MU_EARTH, MU_MOON, R_EARTH, R_MOON, G
from src.orbital_elements import (
    coe2eci, eci2coe, coe2mee, mee2coe, mee2eci, eci2mee, lvlh_rotation
)
from src.perturbations import (
    j2_acceleration, third_body_acceleration, eclipse_conical,
    srp_acceleration,
)
from src.control import ControlWeights, thrust_direction_lvlh
from src.plotting import save_orbital_history_plot, save_trajectory_views
from src.propagator import (
    propagate_earth_phase, propagate_moon_phase,
    make_earth_phase_third_body, make_moon_phase_third_body,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_PLOTS_DIR = REPO_ROOT / "tmp_plots"
TMP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---- Fixtures -------------------------------------------------------------

def gto_coe():
    """Fictitious GTO-like orbit."""
    return np.array([
        24_500e3 + R_EARTH,  # a [m]
        0.71,                 # e
        np.radians(7.0),      # i [rad]
        np.radians(0.0),      # omega
        np.radians(0.0),      # RAAN
        np.radians(0.0),      # nu
    ])


def moon_orbit_coe():
    """Approximate Moon mean orbit."""
    return np.array([
        384_400e3,
        0.055,
        np.radians(5.14),
        0.0, 0.0, 0.0,
    ])


# ---- Orbital element conversions ------------------------------------------

def test_coe_eci_round_trip():
    coe = gto_coe()
    r, v = coe2eci(MU_EARTH, coe)
    coe2 = eci2coe(MU_EARTH, r, v)
    np.testing.assert_allclose(coe, coe2, rtol=1e-9, atol=1e-3)


def test_coe_mee_round_trip():
    coe = gto_coe()
    mee = coe2mee(coe)
    coe2 = mee2coe(mee)
    np.testing.assert_allclose(coe, coe2, rtol=1e-9, atol=1e-6)


def test_mee_eci_round_trip():
    coe  = gto_coe()
    mee  = coe2mee(coe)
    r, v = mee2eci(MU_EARTH, mee)
    mee2 = eci2mee(MU_EARTH, r, v)
    np.testing.assert_allclose(mee[:5], mee2[:5], rtol=1e-8)


def test_lvlh_rotation_orthogonal():
    coe = gto_coe()
    r, v = coe2eci(MU_EARTH, coe)
    Q = lvlh_rotation(r, v)
    # Q should be orthogonal: Q @ Q.T ≈ I
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-12)


# ---- Perturbations --------------------------------------------------------

def test_j2_not_zero_at_equator():
    coe = gto_coe()
    r, v = coe2eci(MU_EARTH, coe)
    Q = lvlh_rotation(r, v)
    dr, dt, dn = j2_acceleration(r, MU_EARTH, 1.08263e-3, R_EARTH, Q)
    # J2 should not be zero at 7-degree inclination
    assert abs(dr) + abs(dt) + abs(dn) > 0.0


def test_third_body_direction_matters():
    """Verify the fix: using +s_vec vs -s_vec gives different results."""
    r_sc  = np.array([10_000e3, 0.0, 0.0])
    v_sc  = np.array([0.0, 7000.0, 0.0])
    Q     = lvlh_rotation(r_sc, v_sc)
    s_pos = np.array([384_400e3, 0.0, 0.0])  # Moon to the right

    dr1, dt1, dn1 = third_body_acceleration(r_sc, s_pos,  MU_MOON, Q)
    dr2, dt2, dn2 = third_body_acceleration(r_sc, -s_pos, MU_MOON, Q)
    # The two should NOT be identical — direction matters
    assert not np.allclose([dr1, dt1, dn1], [dr2, dt2, dn2])


def test_eclipse_behind_earth():
    """Spacecraft directly behind Earth should be in eclipse."""
    r_sun = np.array([1.5e11, 0.0, 0.0])
    r_sc  = np.array([-8000e3, 0.0, 0.0])    # directly anti-Sun, low orbit
    assert eclipse_conical(r_sc, r_sun, R_EARTH)


def test_no_eclipse_in_sunlight():
    """Spacecraft on the Sun-side should NOT be in eclipse."""
    r_sun = np.array([1.5e11, 0.0, 0.0])
    r_sc  = np.array([8000e3, 0.0, 0.0])     # Sun-facing side
    assert not eclipse_conical(r_sc, r_sun, R_EARTH)


def test_srp_zero_in_eclipse():
    r_sun = np.array([1.5e11, 0.0, 0.0])
    r_sc  = np.array([8000e3, 0.0, 0.0])
    Q = np.eye(3)
    dr, dt, dn = srp_acceleration(r_sc, r_sun, 500.0, 20.0, 5.0, 1.8, Q,
                                   in_eclipse=True)
    assert dr == 0.0 and dt == 0.0 and dn == 0.0


def test_srp_nonzero_in_sunlight():
    r_sun = np.array([1.5e11, 0.0, 0.0])
    r_sc  = np.array([8000e3, 0.0, 0.0])
    Q = np.eye(3)
    dr, dt, dn = srp_acceleration(r_sc, r_sun, 500.0, 20.0, 5.0, 1.8, Q,
                                   in_eclipse=False)
    assert abs(dr) + abs(dt) + abs(dn) > 0.0


# ---- Control law ----------------------------------------------------------

def test_thrust_unit_vector():
    coe = gto_coe()
    enable = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)
    ur, ut, un = thrust_direction_lvlh(coe, coe, moon_orbit_coe(), enable)
    norm = np.sqrt(ur**2 + ut**2 + un**2)
    # Should be a unit vector (or zero if all weights are zero)
    assert norm < 1.0 + 1e-9


def test_thrust_zero_when_all_weights_zero():
    coe = gto_coe()
    enable = ControlWeights(0.0, 0.0, 0.0, 0.0, 0.0)
    ur, ut, un = thrust_direction_lvlh(coe, coe, moon_orbit_coe(), enable)
    assert ur == 0.0 and ut == 0.0 and un == 0.0


# ---- Propagator (short runs for speed) -----------------------------------

def test_earth_phase_runs_and_conserves_mass():
    coe_i = gto_coe()
    coe_f = moon_orbit_coe()
    enable = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=0.0, kraan=0.0)

    res = propagate_earth_phase(
        coe_i, coe_f, 500.0,
        n_thrusters=4, thrust_per_thruster=0.01, isp=1600.0,
        enable=enable,
        enable_eclipse=False,
        smart_mode=False,
        max_days=5.0,          # short run for test speed
        rtol=1e-6, atol=1e-8,
    )
    # Mass should be monotonically decreasing
    assert np.all(np.diff(res.mass) <= 1e-6)
    assert res.mass[-1] < res.mass[0]


def test_earth_phase_output_shapes():
    coe_i = gto_coe()
    coe_f = moon_orbit_coe()
    enable = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=0.0, kraan=0.0)
    third_body = make_earth_phase_third_body()

    res = propagate_earth_phase(
        coe_i, coe_f, 500.0,
        n_thrusters=2, thrust_per_thruster=0.01, isp=1600.0,
        enable=enable,
        enable_eclipse=False,
        smart_mode=False,
        max_days=3.0,
        rtol=1e-6, atol=1e-8,
        get_third_body=third_body,
    )
    N = len(res.t)
    assert res.mee.shape  == (N, 6)
    assert res.coe.shape  == (N, 6)
    assert res.r_eci.shape == (N, 3)
    assert res.mass.shape  == (N,)

    history_path = TMP_PLOTS_DIR / "test_earth_phase_history.png"
    trajectory_path = TMP_PLOTS_DIR / "test_earth_phase_trajectory.png"
    moon_track = np.array([third_body(t)[0] for t in res.t])

    save_orbital_history_plot(
        t_days=res.t / 86400.0,
        coe=res.coe,
        mass=res.mass,
        save_path=history_path,
        title="Test Earth-phase history (EP)",
    )
    save_trajectory_views(
        trajectory=res.r_eci,
        reference_trajectory=moon_track,
        central_body_radius=R_EARTH,
        save_path=trajectory_path,
        title="Test Earth-phase trajectory (EP)",
        axis_unit_label="10^3 km",
        scale=1e6,
        trajectory_label="Transfer trajectory",
        reference_label="Moon ephemeris",
        body_label="Earth",
        body_color="steelblue",
        end_label="Test end state",
    )

    assert history_path.is_file() and history_path.stat().st_size > 0
    assert trajectory_path.is_file() and trajectory_path.stat().st_size > 0


def test_moon_phase_runs():
    coe_i = np.array([60_000e3, 0.92, np.radians(10.0), 0.0, 0.0, 0.0])
    coe_f = np.array([5_000e3 + R_MOON, 0.0, np.radians(90.0),
                       np.radians(270.0), np.radians(90.0), 0.0])
    enable = ControlWeights(ka=1.0, ke=1.0, ki=0.0, kw=0.0, kraan=0.0)

    res = propagate_moon_phase(
        coe_i, coe_f, 200.0,
        n_thrusters=4, thrust_per_thruster=0.01, isp=1600.0,
        enable=enable,
        max_days=5.0,
        rtol=1e-6, atol=1e-8,
    )
    assert len(res.t) > 1
    assert res.mass[-1] < res.mass[0]
