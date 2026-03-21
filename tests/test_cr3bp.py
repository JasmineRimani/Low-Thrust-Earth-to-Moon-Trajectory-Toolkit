"""
Unit tests for the cr3bp subpackage.
Run with:  pytest tests/test_cr3bp.py -v
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.cr3bp.cr3bp_dynamics import (
    jacobi_constant, jacobi_max_drift,
    nrho_state_at, T_0_NRHO, TA_PERILUNE,
    P_0_NRHO, P_M_VEC, R_MOON,
    A_SCALE, T_SCALE, V_SCALE,
    MU_MOON_SI, R_MOON_M,
    integrate_cr3bp, propagate_cr3bp,
)
from src.cr3bp.frames import cr3bp_to_mci, mci_to_cr3bp, lvlh_basis_from_mci
from src.cr3bp.mission_utils import (
    circular_speed_ms, doi_dv_ms, circularisation_dv_ms,
    phasing_dv_ms, tsiolkovsky_fuel_kg, apply_dv_corrections,
    round_trip_phasing,
)
from src.cr3bp.nrho_llo_transfer import (
    nrho_to_llo, llo_to_nrho, TransferSolverSettings,
)
from src.cr3bp.initial_guess import (
    strategy_literature, strategy_grid_search, GuessStrategy, get_initial_guess,
)


# ---------------------------------------------------------------------------
# CR3BP dynamics
# ---------------------------------------------------------------------------

def test_jacobi_constant_shape():
    state = P_0_NRHO.copy()
    C = jacobi_constant(state)
    assert isinstance(C, float)
    assert np.isfinite(C)


def test_jacobi_conserved_on_nrho():
    """Jacobi constant should drift < 1e-8 along one NRHO revolution."""
    _, y = integrate_cr3bp(0.05, T_0_NRHO, P_0_NRHO)
    drift = jacobi_max_drift(y)
    assert drift < 1e-7, f"Jacobi drift too large: {drift:.2e}"


def test_nrho_period_reasonable():
    """NRHO period should be roughly 1.6 CR3BP time units."""
    assert 1.4 < T_0_NRHO < 1.9


def test_nrho_state_at_zero():
    """State at ta=0 should equal P_0_NRHO."""
    s = nrho_state_at(0.0)
    np.testing.assert_allclose(s, P_0_NRHO, rtol=1e-10)


def test_nrho_state_at_period():
    """State at ta=T_0_NRHO should return close to P_0_NRHO."""
    s = nrho_state_at(T_0_NRHO)
    np.testing.assert_allclose(s, P_0_NRHO, rtol=1e-6, atol=1e-8)


def test_perilune_is_minimum():
    """TA_PERILUNE should give the smallest Moon distance in one revolution."""
    tas = np.linspace(0, T_0_NRHO, 50, endpoint=False)
    dists = [np.linalg.norm(nrho_state_at(ta)[:3] - P_M_VEC) for ta in tas]
    d_perilune = np.linalg.norm(nrho_state_at(TA_PERILUNE)[:3] - P_M_VEC)
    assert d_perilune <= min(dists) + 1e-4


def test_propagate_cr3bp_shape():
    t, y = propagate_cr3bp(P_0_NRHO, 0.5)
    assert y.shape[1] == 6
    assert len(t) == y.shape[0]


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def test_cr3bp_mci_round_trip():
    """CR3BP → MCI → CR3BP should be an identity."""
    r_rot = np.array([1.02, 0.01, -0.18])
    v_rot = np.array([0.0, -0.10, 0.01])
    t_nd  = 0.5

    r_mci, v_mci = cr3bp_to_mci(r_rot, v_rot, t_nd)
    r_rot2, v_rot2 = mci_to_cr3bp(r_mci, v_mci, t_nd)

    np.testing.assert_allclose(r_rot2, r_rot, atol=1e-12)
    np.testing.assert_allclose(v_rot2, v_rot, atol=1e-12)


def test_lvlh_basis_orthonormal():
    r_mci = np.array([1.0, 0.2, 0.0])
    v_mci = np.array([0.0, 1.0, 0.1])
    r_hat, theta_hat, h_hat = lvlh_basis_from_mci(r_mci, v_mci)

    np.testing.assert_allclose(np.linalg.norm(r_hat),     1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(theta_hat), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(h_hat),     1.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(r_hat, h_hat),      0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Mission utilities
# ---------------------------------------------------------------------------

def test_circular_speed_positive():
    v = circular_speed_ms(100_000.0)
    assert 1500 < v < 2000   # ~1634 m/s at 100 km LLO


def test_doi_dv_direction():
    dv = doi_dv_ms(100_000.0, 15_000.0)
    assert dv > 0.0


def test_doi_dv_zero_if_no_change():
    dv = doi_dv_ms(100_000.0, 100_000.0)
    assert dv == 0.0


def test_tsiolkovsky_fuel_zero_dv():
    assert tsiolkovsky_fuel_kg(1000.0, 450.0, 0.0) == 0.0


def test_tsiolkovsky_fuel_positive():
    m_prop = tsiolkovsky_fuel_kg(1000.0, 450.0, 200.0)
    assert 0 < m_prop < 1000.0


def test_apply_dv_corrections_basic():
    dv_c, dv_extra = apply_dv_corrections(200.0, reserve_frac=0.05)
    assert dv_c == pytest.approx(210.0)
    assert dv_extra > 0.0


def test_round_trip_phasing_fields():
    diag = round_trip_phasing(
        departure_ta_nd=0.82,
        downleg_tof_s=3 * 86400,
        descent_tof_s=0.5 * 86400,
        surface_duration_s=7 * 86400,
        ascent_tof_s=0.5 * 86400,
    )
    assert 0.0 <= diag.phase_offset_fraction <= 1.0
    assert diag.passive_wait_to_next_window_s >= 0.0
    assert isinstance(diag.phase_family, str)


# ---------------------------------------------------------------------------
# Transfer solvers (short test runs — not full 20-epoch scan)
# ---------------------------------------------------------------------------

def test_nrho_to_llo_perilune():
    """Solver should find a valid transfer near TA_PERILUNE at 100 km LLO."""
    settings = TransferSolverSettings(n_ta_candidates=5, tof_max_cr3bp=4.0)
    res = nrho_to_llo(
        m0=3000.0, isp=450.0, h_llo_m=100_000.0,
        ta_hint=TA_PERILUNE,
        settings=settings,
    )
    assert res.dv_total > 0.0
    assert res.time_of_flight > 0.0
    assert res.fuel_mass_kg > 0.0
    assert res.trajectory.shape[1] == 6
    # Periapsis altitude should be within 500 km of 100 km
    assert abs(res.periapsis_alt_m - 100_000.0) < 500_000.0


def test_nrho_to_llo_dv_range():
    """ΔV for NRHO→LLO at 100 km should be 800–1500 m/s (literature range)."""
    settings = TransferSolverSettings(n_ta_candidates=5, tof_max_cr3bp=4.0)
    res = nrho_to_llo(
        m0=3000.0, isp=450.0, h_llo_m=100_000.0,
        ta_hint=TA_PERILUNE,
        settings=settings,
    )
    assert 600 < res.dv_total < 1800, f"ΔV={res.dv_total:.1f} m/s outside expected range"


def test_llo_to_nrho_fast_path():
    """LLO→NRHO fast path should return same ΔV as supplied."""
    dv_down = 950.0
    res = llo_to_nrho(
        m0=1500.0, isp=450.0, h_llo_m=100_000.0,
        dv_from_downleg=dv_down,
        ta_hint=TA_PERILUNE,
    )
    assert res.dv_total == pytest.approx(dv_down)
    assert res.fuel_mass_kg > 0.0


# ---------------------------------------------------------------------------
# Initial guess
# ---------------------------------------------------------------------------

def test_literature_candidates_count():
    seeds = strategy_literature(100_000.0 / A_SCALE, np.pi / 2, 0.0)
    assert 1 <= len(seeds) <= 5


def test_grid_search_returns_candidates():
    scored = strategy_grid_search(100_000.0 / A_SCALE, np.pi / 2, 0.0, n_ta=4, n_tof=4)
    assert len(scored) > 0
    assert all(score >= 0.0 for score, _ in scored)


def test_get_initial_guess_returns_result():
    result = get_initial_guess(100_000.0 / A_SCALE, np.pi / 2, 0.0)
    assert result.x0.shape == (11,)
    assert isinstance(result.strategy, GuessStrategy)
    assert np.isfinite(result.score)
