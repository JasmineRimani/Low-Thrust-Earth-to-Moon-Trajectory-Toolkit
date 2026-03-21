"""
NRHO ↔ LLO transfer ΔV estimator for preliminary analysis.

NRHO → LLO
-----------
Retrograde kick at NRHO departure epoch + circularisation burn at periapsis.
Uses Brent's method to find the unique dv1 placing periapsis at the target LLO
radius.  Scans departure epochs over one NRHO revolution and returns the minimum
total ΔV solution.

The ΔV values are *trajectory-geometry* quantities (independent of propulsion
type). Propellant mass is computed via the Tsiolkovsky equation using the
caller-supplied Isp.

LLO → NRHO  (return leg)
------------------------
The CR3BP equations of motion are time-reversible for free coast arcs.
For preliminary analysis the total ΔV magnitude is symmetric:
    ΔV(LLO→NRHO) ≈ ΔV(NRHO→LLO)
The return leg reuses the down-leg periapsis scan.

References
----------
Parker & Anderson (2014): Low-Energy Lunar Trajectory Design, Ch. 4.
Zimovan-Spreen et al. (2020): Near Rectilinear Halo Orbits, J. Astronautical Sci.
Capdevila et al. (2014): Transfer Network Linking Earth, Moon and L4/L5.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import brentq

from .cr3bp_dynamics import (
    MU_M, T_SCALE, A_SCALE, V_SCALE, R_MOON,
    MU_MOON_SI, R_MOON_M,
    P_M_VEC, T_0_NRHO, TA_PERILUNE,
    DT_TRAJ, DT_COARSE,
    nrho_state_at, propagate_cr3bp,
)
from .llo_state import llo_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solver settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransferSolverSettings:
    """
    Numerical parameters for the NRHO ↔ LLO transfer scan.

    All defaults are calibrated for a 9:2 NRHO → polar LLO transfer
    at roughly 100 km altitude.
    """
    # Departure-epoch scan
    n_ta_candidates:       int   = 20      # departure epochs to scan per revolution
    ta_window_half_width:  float = 0.15    # local window half-width [CR3BP time]

    # Brent bracket for departure kick magnitude
    dv1_low_ms:            float = 10.0    # [m/s]  lower bracket
    dv1_high_ms:           float = 400.0   # [m/s]  upper bracket

    # Transfer arc
    tof_max_cr3bp:         float = 6.0     # max propagation time [CR3BP time]

    # Selection: ΔV + (optional) time-of-flight trade
    tof_weight_s_per_ms:   float = 0.0     # [s per m/s] — 0 = pure ΔV
    tof_cap_days:          float | None = None

    # Periapsis detection tolerance
    first_pass_match_tol_m: float = 500_000.0   # [m] ±500 km

    # Fast-path placeholders (time-reversal mode)
    fast_path_tof_fraction_of_tof_max: float = 0.5
    fast_path_dv_split: tuple[float, float] = (0.66, 0.34)

    # Warning
    near_polar_warning_band_deg: float = 30.0


# ---------------------------------------------------------------------------
# Transfer result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    """Output of a NRHO ↔ LLO transfer ΔV computation."""
    dv_total:         float                   # total ΔV [m/s]
    dv1:              np.ndarray              # departure burn vector [m/s]
    dv2:              np.ndarray              # insertion burn vector [m/s]
    time_of_flight:   float                   # coast arc time [s]
    trajectory:       np.ndarray              # (N, 6) CR3BP states
    departure_ta:     float                   # departure epoch [CR3BP time]
    periapsis_alt_m:  float                   # periapsis altitude [m]
    fuel_mass_kg:     float = 0.0             # propellant used [kg] (Tsiolkovsky)
    wait_time_s:      float = 0.0             # optional LLO wait for phasing


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _periapsis_radius_m(
    dv1_ms: float,
    p_dep: np.ndarray,
    settings: TransferSolverSettings,
) -> float:
    """
    Propagate state with retrograde kick dv1_ms and return first periapsis
    distance from Moon centre [m].
    """
    v_hat   = p_dep[3:] / np.linalg.norm(p_dep[3:])
    p0      = p_dep.copy()
    p0[3:] -= (dv1_ms / V_SCALE) * v_hat

    t_arr, y = propagate_cr3bp(p0, float(settings.tof_max_cr3bp))
    dists    = np.linalg.norm(y[:, :3] - P_M_VEC, axis=1) * A_SCALE
    return float(dists[_first_local_minimum_idx(dists)])


def _first_local_minimum_idx(arr: np.ndarray) -> int:
    """Index of the first local minimum in arr; falls back to global minimum."""
    if arr.size < 3:
        return int(np.argmin(arr))
    mids    = np.arange(1, arr.size - 1)
    minima  = mids[(arr[1:-1] <= arr[:-2]) & (arr[1:-1] <= arr[2:])]
    return int(minima[0]) if minima.size > 0 else int(np.argmin(arr))


def _select_arrival_idx(
    dists_m: np.ndarray,
    r_tgt_m: float,
    settings: TransferSolverSettings,
) -> int:
    """Pick the arrival pass closest to the target LLO radius."""
    first = _first_local_minimum_idx(dists_m)
    tol   = float(settings.first_pass_match_tol_m)
    if abs(float(dists_m[first]) - r_tgt_m) <= tol:
        return first
    # Search all local minima
    if dists_m.size >= 3:
        mids   = np.arange(1, dists_m.size - 1)
        minima = mids[(dists_m[1:-1] <= dists_m[:-2]) & (dists_m[1:-1] <= dists_m[2:])]
        matched = minima[np.abs(dists_m[minima] - r_tgt_m) <= tol]
        if matched.size > 0:
            return int(matched[0])
    return first


def _solve_one_ta(
    ta: float,
    r_tgt_m: float,
    settings: TransferSolverSettings,
) -> tuple[float, float, float, float, float] | None:
    """
    Two-impulse transfer for one departure epoch.

    Returns (dv_total_ms, dv1_ms, dv2_ms, r_peri_km, tof_s) or None.
    """
    p_dep = nrho_state_at(ta)
    try:
        r_lo = _periapsis_radius_m(float(settings.dv1_low_ms),  p_dep, settings)
        r_hi = _periapsis_radius_m(float(settings.dv1_high_ms), p_dep, settings)

        if not (r_hi < r_tgt_m < r_lo):
            return None

        dv1 = brentq(
            lambda d: _periapsis_radius_m(d, p_dep, settings) - r_tgt_m,
            float(settings.dv1_low_ms),
            float(settings.dv1_high_ms),
            xtol=5.0, maxiter=15,
        )

        v_hat  = p_dep[3:] / np.linalg.norm(p_dep[3:])
        p0     = p_dep.copy()
        p0[3:] -= (dv1 / V_SCALE) * v_hat

        t_arr, y = propagate_cr3bp(p0, float(settings.tof_max_cr3bp))
        dists    = np.linalg.norm(y[:, :3] - P_M_VEC, axis=1) * A_SCALE
        i_arr    = _select_arrival_idx(dists, r_tgt_m, settings)

        r_peri  = float(dists[i_arr])
        v_peri  = float(np.linalg.norm(y[i_arr, 3:])) * V_SCALE
        v_circ  = float(np.sqrt(MU_MOON_SI / r_peri))
        dv2     = abs(v_peri - v_circ)
        tof_s   = float(t_arr[i_arr]) * T_SCALE

        return (dv1 + dv2, dv1, dv2, r_peri / 1e3, tof_s)

    except RuntimeError:
        logger.debug(f"_solve_one_ta: propagation failed at TA={ta:.3f}")
        return None


def _best_candidate(
    candidates: list,
    settings: TransferSolverSettings,
) -> tuple:
    """Select minimum-ΔV candidate (with optional TOF cap and weighting)."""
    pool = list(candidates)
    if settings.tof_cap_days is not None:
        cap_s  = float(settings.tof_cap_days) * 86400.0
        capped = [c for c in pool if float(c[4]) <= cap_s]
        if capped:
            pool = capped

    def score(c):
        return float(c[0]) + float(settings.tof_weight_s_per_ms) * float(c[4])

    return min(pool, key=score)


# ---------------------------------------------------------------------------
# Public API — NRHO → LLO
# ---------------------------------------------------------------------------

def nrho_to_llo(
    m0: float,
    isp: float,
    h_llo_m: float,
    *,
    inc_llo: float = np.pi / 2,
    raan_llo: float = 0.0,
    ta_hint: float | None = None,
    settings: TransferSolverSettings | None = None,
    g0: float = 9.80665,
) -> TransferResult:
    """
    Compute NRHO → LLO transfer ΔV budget for preliminary analysis.

    The ΔV is a trajectory-geometry quantity.  Propellant mass is computed
    via Tsiolkovsky using the supplied **electric-propulsion Isp**.

    Parameters
    ----------
    m0       : spacecraft mass at NRHO departure [kg].
    isp      : electric propulsion specific impulse [s].
    h_llo_m  : target circular LLO altitude [m].
    inc_llo  : LLO inclination [rad].  Default: polar (π/2).
    raan_llo : LLO RAAN [rad].
    ta_hint  : preferred departure epoch [CR3BP time].
    settings : solver settings.  None → defaults.
    g0       : standard gravity [m/s²].

    Returns
    -------
    TransferResult
    """
    if settings is None:
        settings = TransferSolverSettings()

    r_tgt_m  = R_MOON_M + float(h_llo_m)

    if abs(float(inc_llo) - np.pi / 2) > np.radians(float(settings.near_polar_warning_band_deg)):
        logger.warning(
            f"nrho_to_llo: inclination {np.degrees(float(inc_llo)):.1f}° is far from "
            "polar. Solver is calibrated for near-polar (south-pole access) LLO."
        )

    # Build TA scan grid
    ta_grid = np.linspace(0.0, T_0_NRHO, int(settings.n_ta_candidates), endpoint=False)
    if ta_hint is not None:
        ta_grid = np.concatenate([[float(ta_hint) % T_0_NRHO], ta_grid])

    logger.info(
        f"nrho_to_llo: scanning {len(ta_grid)} departure epochs  "
        f"[h_LLO={h_llo_m/1e3:.0f} km, m0={m0:.0f} kg, Isp={isp:.0f} s]"
    )

    candidates = []
    for ta in ta_grid:
        res = _solve_one_ta(ta, r_tgt_m, settings)
        if res is not None:
            dvt, dv1, dv2, r_km, tof_s = res
            candidates.append((dvt, dv1, dv2, r_km, tof_s, ta))

    if not candidates:
        raise RuntimeError(
            f"nrho_to_llo: no valid transfer found for h_llo={h_llo_m/1e3:.0f} km. "
            "Try widening dv1_low_ms / dv1_high_ms in TransferSolverSettings."
        )

    dv_total, dv1_ms, dv2_ms, r_peri_km, tof_s, ta_best = _best_candidate(
        candidates, settings
    )

    # Reconstruct trajectory
    p_dep  = nrho_state_at(ta_best)
    v_hat  = p_dep[3:] / np.linalg.norm(p_dep[3:])
    p0     = p_dep.copy()
    p0[3:] -= (dv1_ms / V_SCALE) * v_hat
    t_arr, y_traj = propagate_cr3bp(p0, float(settings.tof_max_cr3bp))

    dists   = np.linalg.norm(y_traj[:, :3] - P_M_VEC, axis=1) * A_SCALE
    i_peri  = _select_arrival_idx(dists, r_tgt_m, settings)
    traj    = y_traj[:i_peri + 1]

    # Burn vectors
    dv1_vec = -dv1_ms * v_hat                                       # retrograde departure
    v_arr   = y_traj[i_peri, 3:]
    dv2_vec =  dv2_ms * v_arr / (np.linalg.norm(v_arr) + 1e-30)   # prograde insertion

    # Tsiolkovsky fuel
    fuel_kg = float(m0) * (1.0 - np.exp(-dv_total / (float(isp) * g0)))

    alt_m = r_peri_km * 1e3 - R_MOON_M

    logger.info(
        f"nrho_to_llo: ΔV={dv_total:.1f} m/s  "
        f"tof={tof_s/3600:.1f} h  alt={alt_m/1e3:.0f} km  fuel={fuel_kg:.1f} kg"
    )

    return TransferResult(
        dv_total=dv_total,
        dv1=dv1_vec,
        dv2=dv2_vec,
        time_of_flight=tof_s,
        trajectory=traj,
        departure_ta=ta_best,
        periapsis_alt_m=alt_m,
        fuel_mass_kg=fuel_kg,
    )


# ---------------------------------------------------------------------------
# Public API — LLO → NRHO  (time reversal)
# ---------------------------------------------------------------------------

def llo_to_nrho(
    m0: float,
    isp: float,
    h_llo_m: float,
    *,
    inc_llo: float = np.pi / 2,
    raan_llo: float = 0.0,
    ta_hint: float | None = None,
    dv_from_downleg: float | None = None,
    tof_from_downleg: float | None = None,
    settings: TransferSolverSettings | None = None,
    g0: float = 9.80665,
) -> TransferResult:
    """
    Compute LLO → NRHO transfer ΔV budget for preliminary analysis.

    By CR3BP time-reversibility ΔV(LLO→NRHO) ≈ ΔV(NRHO→LLO).
    Propellant mass is computed via Tsiolkovsky using the electric-propulsion Isp.

    Two modes
    ---------
    1. **Fast path**: supply ``dv_from_downleg`` from a prior NRHO→LLO call.
       No additional integration needed.
    2. **Independent scan**: ``dv_from_downleg=None`` runs the same periapsis
       scan as the down-leg.

    Parameters
    ----------
    m0, isp, h_llo_m : same as ``nrho_to_llo``.  isp = EP specific impulse [s].
    dv_from_downleg  : down-leg total ΔV [m/s].  Used in fast-path mode.
    tof_from_downleg : down-leg ToF [s].  Used in fast-path mode.
    """
    if settings is None:
        settings = TransferSolverSettings()

    if dv_from_downleg is not None:
        # Fast path — time reversal symmetry
        dv_total = float(dv_from_downleg)
        tof_s    = (float(tof_from_downleg) if tof_from_downleg is not None
                    else float(settings.tof_max_cr3bp) * T_SCALE
                       * float(settings.fast_path_tof_fraction_of_tof_max))
        dv1_ms = dv_total * settings.fast_path_dv_split[0]
        dv2_ms = dv_total * settings.fast_path_dv_split[1]

        ta_dep = float(ta_hint) if ta_hint is not None else TA_PERILUNE
        p_llo  = nrho_state_at(ta_dep)   # approximate LLO departure direction
        v_hat  = p_llo[3:] / (np.linalg.norm(p_llo[3:]) + 1e-30)

        fuel_kg = float(m0) * (1.0 - np.exp(-dv_total / (float(isp) * g0)))

        logger.info(
            f"llo_to_nrho (fast path / time-reversal): ΔV={dv_total:.1f} m/s  "
            f"tof={tof_s/3600:.1f} h  fuel={fuel_kg:.1f} kg"
        )

        return TransferResult(
            dv_total=dv_total,
            dv1=dv1_ms * v_hat,
            dv2=dv2_ms * v_hat,
            time_of_flight=tof_s,
            trajectory=np.zeros((2, 6)),   # not reconstructed in fast path
            departure_ta=ta_dep,
            periapsis_alt_m=float(h_llo_m),
            fuel_mass_kg=fuel_kg,
        )

    # Independent scan — identical algorithm to down-leg
    logger.info("llo_to_nrho: running independent periapsis scan (time-reversal disabled).")
    return nrho_to_llo(
        m0, isp, h_llo_m,
        inc_llo=inc_llo,
        raan_llo=raan_llo,
        ta_hint=ta_hint,
        settings=settings,
        g0=g0,
    )
