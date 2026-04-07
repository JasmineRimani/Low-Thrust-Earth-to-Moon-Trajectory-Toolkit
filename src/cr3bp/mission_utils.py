"""
Post-transfer ΔV corrections and mission-level utilities.

Covers:
- DOI (descent orbit insertion) burn estimation
- Circularisation ΔV from an elliptic ascent orbit
- Small-angle phasing estimate
- 9:2 NRHO mission-clock phasing diagnostics
- Propellant mass (Tsiolkovsky)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .cr3bp_dynamics import MU_MOON_SI, R_MOON_M, T_SCALE, T_0_NRHO

G0: float = 9.80665   # standard gravity [m/s²]

# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------

def circular_speed_ms(h_m: float) -> float:
    """Circular orbital speed [m/s] at altitude h_m [m] above Moon surface."""
    r = R_MOON_M + float(h_m)
    if r <= 0.0:
        raise ValueError("Altitude must give positive orbital radius.")
    return float(math.sqrt(MU_MOON_SI / r))


def doi_dv_ms(h_llo_m: float, h_periapsis_m: float) -> float:
    """
    Descent-orbit-insertion (DOI) ΔV [m/s].

    Impulsive burn at apolune of a transfer ellipse from circular LLO to an
    ellipse with periapsis at h_periapsis_m.
    """
    r_a = R_MOON_M + float(h_llo_m)
    r_p = R_MOON_M + float(h_periapsis_m)
    if r_p >= r_a:
        return 0.0
    a      = 0.5 * (r_a + r_p)
    v_circ = math.sqrt(MU_MOON_SI / r_a)
    v_ell  = math.sqrt(MU_MOON_SI * (2.0 / r_a - 1.0 / a))
    return float(abs(v_circ - v_ell))


def circularisation_dv_ms(h_periapsis_m: float, h_apoapsis_m: float) -> float:
    """
    Circularisation ΔV [m/s] at apolune of a lunar elliptic orbit.

    Used for ascent trajectories that insert into an ellipse before circularising.
    """
    r_p = R_MOON_M + float(h_periapsis_m)
    r_a = R_MOON_M + float(h_apoapsis_m)
    if r_a <= r_p:
        return 0.0
    a       = 0.5 * (r_a + r_p)
    v_ell_a = math.sqrt(MU_MOON_SI * (2.0 / r_a - 1.0 / a))
    v_circ  = math.sqrt(MU_MOON_SI / r_a)
    return float(abs(v_circ - v_ell_a))


def phasing_dv_ms(h_llo_m: float, delta_theta_deg: float, n_orbits: int) -> float:
    """
    Small-angle phasing ΔV estimate [m/s].

    Formula: ΔV ≈ (2/3) × v_circ × |Δθ| / (2πN)
    """
    if n_orbits < 1:
        raise ValueError("n_orbits must be ≥ 1.")
    dtheta = math.radians(abs(float(delta_theta_deg)))
    if dtheta == 0.0:
        return 0.0
    v_circ = circular_speed_ms(h_llo_m)
    return float((2.0 / 3.0) * v_circ * dtheta / (2.0 * math.pi * float(n_orbits)))


def tsiolkovsky_fuel_kg(m0: float, isp: float, dv: float, g0: float = G0) -> float:
    """Propellant mass [kg] for a given ΔV from initial mass m0."""
    if dv <= 0.0:
        return 0.0
    return float(m0) * (1.0 - math.exp(-float(dv) / (float(isp) * g0)))


# ---------------------------------------------------------------------------
# Budget correction helper
# ---------------------------------------------------------------------------

def apply_dv_corrections(
    dv_ideal: float,
    *,
    terminal_dv: float = 0.0,
    reserve_frac: float = 0.0,
    phase_dv: float = 0.0,
    plane_dv: float = 0.0,
) -> tuple[float, float]:
    """
    Apply standard mission-budget corrections to an ideal ΔV.

    Returns (dv_corrected, dv_extra_above_ideal).
    Formula: dv_corrected = (dv_ideal + plane + phase + terminal) × (1 + reserve)
    """
    dv_base     = float(dv_ideal) + float(plane_dv) + float(phase_dv) + float(terminal_dv)
    dv_corrected = dv_base * (1.0 + float(reserve_frac))
    return float(dv_corrected), float(max(dv_corrected - dv_ideal, 0.0))


# ---------------------------------------------------------------------------
# 9:2 NRHO phasing diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundTripPhasingDiagnostic:
    """Mission-clock phasing state at the point the crew is back in LLO."""

    departure_epoch_s:            float
    llo_ready_epoch_s:            float
    llo_ready_ta_nd:              float   # [CR3BP time]
    elapsed_before_return_s:      float
    mission_period_s:             float
    phase_offset_s:               float
    phase_offset_fraction:        float
    passive_wait_to_next_window_s: float
    phase_family:                 str

    @property
    def mission_period_days(self) -> float:
        return self.mission_period_s / 86400.0


def _classify_phase_family(phi: float) -> str:
    """Bucket a [0, 1] phase fraction into a human-readable family."""
    phi = float(phi) % 1.0
    buckets = [
        (0.000, "perfect symmetry"),
        (0.125, "near-symmetric"),
        (0.250, "quarter-phase"),
        (0.375, "three-quarter-ish"),
        (0.500, "opposite-phase / worst-case"),
        (0.625, "mirror of three-quarter-ish"),
        (0.750, "mirror of quarter-phase"),
        (0.875, "near-symmetric"),
    ]
    _, label = min(buckets, key=lambda b: min(abs(phi - b[0]), 1.0 - abs(phi - b[0])))
    return label


def round_trip_phasing(
    departure_ta_nd: float,
    downleg_tof_s: float,
    descent_tof_s: float,
    surface_duration_s: float,
    ascent_tof_s: float,
    mission_period_s: float | None = None,
) -> RoundTripPhasingDiagnostic:
    """
    Build mission-clock phasing state at LLO readiness after ascent.

    Parameters
    ----------
    departure_ta_nd    : NRHO departure epoch [CR3BP time].
    downleg_tof_s      : NRHO→LLO transfer time [s].
    descent_tof_s      : LLO→surface descent time [s].
    surface_duration_s : surface stay duration [s].
    ascent_tof_s       : surface→LLO ascent time [s].
    mission_period_s   : reference NRHO period [s].
                         Defaults to the canonical 9:2 reference period from
                         ``cr3bp_dynamics`` (~6.56 days).
    """
    from .cr3bp_dynamics import REFERENCE_92_PERIOD_S
    if mission_period_s is None:
        mission_period_s = REFERENCE_92_PERIOD_S

    dep_s     = float(departure_ta_nd) * float(T_SCALE)
    elapsed   = (float(downleg_tof_s) + float(descent_tof_s)
                 + float(surface_duration_s) + float(ascent_tof_s))
    ready_s   = dep_s + elapsed

    period_s  = float(mission_period_s)
    offset_s  = elapsed % period_s
    frac      = offset_s / period_s
    wait_s    = (period_s - offset_s) % period_s

    return RoundTripPhasingDiagnostic(
        departure_epoch_s=dep_s,
        llo_ready_epoch_s=ready_s,
        llo_ready_ta_nd=ready_s / float(T_SCALE),
        elapsed_before_return_s=elapsed,
        mission_period_s=period_s,
        phase_offset_s=offset_s,
        phase_offset_fraction=frac,
        passive_wait_to_next_window_s=wait_s,
        phase_family=_classify_phase_family(frac),
    )
