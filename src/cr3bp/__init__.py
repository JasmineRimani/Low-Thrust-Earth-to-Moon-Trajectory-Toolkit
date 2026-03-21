"""
CR3BP-based NRHO ↔ LLO mission analysis tools.

Experimental extension utilities for near-Moon dynamics.

After the spacecraft enters the Moon's sphere of influence, this subpackage
provides CR3BP helpers for analysis around the 9:2 NRHO and low lunar orbit
(LLO).

Key modules
-----------
cr3bp_dynamics   : CR3BP equations of motion, Jacobi constant, NRHO propagation.
frames           : CR3BP / MCI / LVLH frame transforms.
llo_state        : Circular LLO state in rotating frame with gradients.
nrho_llo_transfer: NRHO ↔ LLO preliminary transfer estimator.
mission_utils    : ΔV corrections, phasing, Tsiolkovsky fuel accounting.
initial_guess    : Three-strategy cascade initial-guess manager.
"""

from .cr3bp_dynamics import (
    jacobi_constant,
    jacobi_max_drift,
    nrho_state_at,
    T_0_NRHO,
    TA_PERILUNE,
    A_SCALE,
    T_SCALE,
    V_SCALE,
    R_MOON_M,
    MU_MOON_SI,
)
from .frames import cr3bp_to_mci, mci_to_cr3bp, lvlh_basis_from_mci
from .nrho_llo_transfer import (
    nrho_to_llo,
    llo_to_nrho,
    TransferResult,
    TransferSolverSettings,
)
from .mission_utils import (
    circular_speed_ms,
    doi_dv_ms,
    circularisation_dv_ms,
    phasing_dv_ms,
    tsiolkovsky_fuel_kg,
    apply_dv_corrections,
    round_trip_phasing,
    RoundTripPhasingDiagnostic,
)
from .initial_guess import (
    get_initial_guess,
    GuessResult,
    GuessStrategy,
)

__all__ = [
    "jacobi_constant", "jacobi_max_drift",
    "nrho_state_at", "T_0_NRHO", "TA_PERILUNE",
    "A_SCALE", "T_SCALE", "V_SCALE", "R_MOON_M", "MU_MOON_SI",
    "cr3bp_to_mci", "mci_to_cr3bp", "lvlh_basis_from_mci",
    "nrho_to_llo", "llo_to_nrho", "TransferResult", "TransferSolverSettings",
    "circular_speed_ms", "doi_dv_ms", "circularisation_dv_ms",
    "phasing_dv_ms", "tsiolkovsky_fuel_kg", "apply_dv_corrections",
    "round_trip_phasing", "RoundTripPhasingDiagnostic",
    "get_initial_guess", "GuessResult", "GuessStrategy",
]
