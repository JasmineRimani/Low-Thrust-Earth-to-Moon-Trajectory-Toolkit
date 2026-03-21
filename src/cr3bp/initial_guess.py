"""
Initial-guess manager for NRHO → LLO transfer optimisation.

Three strategies are tried in cascade until a viable seed is found:

1. **Transfer scan**   — Reuses the two-impulse TA scan from ``nrho_llo_transfer``
                         and converts the best candidate into an x0 vector.
                         Most reliable; matches the active transfer basin.

2. **Literature**      — Five (TA, ToF) pairs extracted from published transfer
                         families for the 9:2 NRHO near perilune.
                         Fast; works well for near-polar LLO close to these families.

3. **Grid search**     — Coarse N_ta × N_tof free-coast scan.
                         Robust; tolerates arbitrary h_llo and inclination.

References
----------
Zimovan-Spreen et al. (2020): Near Rectilinear Halo Orbits, J. Astronautical Sci.
Capdevila et al. (2014): Transfer Network Linking Earth, Moon and L4/L5, Adv. Space Res.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .cr3bp_dynamics import (
    T_SCALE, A_SCALE, R_MOON, P_M_VEC,
    T_0_NRHO, TA_PERILUNE,
    DT_COARSE, integrate_cr3bp, dyn_no_stm, nrho_state_at,
)
from .nrho_llo_transfer import (
    TransferSolverSettings,
    _solve_one_ta,
    _best_candidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public enums / result types
# ---------------------------------------------------------------------------

class GuessStrategy(Enum):
    TRANSFER_SCAN = auto()
    LITERATURE    = auto()
    GRID_SEARCH   = auto()


@dataclass
class GuessResult:
    """Result returned by ``get_initial_guess``."""
    x0:         np.ndarray      # (11,) initial parameter vector
    strategy:   GuessStrategy   # which strategy produced this guess
    score:      float           # proxy quality metric (lower = better)
    candidates: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# x0 layout:  [TA, u, inc, RAAN, ToF, dv1x, dv1y, dv1z, dv2x, dv2y, dv2z]
# ---------------------------------------------------------------------------

def _build_x0_from_scan(
    ta: float,
    tof_s: float,
    dv1_ms: float,
    dv2_ms: float,
    inc: float,
    raan: float,
) -> np.ndarray:
    """Convert a two-impulse scan result into x0."""
    tof_nd = tof_s / T_SCALE
    p_dep  = nrho_state_at(float(ta))
    v_hat  = p_dep[3:] / (np.linalg.norm(p_dep[3:]) + 1e-30)
    dv1_nd = -(dv1_ms / (A_SCALE / T_SCALE)) * v_hat

    # Propagate to arrival to get dv2 direction
    p0 = p_dep.copy()
    p0[3:] += dv1_nd
    _, y = integrate_cr3bp(DT_COARSE, tof_nd, p0)
    v_arr  = y[-1, 3:]
    dv2_nd = (dv2_ms / (A_SCALE / T_SCALE)) * v_arr / (np.linalg.norm(v_arr) + 1e-30)

    return np.array([
        float(ta), np.pi / 2.0, float(inc), float(raan), float(tof_nd),
        dv1_nd[0], dv1_nd[1], dv1_nd[2],
        dv2_nd[0], dv2_nd[1], dv2_nd[2],
    ])


# ---------------------------------------------------------------------------
# Strategy 1 — Transfer-informed scan
# ---------------------------------------------------------------------------

def strategy_transfer_scan(
    h_llo_nd: float,
    inc: float,
    raan: float,
    ta_hint: float | None = None,
) -> list[tuple[float, np.ndarray]]:
    """
    Build x0 seeds from the two-impulse periapsis-targeting scan.

    Returns list of (dv_total_ms, x0) tuples, best candidate first.
    """
    from .cr3bp_dynamics import A_SCALE
    r_tgt_m  = (R_MOON + h_llo_nd) * A_SCALE
    settings = TransferSolverSettings()

    ta_grid = np.linspace(0.0, T_0_NRHO, int(settings.n_ta_candidates), endpoint=False)
    if ta_hint is not None:
        ta_grid = np.concatenate([[float(ta_hint) % T_0_NRHO], ta_grid])

    raw = []
    for ta in ta_grid:
        res = _solve_one_ta(ta, r_tgt_m, settings)
        if res is not None:
            dvt, dv1, dv2, r_km, tof_s = res
            raw.append((dvt, dv1, dv2, r_km, tof_s, float(ta)))

    if not raw:
        return []

    best = _best_candidate(raw, settings)
    others = sorted([c for c in raw if c != best], key=lambda c: (c[0], c[4]))
    ordered = [best, *others]

    return [
        (float(dvt),
         _build_x0_from_scan(ta, tof_s, dv1, dv2, inc, raan))
        for (dvt, dv1, dv2, _r, tof_s, ta) in ordered
    ]


# ---------------------------------------------------------------------------
# Strategy 2 — Literature candidates
# ---------------------------------------------------------------------------

# Five (TA offset from perilune [CR3BP], ToF [CR3BP]) pairs from published
# transfer families for the 9:2 synodic-resonant NRHO.
_LITERATURE_RAW = [
    (0.00,  2.5),   # exact perilune, short transfer
    (0.05,  3.5),
    (-0.05, 4.5),
    (0.10,  5.5),
    (-0.10, 6.5),
]
# Convert relative offsets to absolute TA (evaluated at import time)
LITERATURE_CANDIDATES = [(TA_PERILUNE + dt, tof) for dt, tof in _LITERATURE_RAW]


def strategy_literature(
    h_llo_nd: float,
    inc: float,
    raan: float,
    ta_hint: float | None = None,
) -> list[np.ndarray]:
    """
    Return up to 5 x0 vectors from published (TA, ToF) pairs near NRHO perilune.

    If ta_hint is provided, sorts by proximity to the hint.
    """
    candidates = list(LITERATURE_CANDIDATES)
    if ta_hint is not None:
        candidates.sort(key=lambda c: abs(c[0] - float(ta_hint) % T_0_NRHO))

    result = []
    for ta, tof in candidates:
        p_dep = nrho_state_at(ta)
        r_hat = p_dep[:3] - P_M_VEC
        r_hat /= max(np.linalg.norm(r_hat), 1e-12)
        t_hat  = np.cross(np.array([0.0, 0.0, 1.0]), r_hat)
        t_hat /= max(np.linalg.norm(t_hat), 1e-12)
        dv1 = -0.01 * t_hat    # small retrograde kick in CR3BP units

        x0 = np.array([
            ta, np.pi / 2.0, float(inc), float(raan), float(tof),
            dv1[0], dv1[1], dv1[2],
            0.0, 0.0, 0.0,
        ])
        result.append(x0)

    return result


# ---------------------------------------------------------------------------
# Strategy 3 — Grid search
# ---------------------------------------------------------------------------

def strategy_grid_search(
    h_llo_nd: float,
    inc: float,
    raan: float,
    n_ta: int = 12,
    n_tof: int = 8,
    ta_range: tuple[float, float] | None = None,
    tof_range: tuple[float, float] = (1.5, 7.0),
) -> list[tuple[float, np.ndarray]]:
    """
    Coarse free-coast grid scan over (TA, ToF).

    Measures final Moon-distance error vs target radius.
    Returns list of (score, x0) sorted by score (ascending).
    """
    r_tgt = R_MOON + h_llo_nd

    if ta_range is None:
        ta_range = (0.0, T_0_NRHO)

    ta_grid  = np.linspace(ta_range[0],  ta_range[1],  n_ta,  endpoint=False)
    tof_grid = np.linspace(tof_range[0], tof_range[1], n_tof)

    scored = []
    for ta in ta_grid:
        p_dep = nrho_state_at(ta)
        for tof in tof_grid:
            _, y = integrate_cr3bp(DT_COARSE, tof, p_dep)
            dist  = np.linalg.norm(y[-1, :3] - P_M_VEC)
            score = abs(dist - r_tgt)
            if score > 3.0 * R_MOON:
                continue
            x0 = np.array([ta, np.pi / 2.0, float(inc), float(raan), float(tof),
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            scored.append((float(score), x0))

    scored.sort(key=lambda s: s[0])
    return scored


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def get_initial_guess(
    h_llo_nd: float,
    inc: float,
    raan: float,
    ta_hint: float | None = None,
) -> GuessResult:
    """
    Cascade initial-guess manager.  Returns the best seed found by any strategy.

    Parameters
    ----------
    h_llo_nd : LLO altitude [CR3BP length].
    inc      : LLO inclination [rad].
    raan     : LLO RAAN [rad].
    ta_hint  : preferred departure epoch hint [CR3BP time].

    Returns
    -------
    GuessResult  (always returns something — worst-case: best grid-search seed)
    """
    _THRESHOLD = 2.0 * R_MOON
    all_candidates: list[np.ndarray] = []
    best_score: float = np.inf
    best_x0: np.ndarray | None = None

    # --- Strategy 1: transfer scan ---
    scan = strategy_transfer_scan(h_llo_nd, inc, raan, ta_hint=ta_hint)
    if scan:
        all_candidates.extend(x0 for _, x0 in scan)
        best_score, best_x0 = float(scan[0][0]), scan[0][1].copy()
        logger.debug(f"get_initial_guess: TRANSFER_SCAN  dv={best_score:.2f} m/s")
        return GuessResult(x0=scan[0][1], strategy=GuessStrategy.TRANSFER_SCAN,
                           score=best_score, candidates=all_candidates)

    # --- Strategy 2: literature ---
    r_tgt = R_MOON + h_llo_nd
    for x0 in strategy_literature(h_llo_nd, inc, raan, ta_hint=ta_hint):
        ta, tof = x0[0], x0[4]
        p_dep = nrho_state_at(ta)
        _, y  = integrate_cr3bp(DT_COARSE, tof, p_dep)
        score = float(abs(np.linalg.norm(y[-1, :3] - P_M_VEC) - r_tgt))
        all_candidates.append(x0)
        if score < best_score:
            best_score, best_x0 = score, x0.copy()
        if score < _THRESHOLD:
            logger.debug(f"get_initial_guess: LITERATURE  score={score:.4f}")
            return GuessResult(x0=x0, strategy=GuessStrategy.LITERATURE,
                               score=score, candidates=all_candidates)

    # --- Strategy 3: grid search ---
    for score, x0 in strategy_grid_search(h_llo_nd, inc, raan):
        all_candidates.append(x0)
        if score < best_score:
            best_score, best_x0 = score, x0.copy()
        if score < _THRESHOLD:
            logger.debug(f"get_initial_guess: GRID_SEARCH  score={score:.4f}")
            return GuessResult(x0=x0, strategy=GuessStrategy.GRID_SEARCH,
                               score=score, candidates=all_candidates)

    # Fallback
    logger.warning(f"get_initial_guess: best score={best_score:.4f}; returning best available.")
    assert best_x0 is not None
    return GuessResult(x0=best_x0, strategy=GuessStrategy.GRID_SEARCH,
                       score=best_score, candidates=all_candidates)
