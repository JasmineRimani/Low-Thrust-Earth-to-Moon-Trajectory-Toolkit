"""
Exploratory Earth-to-LLO public demo.

This script saves a paper-style mission profile in two segments:

1. Earth-phase low-thrust propagation from a GTO-like orbit toward the Moon.
2. Moon-phase low-thrust circularisation attempt from a representative
   captured lunar ellipse toward a 100 km circular LLO.

The second leg is intentionally labelled exploratory. In the current public
version the Earth-to-Moon handoff is not yet validated as a continuous
end-to-end Earth-to-LLO workflow, so the Moon-phase starts from a
representative capture orbit instead of a proven SOI-entry state.

The Earth-phase initial orbit and control flags are aligned with the public
MATLAB reference repository the user pointed to, while keeping the output
limited to the public mission-analysis core.

Run
---
    python examples/earth_to_llo_exploratory.py

Produces
--------
    outputs/earth_to_llo_exploratory/earth_phase_history.png
    outputs/earth_to_llo_exploratory/earth_phase_trajectory.png
    outputs/earth_to_llo_exploratory/moon_phase_history.png
    outputs/earth_to_llo_exploratory/moon_phase_trajectory.png
    outputs/earth_to_llo_exploratory/mission_summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import MU_EARTH, MU_MOON, R_EARTH, R_MOON
from src.control import ControlWeights
from src.orbital_elements import coe2eci
from src.plotting import (
    save_orbital_history_plot,
    save_paper_style_transfer_plot,
    save_trajectory_views,
)
from src.propagator import (
    make_earth_phase_third_body,
    propagate_earth_phase,
    propagate_moon_phase,
)
from src.validation import format_validation_report, validate_earth_phase


OUTPUT_DIR = REPO_ROOT / "outputs" / "earth_to_llo_exploratory"


# ---------------------------------------------------------------------------
# Spacecraft and guidance setup
# ---------------------------------------------------------------------------

N_THRUSTERS = 4
THRUST_PER_THRUSTER = 0.010   # [N]
ISP = 1600.0                  # [s]
MASS0 = 400.0                 # [kg]
S_SP = 25.0                   # [m^2]
S_SL = 8.0                    # [m^2]
C_R = 1.8

EARTH_WEIGHTS = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)
MOON_WEIGHTS = ControlWeights(ka=1.0, ke=1.0, ki=0.0, kw=0.0, kraan=0.0)


# ---------------------------------------------------------------------------
# Mission setup
# ---------------------------------------------------------------------------

EARTH_INITIAL_COE = np.array([
    35_787e3 + R_EARTH,
    0.7285,
    np.radians(6.0),
    0.0,
    0.0,
    0.0,
])

EARTH_TARGET_COE = np.array([
    384_400e3,
    0.055,
    np.radians(6.0),
    0.0,
    0.0,
    0.0,
])

LLO_ALTITUDE_M = 100e3
CAPTURE_PERILUNE_ALT_M = 300e3
CAPTURE_APOLUNE_ALT_M = 20_000e3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running exploratory Earth-to-LLO mission profile")
    print(f"  Outputs      : {_display_path(OUTPUT_DIR)}")
    print(f"  Initial mass : {MASS0:.1f} kg")
    print(f"  Total thrust : {N_THRUSTERS * THRUST_PER_THRUSTER * 1e3:.1f} mN")
    print(f"  Isp          : {ISP:.0f} s")

    third_body = make_earth_phase_third_body(t0_phase=0.0)
    earth_result = propagate_earth_phase(
        EARTH_INITIAL_COE,
        EARTH_TARGET_COE,
        MASS0,
        N_THRUSTERS,
        THRUST_PER_THRUSTER,
        ISP,
        EARTH_WEIGHTS,
        S_sl=S_SL,
        S_sp=S_SP,
        c_r=C_R,
        enable_eclipse=True,
        smart_mode=True,
        max_days=400.0,
        rtol=2e-6,
        atol=1e-8,
        get_third_body=third_body,
    )

    moon_initial = build_capture_orbit()
    moon_target = build_target_llo()
    moon_result = propagate_moon_phase(
        moon_initial,
        moon_target,
        earth_result.mass[-1],
        N_THRUSTERS,
        THRUST_PER_THRUSTER,
        ISP,
        MOON_WEIGHTS,
        max_days=10.0,
        rtol=2e-6,
        atol=1e-8,
    )

    save_outputs(earth_result, moon_result, moon_initial, moon_target, third_body)


def build_capture_orbit() -> np.ndarray:
    """Representative captured lunar ellipse used for the exploratory leg."""
    rp = R_MOON + CAPTURE_PERILUNE_ALT_M
    ra = R_MOON + CAPTURE_APOLUNE_ALT_M
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    return np.array([a, e, np.radians(90.0), 0.0, 0.0, 0.0])


def build_target_llo() -> np.ndarray:
    """100 km circular polar LLO reference."""
    return np.array([R_MOON + LLO_ALTITUDE_M, 0.0, np.radians(90.0), 0.0, 0.0, 0.0])


def save_outputs(
    earth_result,
    moon_result,
    moon_initial: np.ndarray,
    moon_target: np.ndarray,
    third_body,
) -> None:
    """Save the exploratory Earth and Moon phase plots plus a summary."""
    earth_history_path = OUTPUT_DIR / "earth_phase_history.png"
    earth_trajectory_path = OUTPUT_DIR / "earth_phase_trajectory.png"
    moon_history_path = OUTPUT_DIR / "moon_phase_history.png"
    moon_trajectory_path = OUTPUT_DIR / "moon_phase_trajectory.png"
    summary_path = OUTPUT_DIR / "mission_summary.txt"

    save_orbital_history_plot(
        t_days=earth_result.t / 86400.0,
        coe=earth_result.coe,
        mass=earth_result.mass,
        save_path=earth_history_path,
        title="Exploratory Earth-phase transfer history",
    )

    reference_gto_orbit = _sample_orbit_track(MU_EARTH, EARTH_INITIAL_COE)
    save_paper_style_transfer_plot(
        trajectory=earth_result.r_eci,
        reference_trajectory=reference_gto_orbit,
        central_body_radius=R_EARTH,
        save_path=earth_trajectory_path,
        title="Exploratory Earth-phase low-thrust transfer",
        axis_unit_label="10^8 m",
        scale=1e8,
        body_color="0.92",
        transfer_color="blue",
        reference_color="black",
    )

    moon_prefix = _stable_prefix(moon_result)
    moon_slice = slice(0, moon_prefix)
    moon_title_suffix = "representative capture orbit"
    if moon_prefix >= 2:
        save_orbital_history_plot(
            t_days=moon_result.t[moon_slice] / 86400.0,
            coe=moon_result.coe[moon_slice],
            mass=moon_result.mass[moon_slice],
            save_path=moon_history_path,
            title="Exploratory Moon-phase circularisation history",
            color="darkorange",
        )

        save_trajectory_views(
            trajectory=moon_result.r_eci[moon_slice],
            reference_trajectory=_target_llo_track(moon_target),
            central_body_radius=R_MOON,
            save_path=moon_trajectory_path,
            title="Representative capture orbit toward 100 km LLO",
            axis_unit_label="km",
            scale=1e3,
            trajectory_label="Low-thrust circularisation attempt",
            reference_label="100 km LLO reference",
            body_label="Moon",
            body_color="dimgray",
            end_label="End of Moon-phase run",
        )
    else:
        moon_title_suffix = "no stable Moon-phase history available"

    summary_path.write_text(
        build_summary(earth_result, moon_result, moon_initial, moon_target, moon_prefix, moon_title_suffix, third_body)
    )

    print(f"\nSaved: {_display_path(earth_history_path)}")
    print(f"Saved: {_display_path(earth_trajectory_path)}")
    if moon_prefix >= 2:
        print(f"Saved: {_display_path(moon_history_path)}")
        print(f"Saved: {_display_path(moon_trajectory_path)}")
    print(f"Saved: {_display_path(summary_path)}")


def build_summary(
    earth_result,
    moon_result,
    moon_initial: np.ndarray,
    moon_target: np.ndarray,
    moon_prefix: int,
    moon_title_suffix: str,
    third_body,
) -> str:
    """Return the mission summary saved alongside the plots."""
    moon_track = np.array([third_body(t)[0] for t in earth_result.t])
    earth_distances_km = np.linalg.norm(earth_result.r_eci - moon_track, axis=1) / 1e3
    earth_checks = validate_earth_phase(
        dv_ms=earth_result.delta_v,
        transfer_days=earth_result.t_transfer / 86400.0,
    )

    lines = [
        "Exploratory Earth-to-LLO Mission Summary",
        "=======================================",
        "",
        "Earth phase",
        "-----------",
        f"  Transfer time         : {earth_result.t_transfer / 86400.0:.1f} days",
        f"  Delta-V               : {earth_result.delta_v:.1f} m/s",
        f"  Propellant used       : {earth_result.m_prop:.1f} kg",
        f"  Final mass            : {earth_result.mass[-1]:.1f} kg",
        f"  Target reached        : {earth_result.target_reached}",
        f"  Stop reason           : {earth_result.stop_reason}",
        f"  Minimum Moon distance : {earth_distances_km.min():.0f} km",
        f"  Final Moon distance   : {earth_distances_km[-1]:.0f} km",
        format_validation_report(earth_checks, "Earth-phase benchmark bounds"),
        "",
        "Moon phase",
        "----------",
        "  This leg is exploratory in the current public version.",
        "  It starts from a representative captured lunar ellipse, not from a",
        "  validated Earth-to-Moon SOI handoff state.",
        f"  Initial perilune alt. : {moon_initial[0] * (1.0 - moon_initial[1]) / 1e3 - R_MOON / 1e3:.0f} km",
        f"  Initial apolune alt.  : {moon_initial[0] * (1.0 + moon_initial[1]) / 1e3 - R_MOON / 1e3:.0f} km",
        f"  Target orbit altitude : {moon_target[0] / 1e3 - R_MOON / 1e3:.0f} km circular",
        f"  Simulated duration    : {moon_result.t_transfer / 86400.0:.1f} days",
        f"  Delta-V               : {moon_result.delta_v:.1f} m/s",
        f"  Propellant used       : {moon_result.m_prop:.1f} kg",
        f"  Target reached        : {moon_result.target_reached}",
        f"  Stop reason           : {moon_result.stop_reason}",
        f"  Stable plotted prefix : {moon_prefix} samples ({moon_title_suffix})",
    ]

    if moon_prefix >= 1:
        moon_last = moon_result.coe[moon_prefix - 1]
        lines.extend([
            f"  Last plotted perilune : {moon_last[0] * (1.0 - moon_last[1]) / 1e3 - R_MOON / 1e3:.0f} km",
            f"  Last plotted apolune  : {moon_last[0] * (1.0 + moon_last[1]) / 1e3 - R_MOON / 1e3:.0f} km",
            f"  Last plotted e        : {moon_last[1]:.4f}",
        ])

    lines.extend([
        "",
        "Notes",
        "-----",
        "  These plots are intended as a public exploratory workflow inspired by",
        "  the mission profile style of the paper, not as a validated reproduction",
        "  of a full end-to-end Earth-to-LLO trajectory.",
        "  The Earth-centred transfer plot uses the initial GTO-like orbit as",
        "  the black guide curve, to better match paper-style transfer figures.",
    ])

    return "\n".join(lines).rstrip() + "\n"


def _target_llo_track(target_coe: np.ndarray) -> np.ndarray:
    """Sample a circular LLO reference orbit for plotting."""
    return _sample_orbit_track(MU_MOON, target_coe)


def _sample_orbit_track(mu: float, base_coe: np.ndarray, n_samples: int = 361) -> np.ndarray:
    """Sample a reference orbit for plotting in a fixed inertial frame."""
    samples = []
    for nu in np.linspace(0.0, 2.0 * np.pi, n_samples):
        coe = np.array(base_coe, dtype=float, copy=True)
        coe[5] = nu
        r_eci, _ = coe2eci(mu, coe)
        samples.append(r_eci)
    return np.asarray(samples)


def _stable_prefix(result) -> int:
    """Return the longest prefix with finite, bounded Moon-phase states."""
    if len(result.t) == 0:
        return 0

    radius = np.linalg.norm(result.r_eci, axis=1)
    mask = np.isfinite(result.t)
    mask &= np.isfinite(result.mass)
    mask &= np.isfinite(result.coe).all(axis=1)
    mask &= np.isfinite(result.r_eci).all(axis=1)
    mask &= result.coe[:, 0] > 0.0
    mask &= result.coe[:, 1] < 0.999
    mask &= radius < 100.0e6

    invalid = np.flatnonzero(~mask)
    if invalid.size == 0:
        return len(result.t)
    return int(invalid[0])


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
