"""
Earth-phase demo: low-thrust GTO to Moon-SOI transfer.

This example intentionally focuses on the first public mission-analysis leg of
the toolkit. Spacecraft data is fictitious and used only to demonstrate the
propagator and plotting workflow.

Run
---
    python examples/mock_mission.py

Produces
--------
    outputs/mock_mission/earth_phase_history.png
    outputs/mock_mission/earth_phase_trajectory.png
    outputs/mock_mission/earth_phase_summary.txt
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

from src.constants import R_EARTH
from src.control import ControlWeights
from src.plotting import save_orbital_history_plot, save_trajectory_views
from src.propagator import make_earth_phase_third_body, propagate_earth_phase


OUTPUT_DIR = REPO_ROOT / "outputs" / "mock_mission"


# ---------------------------------------------------------------------------
# Fictitious demonstration spacecraft
# ---------------------------------------------------------------------------

N_THRUSTERS = 4
THRUST_PER_THRUSTER = 0.010   # [N]
ISP = 1600.0                  # [s]
MASS0 = 530.0                 # [kg]
S_SP = 25.0                   # [m^2]
S_SL = 8.0                    # [m^2]
C_R = 1.8


# ---------------------------------------------------------------------------
# Transfer setup
# ---------------------------------------------------------------------------

COE_INITIAL = np.array([
    24_500e3 + R_EARTH,
    0.71,
    np.radians(7.0),
    0.0,
    0.0,
    0.0,
])

COE_TARGET = np.array([
    384_400e3,
    0.055,
    np.radians(5.14),
    0.0,
    0.0,
    0.0,
])

CONTROL_WEIGHTS = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Earth-phase low-thrust demo")
    print(f"  Outputs      : {_display_path(OUTPUT_DIR)}")
    print(f"  Initial mass : {MASS0:.1f} kg")
    print(f"  Total thrust : {N_THRUSTERS * THRUST_PER_THRUSTER * 1e3:.1f} mN")
    print(f"  Isp          : {ISP:.0f} s")

    third_body = make_earth_phase_third_body(t0_phase=0.0)
    result = propagate_earth_phase(
        COE_INITIAL,
        COE_TARGET,
        MASS0,
        N_THRUSTERS,
        THRUST_PER_THRUSTER,
        ISP,
        CONTROL_WEIGHTS,
        S_sl=S_SL,
        S_sp=S_SP,
        c_r=C_R,
        enable_eclipse=True,
        smart_mode=True,
        max_days=400.0,
        rtol=1e-7,
        atol=1e-9,
        get_third_body=third_body,
    )

    print("\nEarth phase complete:")
    print(f"  Transfer time : {result.t_transfer / 86400:.1f} days")
    print(f"  Delta-V       : {result.delta_v:.1f} m/s")
    print(f"  Propellant    : {result.m_prop:.1f} kg")
    print(f"  Final mass    : {result.mass[-1]:.1f} kg")
    print(f"  Converged     : {result.converged}")

    save_outputs(result, third_body)


def save_outputs(result, third_body) -> None:
    """Save history and trajectory plots for the Earth-phase transfer."""
    history_path = OUTPUT_DIR / "earth_phase_history.png"
    trajectory_path = OUTPUT_DIR / "earth_phase_trajectory.png"
    summary_path = OUTPUT_DIR / "earth_phase_summary.txt"

    save_orbital_history_plot(
        t_days=result.t / 86400.0,
        coe=result.coe,
        mass=result.mass,
        save_path=history_path,
        title="Earth-phase low-thrust transfer history",
    )

    moon_track = np.array([third_body(t)[0] for t in result.t])
    save_trajectory_views(
        trajectory=result.r_eci,
        reference_trajectory=moon_track,
        central_body_radius=R_EARTH,
        save_path=trajectory_path,
        title="Earth to Moon-SOI low-thrust transfer (demo data)",
        axis_unit_label="10^3 km",
        scale=1e6,
        trajectory_label="Transfer trajectory",
        reference_label="Moon ephemeris",
        body_label="Earth",
        body_color="steelblue",
        end_label="Moon SOI arrival",
    )

    summary_text = build_summary(result)
    summary_path.write_text(summary_text)

    print(f"\nSaved: {_display_path(history_path)}")
    print(f"Saved: {_display_path(trajectory_path)}")
    print(f"Saved: {_display_path(summary_path)}")


def build_summary(result) -> str:
    """Return the text summary saved alongside the plots."""
    return f"""
Earth-phase Demo Summary
========================
Initial mass      : {MASS0:.1f} kg
Thruster cluster  : {N_THRUSTERS} x {THRUST_PER_THRUSTER * 1e3:.0f} mN  (Isp = {ISP:.0f} s)

Transfer result
---------------
  Transfer time   : {result.t_transfer / 86400:.1f} days
  Propellant used : {result.m_prop:.1f} kg
  Delta-V         : {result.delta_v:.1f} m/s
  Final mass      : {result.mass[-1]:.1f} kg
  Converged       : {result.converged}

NOTE: This example uses fictitious spacecraft parameters for public mission
      analysis only.
""".strip() + "\n"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
