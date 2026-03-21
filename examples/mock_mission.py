"""
Mock mission example: low-thrust Earth-to-Moon transfer.

Uses a completely fictitious electric-propulsion tug ("Helios-1") to
demonstrate the trajectory propagators and the example plotting workflow.

Run
---
    python examples/mock_mission.py

Produces
--------
    outputs/mock_mission/helios1_earth_phase.png
    outputs/mock_mission/helios1_trajectory_3d.png
    outputs/mock_mission/helios1_summary.txt
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

from src.constants import R_EARTH, R_MOON
from src.control import ControlWeights
from src.plotting import save_orbital_history_plot, save_trajectory_views
from src.propagator import (
    make_earth_phase_third_body,
    make_moon_phase_third_body,
    propagate_earth_phase,
    propagate_moon_phase,
)


OUTPUT_DIR = REPO_ROOT / "outputs" / "mock_mission"


# ---------------------------------------------------------------------------
# Mock spacecraft: "Helios-1" fictitious electric-propulsion tug
# ---------------------------------------------------------------------------

N_THRUSTERS = 4
THRUST_PER_THRUSTER = 0.010   # [N] 10 mN each
ISP = 1600.0                  # [s]
MASS_DRY = 350.0              # [kg]
MASS_PROPELLANT = 180.0       # [kg]
MASS0 = MASS_DRY + MASS_PROPELLANT

S_SP = 25.0   # solar panel area [m^2]
S_SL = 8.0    # lateral area [m^2]
C_R = 1.8     # reflectivity coefficient [-]


# ---------------------------------------------------------------------------
# Mission geometry
# ---------------------------------------------------------------------------

COE_INITIAL = np.array([
    24_500e3 + R_EARTH,
    0.71,
    np.radians(7.0),
    0.0,
    0.0,
    0.0,
])

COE_TARGET_EARTH = np.array([
    384_400e3,
    0.055,
    np.radians(5.14),
    0.0,
    0.0,
    0.0,
])

COE_TARGET_MOON = np.array([
    5_000e3 + R_MOON,
    0.0,
    np.radians(90.0),
    np.radians(270.0),
    np.radians(90.0),
    0.0,
])

COE_MOON_INITIAL = np.array([
    60_000e3,
    0.95,
    np.radians(10.0),
    0.0,
    0.0,
    0.0,
])

ENABLE_EARTH = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)
ENABLE_MOON = ControlWeights(ka=1.0, ke=1.0, ki=0.0, kw=0.0, kraan=0.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Helios-1 mock mission")
    print(f"  Outputs        : {_display_path(OUTPUT_DIR)}")
    print(f"  Initial mass   : {MASS0:.1f} kg")
    print(f"  Total thrust   : {N_THRUSTERS * THRUST_PER_THRUSTER * 1e3:.1f} mN")
    print(f"  Isp            : {ISP:.0f} s")

    third_body_earth = make_earth_phase_third_body(t0_phase=0.0)
    result_earth = run_earth_phase(third_body_earth)
    result_moon = run_moon_phase(result_earth.mass[-1])

    save_outputs(result_earth, result_moon, third_body_earth)


def run_earth_phase(third_body_earth):
    """Propagate the Earth-centred low-thrust leg."""
    print("\nRunning Earth-phase propagation ...")
    result_earth = propagate_earth_phase(
        COE_INITIAL,
        COE_TARGET_EARTH,
        MASS0,
        N_THRUSTERS,
        THRUST_PER_THRUSTER,
        ISP,
        ENABLE_EARTH,
        S_sl=S_SL,
        S_sp=S_SP,
        c_r=C_R,
        enable_eclipse=True,
        smart_mode=True,
        max_days=400.0,
        rtol=1e-7,
        atol=1e-9,
        get_third_body=third_body_earth,
    )

    print("  Earth phase complete:")
    print(f"    Transfer time : {result_earth.t_transfer / 86400:.1f} days")
    print(f"    Delta-V       : {result_earth.delta_v:.1f} m/s")
    print(f"    Propellant    : {result_earth.m_prop:.1f} kg")
    print(f"    Final mass    : {result_earth.mass[-1]:.1f} kg")
    print(f"    Converged     : {result_earth.converged}")
    return result_earth


def run_moon_phase(mass_at_soi: float):
    """Propagate the lunar-orbit capture and circularisation leg."""
    print("\nRunning Moon-phase propagation ...")
    result_moon = propagate_moon_phase(
        COE_MOON_INITIAL,
        COE_TARGET_MOON,
        mass_at_soi,
        N_THRUSTERS,
        THRUST_PER_THRUSTER,
        ISP,
        ENABLE_MOON,
        S_sp=S_SP,
        c_r=C_R,
        max_days=150.0,
        rtol=1e-7,
        atol=1e-9,
        get_third_body=make_moon_phase_third_body(),
    )

    print("  Moon phase complete:")
    print(f"    Transfer time : {result_moon.t_transfer / 86400:.1f} days")
    print(f"    Delta-V       : {result_moon.delta_v:.1f} m/s")
    print(f"    Propellant    : {result_moon.m_prop:.1f} kg")
    print(f"    Final mass    : {result_moon.mass[-1]:.1f} kg")
    return result_moon


def save_outputs(result_earth, result_moon, third_body_earth) -> None:
    """Save plots and the text summary for the mock mission."""
    earth_history_path = OUTPUT_DIR / "helios1_earth_phase.png"
    earth_traj_path = OUTPUT_DIR / "helios1_trajectory_3d.png"
    summary_path = OUTPUT_DIR / "helios1_summary.txt"

    save_orbital_history_plot(
        t_days=result_earth.t / 86400.0,
        coe=result_earth.coe,
        mass=result_earth.mass,
        save_path=earth_history_path,
        title="Helios-1 Mock Tug - Earth Phase Low-Thrust Transfer",
    )

    moon_track = np.array([third_body_earth(t)[0] for t in result_earth.t])
    save_trajectory_views(
        trajectory=result_earth.r_eci,
        reference_trajectory=moon_track,
        central_body_radius=R_EARTH,
        save_path=earth_traj_path,
        title="Low-thrust Earth to Moon SOI trajectory (mock data)",
        axis_unit_label="10^3 km",
        scale=1e6,
        trajectory_label="Transfer trajectory",
        reference_label="Moon ephemeris",
        body_label="Earth",
        body_color="steelblue",
        end_label="Moon SOI arrival",
    )

    summary_text = build_summary(result_earth, result_moon)
    summary_path.write_text(summary_text)

    print(f"\nSaved: {_display_path(earth_history_path)}")
    print(f"Saved: {_display_path(earth_traj_path)}")
    print(f"Saved: {_display_path(summary_path)}")


def build_summary(result_earth, result_moon) -> str:
    """Return the plain-text summary saved alongside the plots."""
    total_prop = result_earth.m_prop + result_moon.m_prop
    total_days = (result_earth.t_transfer + result_moon.t_transfer) / 86400.0
    total_dv = result_earth.delta_v + result_moon.delta_v

    return f"""
Helios-1 Mock Mission Summary
=============================
Initial mass      : {MASS0:.1f} kg
Dry mass          : {MASS_DRY:.1f} kg
Thruster cluster  : {N_THRUSTERS} x {THRUST_PER_THRUSTER * 1e3:.0f} mN  (Isp = {ISP:.0f} s)

Earth phase
-----------
  Transfer time   : {result_earth.t_transfer / 86400:.1f} days
  Propellant      : {result_earth.m_prop:.1f} kg
  Delta-V         : {result_earth.delta_v:.1f} m/s

Moon phase
----------
  Transfer time   : {result_moon.t_transfer / 86400:.1f} days
  Propellant      : {result_moon.m_prop:.1f} kg
  Delta-V         : {result_moon.delta_v:.1f} m/s

Total mission
-------------
  Transfer time   : {total_days:.1f} days
  Total propellant: {total_prop:.1f} kg
  Total Delta-V   : {total_dv:.1f} m/s
  Final mass      : {result_moon.mass[-1]:.1f} kg

NOTE: All values are from a mock (fictitious) spacecraft.
      No real mission or ESA vehicle data is represented.
""".strip() + "\n"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
