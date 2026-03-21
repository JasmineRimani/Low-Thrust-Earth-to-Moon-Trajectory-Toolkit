"""
Full GTO to NRHO to LLO mission chain.

Demonstrates the complete three-phase trajectory:

  Phase 1 - Earth phase    (MAGNETO Lyapunov low-thrust, GTO to Moon SOI)
  Phase 2 - Moon approach  (MAGNETO MEE propagator, SOI to high lunar orbit)
  Phase 3 - NRHO insertion (CR3BP two-impulse, NRHO to LLO)

Spacecraft sizing is intentionally minimal:
  - Only m_dry and m_prop are specified as inputs.
  - All mass accounting is Tsiolkovsky only.
  - No subsystem breakdown, no ESA margins, no component databases.

Run
---
    python examples/full_mission_chain.py

Produces
--------
    outputs/full_mission/full_mission_summary.txt
    outputs/full_mission/full_mission_earth_phase.png
    outputs/full_mission/full_mission_earth_trajectory.png
    outputs/full_mission/full_mission_nrho_transfer.png
    outputs/full_mission/full_mission_dv_budget.png
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

import matplotlib.pyplot as plt

from src.constants import R_EARTH
from src.control import ControlWeights
from src.plotting import save_orbital_history_plot, save_trajectory_views
from src.propagator import (
    make_earth_phase_third_body,
    propagate_earth_phase,
    propagate_moon_phase,
)
from src.validation import (
    print_validation_report,
    validate_earth_phase,
    validate_nrho_llo,
    validate_tsiolkovsky,
)
from src.cr3bp import (
    A_SCALE,
    R_MOON_M,
    TA_PERILUNE,
    T_0_NRHO,
    TransferSolverSettings,
    apply_dv_corrections,
    doi_dv_ms,
    gateway_round_trip_phasing,
    get_initial_guess,
    jacobi_max_drift,
    llo_to_nrho,
    nrho_state_at,
    nrho_to_llo,
)


OUTPUT_DIR = REPO_ROOT / "outputs" / "full_mission"


# ============================================================
# Spacecraft (fictitious - EP only)
# ============================================================
N_THR = 4
THRUST_N = 0.010
ISP_EP = 1600.0
M_DRY = 400.0
M_PROP_EP = 160.0
M0 = M_DRY + M_PROP_EP


# ============================================================
# Mission setup
# ============================================================
COE_GTO = np.array([
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

COE_MOON_IN = np.array([
    50_000e3,
    0.93,
    np.radians(15.0),
    0.0,
    0.0,
    0.0,
])

COE_MOON_TGT = np.array([
    20_000e3,
    0.75,
    np.radians(90.0),
    0.0,
    0.0,
    0.0,
])

ENABLE_EARTH = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)
ENABLE_MOON = ControlWeights(ka=1.0, ke=1.0, ki=0.0, kw=0.0, kraan=0.0)

H_LLO_M = 100_000.0
INC_LLO = np.pi / 2


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Helios-2 Mock Mission - Full GTO to NRHO to LLO Chain")
    print("=" * 60)
    print(f"  Outputs : {_display_path(OUTPUT_DIR)}")
    print(f"  m_dry   = {M_DRY:.0f} kg")
    print(f"  m_prop  = {M_PROP_EP:.0f} kg  (EP, Isp={ISP_EP:.0f} s)")
    print(f"  m0      = {M0:.0f} kg")

    third_body_earth = make_earth_phase_third_body()

    res_earth = run_earth_phase(third_body_earth)
    res_moon = run_moon_phase(res_earth.mass[-1])
    phase3 = run_cr3bp_phase(res_moon.mass[-1])

    save_outputs(
        res_earth=res_earth,
        res_moon=res_moon,
        third_body_earth=third_body_earth,
        res_nrho_llo=phase3["res_nrho_llo"],
        res_llo_nrho=phase3["res_llo_nrho"],
    )

    summary_text = build_summary(
        res_earth=res_earth,
        res_moon=res_moon,
        res_nrho_llo=phase3["res_nrho_llo"],
        res_llo_nrho=phase3["res_llo_nrho"],
        jac_drift=phase3["jac_drift"],
        phasing=phase3["phasing"],
    )
    summary_path = OUTPUT_DIR / "full_mission_summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"Saved: {_display_path(summary_path)}")
    print("\nDone.")


def run_earth_phase(third_body_earth):
    """Phase 1: Earth low-thrust spiral from GTO to Moon SOI."""
    print("\n[1/3] Earth low-thrust phase (GTO to Moon SOI) ...")
    res_earth = propagate_earth_phase(
        COE_GTO,
        COE_TARGET_EARTH,
        M0,
        N_THR,
        THRUST_N,
        ISP_EP,
        ENABLE_EARTH,
        enable_eclipse=True,
        smart_mode=True,
        max_days=400.0,
        rtol=1e-7,
        atol=1e-9,
        get_third_body=third_body_earth,
    )

    print(f"  Transfer time : {res_earth.t_transfer / 86400:.1f} days")
    print(f"  Delta-V       : {res_earth.delta_v:.0f} m/s")
    print(f"  Propellant    : {res_earth.m_prop:.1f} kg")
    print(f"  Mass at SOI   : {res_earth.mass[-1]:.1f} kg")
    print(f"  Converged     : {res_earth.converged}")

    checks_earth = validate_earth_phase(res_earth.delta_v, res_earth.t_transfer / 86400.0)
    print_validation_report(checks_earth, "Earth low-thrust phase")
    return res_earth


def run_moon_phase(mass_at_soi: float):
    """Phase 2: lunar-orbit shaping before the CR3BP handoff."""
    print("\n[2/3] Moon approach phase (SOI to high lunar orbit) ...")
    res_moon = propagate_moon_phase(
        COE_MOON_IN,
        COE_MOON_TGT,
        mass_at_soi,
        N_THR,
        THRUST_N,
        ISP_EP,
        ENABLE_MOON,
        max_days=60.0,
        rtol=1e-7,
        atol=1e-9,
    )

    print(f"  Transfer time : {res_moon.t_transfer / 86400:.1f} days")
    print(f"  Delta-V       : {res_moon.delta_v:.0f} m/s")
    print(f"  Propellant    : {res_moon.m_prop:.1f} kg")
    print(f"  Mass post-EP  : {res_moon.mass[-1]:.1f} kg")
    return res_moon


def run_cr3bp_phase(mass_cr3bp: float) -> dict[str, object]:
    """Phase 3: two-impulse NRHO to LLO transfer and return leg sizing."""
    print("\n[3/3] CR3BP two-impulse phase (high orbit to NRHO to LLO) ...")

    h_llo_nd = H_LLO_M / A_SCALE
    guess = get_initial_guess(h_llo_nd, INC_LLO, 0.0, ta_hint=TA_PERILUNE)
    print(f"  Initial guess : strategy={guess.strategy.name}  score={guess.score:.4f}")

    settings = TransferSolverSettings(n_ta_candidates=12, tof_max_cr3bp=5.0)
    res_nrho_llo = nrho_to_llo(
        m0=mass_cr3bp,
        isp=ISP_EP,
        h_llo_m=H_LLO_M,
        inc_llo=INC_LLO,
        ta_hint=TA_PERILUNE,
        settings=settings,
    )

    print(f"  Delta-V total : {res_nrho_llo.dv_total:.1f} m/s")
    print(f"  Delta-V dep.  : {np.linalg.norm(res_nrho_llo.dv1):.1f} m/s")
    print(f"  Delta-V ins.  : {np.linalg.norm(res_nrho_llo.dv2):.1f} m/s")
    print(
        f"  Time of flight: {res_nrho_llo.time_of_flight / 3600:.1f} h"
        f"  ({res_nrho_llo.time_of_flight / 86400:.2f} days)"
    )
    print(f"  Periapsis alt : {res_nrho_llo.periapsis_alt_m / 1e3:.0f} km")
    print(f"  Fuel used     : {res_nrho_llo.fuel_mass_kg:.1f} kg")

    jac_drift = jacobi_max_drift(res_nrho_llo.trajectory)
    print(f"  Jacobi drift  : {jac_drift:.2e}  (< 1e-6 is good)")

    checks_nrho = validate_nrho_llo(
        res_nrho_llo.dv_total,
        res_nrho_llo.time_of_flight / 86400.0,
        H_LLO_M,
    )
    checks_tsiol = validate_tsiolkovsky(
        mass_cr3bp,
        res_nrho_llo.dv_total,
        ISP_EP,
        res_nrho_llo.fuel_mass_kg,
    )
    print_validation_report(checks_nrho + checks_tsiol, "NRHO to LLO transfer (EP)")

    res_llo_nrho = llo_to_nrho(
        m0=mass_cr3bp - res_nrho_llo.fuel_mass_kg,
        isp=ISP_EP,
        h_llo_m=H_LLO_M,
        dv_from_downleg=res_nrho_llo.dv_total,
        tof_from_downleg=res_nrho_llo.time_of_flight,
        ta_hint=res_nrho_llo.departure_ta,
    )
    print("\n  Return leg (LLO to NRHO, time-reversal):")
    print(f"    Delta-V     : {res_llo_nrho.dv_total:.1f} m/s")
    print(f"    Fuel used   : {res_llo_nrho.fuel_mass_kg:.1f} kg")

    dv_corr, _ = apply_dv_corrections(
        res_nrho_llo.dv_total,
        terminal_dv=doi_dv_ms(H_LLO_M, 15_000.0),
        reserve_frac=0.05,
        phase_dv=10.0,
    )
    print(f"\n  Corrected DV  : {dv_corr:.1f} m/s (DOI + reserve + phasing)")

    phasing = gateway_round_trip_phasing(
        departure_ta_nd=res_nrho_llo.departure_ta,
        downleg_tof_s=res_nrho_llo.time_of_flight,
        descent_tof_s=0.5 * 86400,
        surface_duration_s=14.0 * 86400,
        ascent_tof_s=0.5 * 86400,
    )
    print("\n  Gateway phasing after 14-day surface stay:")
    print(f"    Phase offset : {phasing.phase_offset_fraction * 100:.1f}%")
    print(f"    Phase family : {phasing.phase_family}")
    print(f"    Passive wait : {phasing.passive_wait_to_next_window_s / 3600:.1f} h")

    return {
        "res_nrho_llo": res_nrho_llo,
        "res_llo_nrho": res_llo_nrho,
        "jac_drift": jac_drift,
        "phasing": phasing,
    }


def save_outputs(
    *,
    res_earth,
    res_moon,
    third_body_earth,
    res_nrho_llo,
    res_llo_nrho,
) -> None:
    """Save the Earth-phase, CR3BP, and budget plots."""
    earth_phase_path = OUTPUT_DIR / "full_mission_earth_phase.png"
    earth_traj_path = OUTPUT_DIR / "full_mission_earth_trajectory.png"
    nrho_plot_path = OUTPUT_DIR / "full_mission_nrho_transfer.png"
    dv_budget_path = OUTPUT_DIR / "full_mission_dv_budget.png"

    save_orbital_history_plot(
        t_days=res_earth.t / 86400.0,
        coe=res_earth.coe,
        mass=res_earth.mass,
        save_path=earth_phase_path,
        title="Helios-2 - Earth Low-Thrust Phase (GTO to Moon SOI)",
    )

    moon_track = np.array([third_body_earth(t)[0] for t in res_earth.t])
    save_trajectory_views(
        trajectory=res_earth.r_eci,
        reference_trajectory=moon_track,
        central_body_radius=R_EARTH,
        save_path=earth_traj_path,
        title="Helios-2 Earth-phase trajectory (mock data)",
        axis_unit_label="10^3 km",
        scale=1e6,
        trajectory_label="Transfer trajectory",
        reference_label="Moon ephemeris",
        body_label="Earth",
        body_color="steelblue",
        end_label="Moon SOI arrival",
    )

    moon_center_offset = np.array([1.0 - 0.012150585609624, 0.0, 0.0])
    nrho_reference = np.array([
        (nrho_state_at(ta)[:3] - moon_center_offset) * A_SCALE
        for ta in np.linspace(0.0, T_0_NRHO, 240)
    ])
    nrho_transfer = (res_nrho_llo.trajectory[:, :3] - moon_center_offset) * A_SCALE

    save_trajectory_views(
        trajectory=nrho_transfer,
        reference_trajectory=nrho_reference,
        central_body_radius=R_MOON_M,
        save_path=nrho_plot_path,
        title="NRHO to LLO transfer arc (Moon-centred frame, mock data)",
        axis_unit_label="km",
        scale=1e3,
        trajectory_label="NRHO to LLO transfer arc",
        reference_label="NRHO reference orbit",
        body_label="Moon",
        body_color="silver",
        start_label="NRHO departure",
        end_label="Periapsis arrival",
    )

    save_dv_budget_plot(
        save_path=dv_budget_path,
        values=[
            res_earth.delta_v,
            res_moon.delta_v,
            res_nrho_llo.dv_total,
            res_llo_nrho.dv_total,
        ],
    )

    print(f"Saved: {_display_path(earth_phase_path)}")
    print(f"Saved: {_display_path(earth_traj_path)}")
    print(f"Saved: {_display_path(nrho_plot_path)}")
    print(f"Saved: {_display_path(dv_budget_path)}")


def save_dv_budget_plot(save_path: Path, values: list[float]) -> None:
    """Save the mission Delta-V budget chart."""
    labels = ["EP: GTO->SOI", "EP: SOI->orbit", "EP: NRHO->LLO", "EP: LLO->NRHO"]
    colors = ["steelblue", "cornflowerblue", "tomato", "salmon"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="k", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 15.0,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Delta-V [m/s]")
    ax.set_title("Mission Delta-V budget by phase - EP throughout (mock Helios-2)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    *,
    res_earth,
    res_moon,
    res_nrho_llo,
    res_llo_nrho,
    jac_drift: float,
    phasing,
) -> str:
    """Return the mission summary text saved to disk."""
    total_ep_prop = (
        res_earth.m_prop
        + res_moon.m_prop
        + res_nrho_llo.fuel_mass_kg
        + res_llo_nrho.fuel_mass_kg
    )
    total_dv_ep = (
        res_earth.delta_v
        + res_moon.delta_v
        + res_nrho_llo.dv_total
        + res_llo_nrho.dv_total
    )
    total_days = (res_earth.t_transfer + res_moon.t_transfer) / 86400.0

    return f"""
Helios-2 Mock Mission - Full Budget (EP only)
=============================================
Spacecraft (fictitious, Tsiolkovsky sizing only)
  m_dry           : {M_DRY:.0f} kg
  m_prop (EP)     : {M_PROP_EP:.0f} kg
  m0              : {M0:.0f} kg
  Isp (all phases): {ISP_EP:.0f} s  [electric propulsion throughout]
Thruster cluster  : {N_THR} x {THRUST_N * 1e3:.0f} mN

Phase 1 - GTO to Moon SOI (Lyapunov low-thrust EP)
  Transfer time   : {res_earth.t_transfer / 86400:.1f} days
  Delta-V         : {res_earth.delta_v:.0f} m/s
  Propellant used : {res_earth.m_prop:.1f} kg

Phase 2 - SOI to high lunar orbit (low-thrust EP)
  Transfer time   : {res_moon.t_transfer / 86400:.1f} days
  Delta-V         : {res_moon.delta_v:.0f} m/s
  Propellant used : {res_moon.m_prop:.1f} kg

Phase 3 - NRHO to LLO (EP, Isp={ISP_EP:.0f} s)
  Delta-V dep.    : {np.linalg.norm(res_nrho_llo.dv1):.0f} m/s
  Delta-V ins.    : {np.linalg.norm(res_nrho_llo.dv2):.0f} m/s
  Delta-V total   : {res_nrho_llo.dv_total:.0f} m/s
  Time of flight  : {res_nrho_llo.time_of_flight / 3600:.1f} h
  Propellant used : {res_nrho_llo.fuel_mass_kg:.1f} kg
  Departure TA    : {res_nrho_llo.departure_ta:.3f} CR3BP  (perilune ~ {TA_PERILUNE:.3f})
  Jacobi drift    : {jac_drift:.2e}

Return leg - LLO to NRHO (EP, time-reversal, Isp={ISP_EP:.0f} s)
  Delta-V total   : {res_llo_nrho.dv_total:.0f} m/s
  Propellant used : {res_llo_nrho.fuel_mass_kg:.1f} kg

Total mission (EP throughout)
  EP transfer time: {total_days:.1f} days
  Total Delta-V   : {total_dv_ep:.0f} m/s
  Total propellant: {total_ep_prop:.1f} kg
  Final mass      : {M0 - total_ep_prop:.1f} kg

Gateway phasing (14-day surface stay)
  Phase offset    : {phasing.phase_offset_fraction * 100:.1f}%
  Phase family    : {phasing.phase_family}
  Passive wait    : {phasing.passive_wait_to_next_window_s / 3600:.1f} h

NOTE: All values are from a mock (fictitious) spacecraft.
      Phase 3 Delta-V is trajectory geometry; propellant is Tsiolkovsky only.
""".strip() + "\n"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
