"""
Full GTO → NRHO → LLO mission chain
=====================================
Demonstrates the complete three-phase trajectory:

  Phase 1 — Earth phase    (MAGNETO Lyapunov low-thrust, GTO → Moon SOI)
  Phase 2 — Moon approach  (MAGNETO MEE propagator, SOI → high lunar orbit)
  Phase 3 — NRHO insertion (CR3BP two-impulse, NRHO → LLO)

Spacecraft sizing is intentionally minimal:
  - Only m_dry and m_prop are specified as inputs.
  - All mass accounting is Tsiolkovsky only.
  - No subsystem breakdown, no ESA margins, no component databases.

All values use a *completely fictitious* spacecraft ("Helios-2").
Computed ΔV values are validated against published literature.

Run
---
    python examples/full_mission_chain.py

Produces
--------
    full_mission_summary.txt
    full_mission_earth_phase.png
    full_mission_nrho_transfer.png
    full_mission_dv_budget.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from src.constants    import R_EARTH, R_MOON as R_MOON_SI
from src.control      import ControlWeights
from src.propagator   import propagate_earth_phase, propagate_moon_phase
from src.soi_handoff  import soi_to_cr3bp, cr3bp_llo_to_coe
from src.validation   import (
    validate_nrho_llo, validate_earth_phase,
    validate_tsiolkovsky, print_validation_report,
)

from src.cr3bp import (
    nrho_to_llo, llo_to_nrho,
    TransferSolverSettings,
    A_SCALE, T_SCALE, V_SCALE, R_MOON_M,
    TA_PERILUNE, T_0_NRHO,
    nrho_state_at, jacobi_max_drift,
    circular_speed_ms, doi_dv_ms,
    tsiolkovsky_fuel_kg, apply_dv_corrections,
    gateway_round_trip_phasing,
    get_initial_guess,
)

print("=" * 60)
print("  Helios-2 Mock Mission — Full GTO → NRHO → LLO Chain")
print("=" * 60)

# ============================================================
# SPACECRAFT  (fictitious — m_dry and m_prop only)
# Sizing is intentionally minimal: Tsiolkovsky rocket equation
# is the only mass model used. No subsystem breakdown.
# Electric propulsion only throughout.
# ============================================================
N_THR         = 4
THRUST_N      = 0.010       # thrust per thruster [N]
ISP_EP        = 1600.0      # electric propulsion Isp [s] — used for ALL phases

# Mass budget: only two numbers matter
M_DRY         = 400.0       # dry mass [kg]
M_PROP_EP     = 160.0       # electric propellant loaded [kg]
M0            = M_DRY + M_PROP_EP   # total initial mass [kg]

print(f"  m_dry  = {M_DRY:.0f} kg")
print(f"  m_prop = {M_PROP_EP:.0f} kg  (EP, Isp={ISP_EP:.0f} s)")
print(f"  m0     = {M0:.0f} kg")

# ============================================================
# PHASE 1 — Earth low-thrust:  GTO → Moon SOI
# ============================================================
print("\n[1/3] Earth low-thrust phase  (GTO → Moon SOI) ...")

a_gto   = 24_500e3 + R_EARTH
e_gto   = 0.71
inc_gto = np.radians(7.0)
COE_GTO    = np.array([a_gto, e_gto, inc_gto, 0.0, 0.0, 0.0])
COE_TARGET = np.array([384_400e3, 0.055, np.radians(5.14), 0.0, 0.0, 0.0])

enable_earth = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)

res_earth = propagate_earth_phase(
    COE_GTO, COE_TARGET, M0,
    N_THR, THRUST_N, ISP_EP,
    enable_earth,
    enable_eclipse=True, smart_mode=True,
    max_days=400.0, rtol=1e-7, atol=1e-9,
)

print(f"  Transfer time : {res_earth.t_transfer/86400:.1f} days")
print(f"  ΔV            : {res_earth.delta_v:.0f} m/s")
print(f"  Propellant    : {res_earth.m_prop:.1f} kg")
print(f"  Mass at SOI   : {res_earth.mass[-1]:.1f} kg")
print(f"  Converged     : {res_earth.converged}")

checks_earth = validate_earth_phase(res_earth.delta_v, res_earth.t_transfer / 86400.0)
print_validation_report(checks_earth, "Earth low-thrust phase")

# ============================================================
# PHASE 2 — Moon approach:  SOI → high elliptic lunar orbit
# ============================================================
print("\n[2/3] Moon approach phase  (SOI → high lunar orbit) ...")

mass_at_soi = res_earth.mass[-1]

# Entry into Moon SOI as a high elliptic orbit
a_entry   = 50_000e3          # very large semi-major axis [m]
e_entry   = 0.93
inc_entry = np.radians(15.0)
COE_MOON_IN  = np.array([a_entry, e_entry, inc_entry, 0.0, 0.0, 0.0])

# Target: bring periapsis down to ~5000 km altitude for CR3BP handoff
a_moon_tgt  = 20_000e3
e_moon_tgt  = 0.75
inc_moon_tgt= np.radians(90.0)
COE_MOON_TGT = np.array([a_moon_tgt, e_moon_tgt, inc_moon_tgt, 0.0, 0.0, 0.0])

enable_moon = ControlWeights(ka=1.0, ke=1.0, ki=0.0, kw=0.0, kraan=0.0)

res_moon = propagate_moon_phase(
    COE_MOON_IN, COE_MOON_TGT, mass_at_soi,
    N_THR, THRUST_N, ISP_EP,
    enable_moon,
    max_days=60.0, rtol=1e-7, atol=1e-9,
)

print(f"  Transfer time : {res_moon.t_transfer/86400:.1f} days")
print(f"  ΔV            : {res_moon.delta_v:.0f} m/s")
print(f"  Propellant    : {res_moon.m_prop:.1f} kg")
print(f"  Mass post-EP  : {res_moon.mass[-1]:.1f} kg")

# ============================================================
# PHASE 3 — CR3BP two-impulse:  high orbit → NRHO → LLO
# ============================================================
print("\n[3/3] CR3BP two-impulse phase  (high orbit → NRHO → LLO) ...")

mass_cr3bp = res_moon.mass[-1]
H_LLO_M    = 100_000.0   # target LLO altitude [m]
INC_LLO    = np.pi / 2   # polar LLO

# Get initial guess for the NRHO→LLO transfer
h_llo_nd = H_LLO_M / A_SCALE
guess    = get_initial_guess(h_llo_nd, INC_LLO, 0.0, ta_hint=TA_PERILUNE)
print(f"  Initial guess : strategy={guess.strategy.name}  score={guess.score:.4f}")

# Two-impulse NRHO → LLO (down-leg)
settings = TransferSolverSettings(n_ta_candidates=12, tof_max_cr3bp=5.0)

res_nrho_llo = nrho_to_llo(
    m0=mass_cr3bp,
    isp=ISP_EP,
    h_llo_m=H_LLO_M,
    inc_llo=INC_LLO,
    ta_hint=TA_PERILUNE,
    settings=settings,
)

print(f"  ΔV total      : {res_nrho_llo.dv_total:.1f} m/s")
print(f"  ΔV departure  : {np.linalg.norm(res_nrho_llo.dv1):.1f} m/s  (retrograde at NRHO)")
print(f"  ΔV insertion  : {np.linalg.norm(res_nrho_llo.dv2):.1f} m/s  (circularise at LLO)")
print(f"  Time-of-flight: {res_nrho_llo.time_of_flight/3600:.1f} h  ({res_nrho_llo.time_of_flight/86400:.2f} days)")
print(f"  Periapsis alt : {res_nrho_llo.periapsis_alt_m/1e3:.0f} km  (target {H_LLO_M/1e3:.0f} km)")
print(f"  Fuel used     : {res_nrho_llo.fuel_mass_kg:.1f} kg")

jac_drift = jacobi_max_drift(res_nrho_llo.trajectory)
print(f"  Jacobi drift  : {jac_drift:.2e}  (coast arc quality, < 1e-6 = good)")

# --- Literature validation ---
checks_nrho = validate_nrho_llo(
    res_nrho_llo.dv_total, res_nrho_llo.time_of_flight / 86400.0, H_LLO_M
)
checks_tsiol = validate_tsiolkovsky(
    mass_cr3bp, res_nrho_llo.dv_total, ISP_EP, res_nrho_llo.fuel_mass_kg
)
print_validation_report(checks_nrho + checks_tsiol, "NRHO→LLO transfer (EP)")

# LLO → NRHO return (time-reversal fast path)
res_llo_nrho = llo_to_nrho(
    m0=mass_cr3bp - res_nrho_llo.fuel_mass_kg,
    isp=ISP_EP,
    h_llo_m=H_LLO_M,
    dv_from_downleg=res_nrho_llo.dv_total,
    tof_from_downleg=res_nrho_llo.time_of_flight,
    ta_hint=res_nrho_llo.departure_ta,
)
print(f"\n  Return leg (LLO→NRHO, time-reversal):")
print(f"    ΔV           : {res_llo_nrho.dv_total:.1f} m/s")
print(f"    Fuel used    : {res_llo_nrho.fuel_mass_kg:.1f} kg")

# ΔV corrections
dv_corr, dv_extra = apply_dv_corrections(
    res_nrho_llo.dv_total,
    terminal_dv=doi_dv_ms(H_LLO_M, 15_000.0),
    reserve_frac=0.05,
    phase_dv=10.0,
)
print(f"\n  Corrected ΔV  : {dv_corr:.1f} m/s  (incl. DOI + 5% reserve + phasing)")

# Gateway phasing
phasing = gateway_round_trip_phasing(
    departure_ta_nd=res_nrho_llo.departure_ta,
    downleg_tof_s=res_nrho_llo.time_of_flight,
    descent_tof_s=0.5 * 86400,
    surface_duration_s=14.0 * 86400,
    ascent_tof_s=0.5 * 86400,
)
print(f"\n  Gateway phasing after 14-day surface stay:")
print(f"    Phase offset  : {phasing.phase_offset_fraction*100:.1f}%  of NRHO period")
print(f"    Phase family  : {phasing.phase_family}")
print(f"    Wait for window: {phasing.passive_wait_to_next_window_s/3600:.1f} h")

# ============================================================
# MISSION BUDGET SUMMARY
# ============================================================
total_ep_prop  = res_earth.m_prop + res_moon.m_prop + res_nrho_llo.fuel_mass_kg + res_llo_nrho.fuel_mass_kg
total_dv_ep    = res_earth.delta_v + res_moon.delta_v + res_nrho_llo.dv_total + res_llo_nrho.dv_total
total_days     = (res_earth.t_transfer + res_moon.t_transfer) / 86400.0

summary = f"""
Helios-2 Mock Mission — Full Budget (EP only)
==============================================
Spacecraft (fictitious, Tsiolkovsky sizing only)
  m_dry          : {M_DRY:.0f} kg
  m_prop (EP)    : {M_PROP_EP:.0f} kg
  m0             : {M0:.0f} kg
  Isp (all phases): {ISP_EP:.0f} s  [electric propulsion throughout]
Thruster cluster   : {N_THR} × {THRUST_N*1e3:.0f} mN

Phase 1 — GTO → Moon SOI  (Lyapunov low-thrust EP)
  Transfer time    : {res_earth.t_transfer/86400:.1f} days
  ΔV               : {res_earth.delta_v:.0f} m/s
  Propellant used  : {res_earth.m_prop:.1f} kg

Phase 2 — SOI → high lunar orbit  (low-thrust EP)
  Transfer time    : {res_moon.t_transfer/86400:.1f} days
  ΔV               : {res_moon.delta_v:.0f} m/s
  Propellant used  : {res_moon.m_prop:.1f} kg

Phase 3 — NRHO → LLO  (EP, Isp={ISP_EP:.0f} s)
  ΔV departure     : {np.linalg.norm(res_nrho_llo.dv1):.0f} m/s
  ΔV insertion     : {np.linalg.norm(res_nrho_llo.dv2):.0f} m/s
  ΔV total         : {res_nrho_llo.dv_total:.0f} m/s
  Time-of-flight   : {res_nrho_llo.time_of_flight/3600:.1f} h
  Propellant used  : {res_nrho_llo.fuel_mass_kg:.1f} kg
  Departure TA     : {res_nrho_llo.departure_ta:.3f} CR3BP  (perilune ≈ {TA_PERILUNE:.3f})
  Jacobi drift     : {jac_drift:.2e}

Return leg — LLO → NRHO  (EP, time-reversal, Isp={ISP_EP:.0f} s)
  ΔV total         : {res_llo_nrho.dv_total:.0f} m/s
  Propellant used  : {res_llo_nrho.fuel_mass_kg:.1f} kg

Total mission (EP throughout)
  EP transfer time : {total_days:.1f} days
  Total ΔV         : {total_dv_ep:.0f} m/s
  Total propellant : {total_ep_prop:.1f} kg
  Final mass       : {M0 - total_ep_prop:.1f} kg

Gateway phasing (14-day surface stay)
  Phase offset     : {phasing.phase_offset_fraction*100:.1f}%
  Phase family     : {phasing.phase_family}
  Passive wait     : {phasing.passive_wait_to_next_window_s/3600:.1f} h

NOTE: All values are from a mock (fictitious) spacecraft.
      Phase 3 ΔV is trajectory geometry; propellant is Tsiolkovsky (EP Isp={ISP_EP:.0f} s).
"""
print(summary)

with open("full_mission_summary.txt", "w") as f:
    f.write(summary)
print("Saved: full_mission_summary.txt")

# ============================================================
# PLOTS
# ============================================================

# --- Plot 1: Earth phase orbital elements ---
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Helios-2 — Earth Low-Thrust Phase (GTO → Moon SOI)",
             fontsize=12, fontweight="bold")

t_days = res_earth.t / 86400.0
coe    = res_earth.coe
titles = ["Semi-major axis [km]", "Eccentricity",
          "Inclination [deg]", "Arg. of perigee [deg]",
          "RAAN [deg]", "Mass [kg]"]
data   = [coe[:, 0]/1e3, coe[:, 1], np.degrees(coe[:, 2]),
          np.degrees(coe[:, 3]), np.degrees(coe[:, 4]), res_earth.mass]

for ax, title, d in zip(axes.flat, titles, data):
    ax.plot(t_days, d, lw=1.2, color="steelblue")
    ax.set_xlabel("Time [days]", fontsize=9)
    ax.set_ylabel(title, fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("full_mission_earth_phase.png", dpi=150, bbox_inches="tight")
print("Saved: full_mission_earth_phase.png")

# --- Plot 2: NRHO transfer trajectory + NRHO orbit ---
fig2 = plt.figure(figsize=(10, 8))
ax3d = fig2.add_subplot(111, projection="3d")

# NRHO reference orbit (one full revolution)
nrho_states = []
ta_vals = np.linspace(0, T_0_NRHO, 200)
for ta in ta_vals:
    nrho_states.append(nrho_state_at(ta)[:3])
nrho_states = np.array(nrho_states)

# Convert NRHO to Moon-centred km
nrho_mc = (nrho_states - np.array([1.0 - 0.012150585609624, 0, 0])) * A_SCALE / 1e3

ax3d.plot(nrho_mc[:, 0], nrho_mc[:, 1], nrho_mc[:, 2],
          "k--", lw=1.0, alpha=0.5, label="NRHO (reference)")

# Transfer arc
if res_nrho_llo.trajectory.shape[0] > 1:
    traj_mc = (res_nrho_llo.trajectory[:, :3]
               - np.array([1.0 - 0.012150585609624, 0, 0])) * A_SCALE / 1e3
    ax3d.plot(traj_mc[:, 0], traj_mc[:, 1], traj_mc[:, 2],
              "b-", lw=1.5, label="NRHO → LLO transfer arc")
    ax3d.scatter(*traj_mc[0],  s=60, c="green",  zorder=5, label="Departure (NRHO)")
    ax3d.scatter(*traj_mc[-1], s=60, c="red",    zorder=5, label="Arrival (periapsis)")

# Moon sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
xm = R_MOON_M / 1e3 * np.cos(u) * np.sin(v)
ym = R_MOON_M / 1e3 * np.sin(u) * np.sin(v)
zm = R_MOON_M / 1e3 * np.cos(v)
ax3d.plot_surface(xm, ym, zm, color="silver", alpha=0.4)

ax3d.set_xlabel("X [km]");  ax3d.set_ylabel("Y [km]");  ax3d.set_zlabel("Z [km]")
ax3d.set_title("NRHO → LLO Transfer Arc (Moon-centred frame, mock data)")
ax3d.legend(fontsize=8)
plt.tight_layout()
plt.savefig("full_mission_nrho_transfer.png", dpi=150, bbox_inches="tight")
print("Saved: full_mission_nrho_transfer.png")

# --- Plot 3: ΔV budget bar chart ---
fig3, ax = plt.subplots(figsize=(9, 5))
labels = ["EP: GTO→SOI", "EP: SOI→orbit", "EP: NRHO→LLO", "EP: LLO→NRHO"]
dvs    = [res_earth.delta_v, res_moon.delta_v,
          res_nrho_llo.dv_total, res_llo_nrho.dv_total]
colors = ["steelblue", "cornflowerblue", "tomato", "salmon"]
bars   = ax.bar(labels, dvs, color=colors, edgecolor="k", linewidth=0.6)
for bar, val in zip(bars, dvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("ΔV [m/s]")
ax.set_title("Mission ΔV Budget by Phase — EP throughout (mock Helios-2)")
ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig("full_mission_dv_budget.png", dpi=150, bbox_inches="tight")
print("Saved: full_mission_dv_budget.png")

plt.close("all")
print("\nDone.")
