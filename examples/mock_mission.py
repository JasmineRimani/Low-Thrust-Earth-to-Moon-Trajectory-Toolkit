"""
Mock mission example: Low-thrust Earth-to-Moon transfer
========================================================

Uses a *completely fictitious* small electric-propulsion tug ("Helios-1")
to demonstrate the trajectory propagator and scipy optimiser.

No real vehicle data is included.

Run
---
    python examples/mock_mission.py

Produces
--------
    helios1_earth_phase.png   -- orbital elements vs time (Earth phase)
    helios1_summary.txt       -- key mission metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.constants import MU_EARTH, R_EARTH, R_MOON, MU_MOON
from src.orbital_elements import coe2mee, mee2coe
from src.control import ControlWeights
from src.propagator import (
    propagate_earth_phase,
    propagate_moon_phase,
    make_earth_phase_third_body,
    make_moon_phase_third_body,
)

# ---------------------------------------------------------------------------
# Mock spacecraft: "Helios-1" fictitious electric-propulsion tug
# ---------------------------------------------------------------------------

# Propulsion (fictitious HET-like thruster)
N_THRUSTERS          = 4
THRUST_PER_THRUSTER  = 0.010    # [N]  10 mN each
ISP                  = 1600.0   # [s]
MASS_DRY             = 350.0    # [kg]
MASS_PROPELLANT      = 180.0    # [kg]
MASS0                = MASS_DRY + MASS_PROPELLANT  # 530 kg

# Geometry
S_SP  = 25.0   # solar panel area [m²]
S_SL  = 8.0    # lateral area [m²]
C_R   = 1.8    # reflectivity

# ---------------------------------------------------------------------------
# Fictitious initial orbit: Modified GTO (not a real mission orbit)
# ---------------------------------------------------------------------------
a_i    = 24_500e3 + R_EARTH   # semi-major axis [m]
e_i    = 0.71                 # eccentricity
inc_i  = np.radians(7.0)      # inclination [rad]
w_i    = np.radians(0.0)
raan_i = np.radians(0.0)
nu_i   = np.radians(0.0)

COE_INITIAL = np.array([a_i, e_i, inc_i, w_i, raan_i, nu_i])

# ---------------------------------------------------------------------------
# Target: mean Moon orbit elements (approximate)
# ---------------------------------------------------------------------------
a_f    = 384_400e3            # Moon semi-major axis [m]
e_f    = 0.055
inc_f  = np.radians(5.14)
w_f    = np.radians(0.0)
raan_f = np.radians(0.0)
nu_f   = np.radians(0.0)

COE_TARGET_EARTH = np.array([a_f, e_f, inc_f, w_f, raan_f, nu_f])

# ---------------------------------------------------------------------------
# Control weights (Earth phase)
# ---------------------------------------------------------------------------
enable_earth = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)

# ---------------------------------------------------------------------------
# Run Earth-phase propagation
# ---------------------------------------------------------------------------
print("Running Earth-phase propagation ...")
print(f"  Initial mass  : {MASS0:.1f} kg")
print(f"  Total thrust  : {N_THRUSTERS * THRUST_PER_THRUSTER * 1e3:.1f} mN")
print(f"  Isp           : {ISP:.0f} s")

third_body_earth = make_earth_phase_third_body(t0_phase=0.0)

result_earth = propagate_earth_phase(
    COE_INITIAL,
    COE_TARGET_EARTH,
    MASS0,
    N_THRUSTERS,
    THRUST_PER_THRUSTER,
    ISP,
    enable_earth,
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

print(f"\n  Earth phase complete:")
print(f"    Transfer time : {result_earth.t_transfer / 86400:.1f} days")
print(f"    ΔV            : {result_earth.delta_v:.1f} m/s")
print(f"    Propellant    : {result_earth.m_prop:.1f} kg")
print(f"    Final mass    : {result_earth.mass[-1]:.1f} kg")
print(f"    Converged     : {result_earth.converged}")

# ---------------------------------------------------------------------------
# Moon-phase propagation  (circularisation to NHRO-like orbit)
# ---------------------------------------------------------------------------

# Entry state at Moon SOI (use last Earth-phase state)
# Convert to Moon-centred frame (approximate — full patched conic would use SPICE)
final_coe_earth = result_earth.coe[-1]
mass_at_soi     = result_earth.mass[-1]

# Fictitious NHRO target (simplified)
a_nhro    = 5_000e3 + R_MOON    # 5000 km altitude circular
e_nhro    = 0.0
inc_nhro  = np.radians(90.0)
w_nhro    = np.radians(270.0)
raan_nhro = np.radians(90.0)
nu_nhro   = np.radians(0.0)

COE_TARGET_MOON  = np.array([a_nhro, e_nhro, inc_nhro, w_nhro, raan_nhro, nu_nhro])

# Entry conditions at Moon SOI (highly elliptic orbit around Moon)
a_entry   = 60_000e3   # large ellipse [m]
e_entry   = 0.95
inc_entry = np.radians(10.0)
COE_MOON_INITIAL = np.array([a_entry, e_entry, inc_entry, 0.0, 0.0, 0.0])

enable_moon = ControlWeights(ka=1.0, ke=1.0, ki=0.0, kw=0.0, kraan=0.0)

print("\nRunning Moon-phase propagation ...")
third_body_moon = make_moon_phase_third_body()

result_moon = propagate_moon_phase(
    COE_MOON_INITIAL,
    COE_TARGET_MOON,
    mass_at_soi,
    N_THRUSTERS,
    THRUST_PER_THRUSTER,
    ISP,
    enable_moon,
    S_sp=S_SP,
    c_r=C_R,
    max_days=150.0,
    rtol=1e-7,
    atol=1e-9,
    get_third_body=third_body_moon,
)

print(f"  Moon phase complete:")
print(f"    Transfer time : {result_moon.t_transfer / 86400:.1f} days")
print(f"    ΔV            : {result_moon.delta_v:.1f} m/s")
print(f"    Propellant    : {result_moon.m_prop:.1f} kg")
print(f"    Final mass    : {result_moon.mass[-1]:.1f} kg")

# ---------------------------------------------------------------------------
# Plot: Earth phase orbital elements
# ---------------------------------------------------------------------------
t_days = result_earth.t / 86400.0
coe    = result_earth.coe

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Helios-1 Mock Tug — Earth Phase Low-Thrust Transfer",
             fontsize=13, fontweight="bold")

labels = ["Semi-major axis [km]", "Eccentricity [-]", "Inclination [deg]",
          "Arg. of perigee [deg]", "RAAN [deg]", "Mass [kg]"]
data   = [coe[:, 0] / 1e3, coe[:, 1], np.degrees(coe[:, 2]),
          np.degrees(coe[:, 3]), np.degrees(coe[:, 4]), result_earth.mass]

for ax, lab, dat in zip(axes.flat, labels, data):
    ax.plot(t_days, dat, linewidth=1.2)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(lab)
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("helios1_earth_phase.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: helios1_earth_phase.png")

# ---------------------------------------------------------------------------
# Plot: 3-D trajectory
# ---------------------------------------------------------------------------
fig3d = plt.figure(figsize=(9, 9))
ax3d  = fig3d.add_subplot(111, projection="3d")

r = result_earth.r_eci / 1e6   # in 1000 km
ax3d.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=0.8, label="Transfer trajectory")

# Earth sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
xe = R_EARTH / 1e6 * np.cos(u) * np.sin(v)
ye = R_EARTH / 1e6 * np.sin(u) * np.sin(v)
ze = R_EARTH / 1e6 * np.cos(v)
ax3d.plot_surface(xe, ye, ze, color="steelblue", alpha=0.3)

ax3d.set_xlabel("X [1000 km]")
ax3d.set_ylabel("Y [1000 km]")
ax3d.set_zlabel("Z [1000 km]")
ax3d.set_title("Low-thrust Earth → Moon SOI trajectory (mock data)")
ax3d.legend()
plt.tight_layout()
plt.savefig("helios1_trajectory_3d.png", dpi=150, bbox_inches="tight")
print("Plot saved: helios1_trajectory_3d.png")

# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------
total_prop = result_earth.m_prop + result_moon.m_prop
total_days = (result_earth.t_transfer + result_moon.t_transfer) / 86400.0
total_dv   = result_earth.delta_v + result_moon.delta_v

summary = f"""
Helios-1 Mock Mission Summary
==============================
Initial mass      : {MASS0:.1f} kg
Dry mass          : {MASS_DRY:.1f} kg
Thruster cluster  : {N_THRUSTERS} x {THRUST_PER_THRUSTER*1e3:.0f} mN  (Isp = {ISP:.0f} s)

Earth phase
-----------
  Transfer time   : {result_earth.t_transfer/86400:.1f} days
  Propellant      : {result_earth.m_prop:.1f} kg
  ΔV              : {result_earth.delta_v:.1f} m/s

Moon phase
----------
  Transfer time   : {result_moon.t_transfer/86400:.1f} days
  Propellant      : {result_moon.m_prop:.1f} kg
  ΔV              : {result_moon.delta_v:.1f} m/s

Total mission
-------------
  Transfer time   : {total_days:.1f} days
  Total propellant: {total_prop:.1f} kg
  Total ΔV        : {total_dv:.1f} m/s
  Final mass      : {result_moon.mass[-1]:.1f} kg

NOTE: All values are from a mock (fictitious) launcher.
      No real mission or ESA vehicle data is represented.
"""

print(summary)
with open("helios1_summary.txt", "w") as fh:
    fh.write(summary)
print("Summary saved: helios1_summary.txt")
