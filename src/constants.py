"""
Astrodynamic and physical constants.

All values in SI units (m, s, kg, rad) unless noted.
Sources: IAU 2012, JPL DE430, NIST CODATA.
"""

import numpy as np

# --- General ---
AU        = 149_597_870_691.0      # Astronomical unit [m]
C_LIGHT   = 299_792_458.0          # Speed of light [m/s]
P_SOLAR   = 4.51e-6                # Solar radiation pressure at 1 AU [N/m²]
PSR       = 1366.1                 # Solar flux at 1 AU [W/m²]

# --- Earth ---
MU_EARTH        = 3.986004415e14   # Gravitational parameter [m³/s²]
R_EARTH         = 6_378_136.3      # Equatorial radius [m]
J2_EARTH        = 1.08263e-3       # Oblateness coefficient [-]
G               = 9.80665          # Surface gravity [m/s²]
OMEGA_EARTH     = 7.292115486e-5   # Inertial rotation rate [rad/s]
R_SOI_MOON      = 66_200_000.0     # Moon sphere of influence radius [m]

# --- Moon ---
MU_MOON         = 4.904869590e12   # Gravitational parameter [m³/s²]
R_MOON          = 1_738_000.0      # Equatorial radius [m]
J2_MOON         = 2.0e-6           # Oblateness coefficient [-]
G_MOON          = 1.625            # Surface gravity [m/s²]
OMEGA_MOON      = 2.649e-6         # Angular velocity in Earth-Moon system [rad/s]

# --- Sun ---
MU_SUN          = 1.32712440041e20 # Gravitational parameter [m³/s²]
R_SUN           = 696_000_000.0    # Equatorial radius [m]
