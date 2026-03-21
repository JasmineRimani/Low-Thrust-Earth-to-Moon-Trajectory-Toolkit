"""
Circular Restricted Three-Body Problem (CR3BP) utilities.

Earth-Moon rotating frame, dimensionless units.

Normalisation
-------------
Length scale  A_SCALE = 384 400 km  (mean Earth-Moon distance)
Time scale    T_SCALE = 1 / (mean motion of Moon) ≈ 375 190 s
Mass scale    μ_E + μ_M

All positions/velocities in this module are dimensionless unless noted.

Reference
---------
Vallado (2013) §2; Parker & Anderson (2014) §2; Zimovan-Spreen et al. (2020).
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CR3BP constants  (SI-consistent, dimensionless system)
# ---------------------------------------------------------------------------

# Earth-Moon mass parameter
MU_M: float = 0.012150585609624      # Moon / (Earth + Moon)  [-]
MU_E: float = 1.0 - MU_M            # Earth / (Earth + Moon) [-]

# Scale factors
A_SCALE: float = 384_400_000.0       # Length unit [m]  (mean E-M distance)
T_SCALE: float = 375_190.258         # Time unit   [s]  (1 / mean Moon motion)
V_SCALE: float = A_SCALE / T_SCALE   # Velocity unit [m/s]

# Moon physical parameters
R_MOON: float  = 1_737_400.0 / A_SCALE   # Moon radius [CR3BP length]
MU_MOON_SI: float = MU_M * A_SCALE**3 / T_SCALE**2  # [m³/s²]
R_MOON_M: float = 1_737_400.0              # [m]

# Fixed body positions in rotating frame (barycentric)
P_M_VEC: np.ndarray = np.array([1.0 - MU_M, 0.0, 0.0])   # Moon
P_E_VEC: np.ndarray = np.array([-MU_M,      0.0, 0.0])   # Earth

# Angular velocity of rotating frame
OMEGA0: float = 1.0
OMEGA0_VEC: np.ndarray = np.array([0.0, 0.0, 1.0])

# ---------------------------------------------------------------------------
# 9:2 synodic-resonant NRHO reference orbit
#
# Reference state P_0_NRHO is at apolune (farthest point from Moon).
# Period T_0_NRHO is computed numerically from this seed.
#
# These values are taken from published NRHO literature
# (Zimovan-Spreen et al. 2020, Table 1; Capdevila 2014).
# They are open-publication orbital mechanics data.
# ---------------------------------------------------------------------------

P_0_NRHO: np.ndarray = np.array([
    1.021881345465263,    # x  [CR3BP]
    0.0,                  # y
   -0.182096761524240,    # z
    0.0,                  # vx
   -0.103270459010000,    # vy
    0.0,                  # vz
])

# Approximate period — refined by numerical search below
_T_0_NRHO_APPROX: float = 1.6323    # [CR3BP time]

# Canonical reference period (9:2 synodic resonance)
REFERENCE_92_PERIOD_DAYS: float = 6.5628  # [days]  from literature
REFERENCE_92_PERIOD_S: float = REFERENCE_92_PERIOD_DAYS * 86400.0
REFERENCE_92_PERIOD_ND: float = REFERENCE_92_PERIOD_S / T_SCALE

# ---------------------------------------------------------------------------
# Integration step sizes  [CR3BP time]
# ---------------------------------------------------------------------------

DT_COARSE: float = 0.05    # grid search
DT_TRAJ:   float = 0.02    # trajectory visualisation
DT_STM:    float = 0.10    # STM propagation (output grid only)

_RTOL:   float = 1e-10
_ATOL:   float = 1e-12
_NSTEPS: int   = 10_000

# ---------------------------------------------------------------------------
# CR3BP dynamics  (no STM)
# ---------------------------------------------------------------------------

def cr3bp_accel(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    CR3BP acceleration in the Earth-Moon rotating frame (dimensionless).

    Includes two-body gravity, centrifugal, and Coriolis.
    """
    d_M = r - P_M_VEC;  rM3 = np.dot(d_M, d_M) ** 1.5
    d_E = r - P_E_VEC;  rE3 = np.dot(d_E, d_E) ** 1.5
    grav        = -(MU_M / rM3) * d_M - (MU_E / rE3) * d_E
    centrifugal = np.array([r[0], r[1], 0.0])
    coriolis    = 2.0 * np.array([v[1], -v[0], 0.0])
    return grav + centrifugal + coriolis


def dyn_no_stm(t: float, p: np.ndarray) -> np.ndarray:
    """CR3BP state derivative [vx, vy, vz, ax, ay, az] without STM."""
    r, v = p[:3], p[3:]
    a = cr3bp_accel(r, v)
    return np.array([v[0], v[1], v[2], a[0], a[1], a[2]])


def jacobi_constant(state: np.ndarray) -> float:
    """
    CR3BP Jacobi constant  C = 2Ω − v²  (conserved on unforced arcs).

    Useful as an integration-accuracy diagnostic.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (6,):
        raise ValueError("jacobi_constant expects shape (6,).")
    r, v = state[:3], state[3:]
    d_M = r - P_M_VEC;  d_E = r - P_E_VEC
    omega = (0.5 * (r[0]**2 + r[1]**2)
             + MU_M / np.linalg.norm(d_M)
             + MU_E / np.linalg.norm(d_E))
    return float(2.0 * omega - np.dot(v, v))


def jacobi_max_drift(states: np.ndarray) -> float:
    """Max absolute Jacobi drift along a coast arc — integration-quality indicator."""
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 1:
        return 0.0
    C = np.array([jacobi_constant(row) for row in arr])
    return float(np.max(np.abs(C - C[0])))


# ---------------------------------------------------------------------------
# CR3BP dynamics  (with 6×6 STM)
# ---------------------------------------------------------------------------

def dyn_stm(t: float, S: np.ndarray) -> np.ndarray:
    """
    CR3BP equations of motion augmented with the State Transition Matrix.

    State vector S = [state(6), STM_flat(36)] → 42 elements.
    dSTM/dt = J(t) @ STM,  where J is the 6×6 Jacobian.
    """
    r, v = S[:3], S[3:6]
    d_M = r - P_M_VEC;  d_E = r - P_E_VEC

    rM2 = np.dot(d_M, d_M);  rM3 = rM2**1.5;  rM5 = rM2**2.5
    rE2 = np.dot(d_E, d_E);  rE3 = rE2**1.5;  rE5 = rE2**2.5

    a = (-(MU_M / rM3) * d_M - (MU_E / rE3) * d_E
         + np.array([r[0], r[1], 0.0])
         + 2.0 * np.array([v[1], -v[0], 0.0]))

    DM  = d_M[:, None] * d_M[None, :]
    DE  = d_E[:, None] * d_E[None, :]
    Jg  = ((3.0 * MU_M / rM5) * DM - (MU_M / rM3) * np.eye(3)
           + (3.0 * MU_E / rE5) * DE - (MU_E / rE3) * np.eye(3))

    J = np.zeros((6, 6))
    J[:3, 3:]  = np.eye(3)
    J[3:, :3]  = Jg + np.diag([1.0, 1.0, 0.0])
    J[3:, 3:]  = np.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    STM  = S[6:42].reshape(6, 6)
    dSTM = J @ STM

    F = np.zeros(42)
    F[:6]  = np.array([v[0], v[1], v[2], a[0], a[1], a[2]])
    F[6:]  = dSTM.ravel()
    return F


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

def integrate_cr3bp(
    dt: float,
    t_end: float,
    y0: np.ndarray,
    fun=None,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate a CR3BP ODE with fixed output step ``dt``.

    Uses scipy's dopri5 (RK45) with tight tolerances.  The last output point
    is always at exactly ``t_end`` so time-of-flight gradients are smooth.

    Parameters
    ----------
    dt    : output time step [CR3BP time].
    t_end : total integration time [CR3BP time].
    y0    : initial state.
    fun   : ODE function  f(t, y).  Defaults to ``dyn_no_stm``.
    rtol  : relative tolerance.
    atol  : absolute tolerance.

    Returns
    -------
    t_arr : (N,)   time array [CR3BP time].
    y_arr : (N, d) state history.
    """
    if fun is None:
        fun = dyn_no_stm

    dim = len(y0)

    if t_end == 0.0:
        return np.array([0.0]), y0[None, :]

    n_interior = max(int(np.floor(abs(t_end) / dt)), 0)
    n_alloc    = n_interior + 2
    t_arr = np.zeros(n_alloc)
    y_arr = np.zeros((n_alloc, dim))
    y_arr[0] = y0

    solver = ode(fun)
    solver.set_integrator("dopri5", rtol=rtol, atol=atol,
                          nsteps=_NSTEPS, first_step=1e-4)
    solver.set_initial_value(y0, 0.0)

    last = 0
    for i in range(1, n_interior + 1):
        solver.integrate(solver.t + dt)
        if not solver.successful():
            logger.debug(f"CR3BP integrator step {i} failed at t={solver.t:.4f}")
            break
        t_arr[i] = solver.t
        y_arr[i] = solver.y
        last = i

    # Final sub-step to exactly t_end
    if solver.successful() and abs(t_end - solver.t) > 1e-14 * max(abs(t_end), 1.0):
        solver.integrate(t_end)
        if solver.successful():
            last += 1
            t_arr[last] = solver.t
            y_arr[last] = solver.y

    return t_arr[:last + 1], y_arr[:last + 1]


def propagate_cr3bp(
    p0: np.ndarray,
    tof: float,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate CR3BP state using scipy solve_ivp (DOP853).

    Returns (t_arr [N], states [N, 6]).
    Preferred over ``integrate_cr3bp`` for long arcs (no nsteps limit).
    """
    sol = solve_ivp(
        dyn_no_stm, [0.0, tof], p0,
        method="DOP853",
        rtol=rtol, atol=atol,
        dense_output=False,
        max_step=DT_TRAJ,
    )
    if not sol.success:
        raise RuntimeError(
            f"propagate_cr3bp: DOP853 failed — {sol.message}"
        )
    return sol.t, sol.y.T


# ---------------------------------------------------------------------------
# NRHO propagation
# ---------------------------------------------------------------------------

def _compute_nrho_period(p0: np.ndarray, t_approx: float) -> float:
    """
    Refine the NRHO period numerically by minimising |y-crossing residual|.

    Searches in [0.9, 1.1] × t_approx for the first y=0 crossing after half-period.
    """
    half = t_approx / 2.0

    def _ycross(t):
        _, y = integrate_cr3bp(DT_COARSE, t, p0)
        return abs(y[-1, 1])   # y-coordinate residual

    res = minimize_scalar(_ycross, bounds=(0.8 * half, 1.2 * half), method="bounded")
    return float(2.0 * res.x)


# Numerically refined NRHO period
T_0_NRHO: float = _compute_nrho_period(P_0_NRHO, _T_0_NRHO_APPROX)


def nrho_state_at(ta: float) -> np.ndarray:
    """
    Propagate the reference NRHO to epoch ta [CR3BP time]. Returns state[6].

    Smooth for all ta > 0 (required when TA is an optimisation variable).
    """
    ta_mod = ta % T_0_NRHO
    if ta_mod == 0.0:
        return P_0_NRHO.copy()
    _, y = integrate_cr3bp(DT_COARSE, ta_mod, P_0_NRHO)
    return y[-1].copy()


def _compute_perilune_epoch() -> float:
    """Return the CR3BP epoch of closest Moon approach within one NRHO revolution."""
    def _dist(ta):
        s = nrho_state_at(ta)
        return np.linalg.norm(s[:3] - P_M_VEC)
    res = minimize_scalar(_dist, bounds=(0.1, T_0_NRHO - 0.1), method="bounded")
    return float(res.x)


TA_PERILUNE: float = _compute_perilune_epoch()
