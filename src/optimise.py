"""
scipy.optimize wrapper for Lyapunov control-weight optimisation.

Tunes the five enable weights  [ka, ke, ki, kw, kraan]  to minimise
propellant consumption for a fixed transfer-time budget, or minimises
transfer time for a fixed propellant budget.

This is the key new scientific contribution over the original MATLAB —
turning a heuristic guidance law into a proper semi-analytical optimisation.

Example
-------
>>> from src.optimise import optimise_weights
>>> result = optimise_weights(
...     coe_initial, coe_target, mass0,
...     n_thrusters=4, thrust=0.01, isp=1600,
...     budget_days=250,
... )
>>> print(result.optimal_weights)
>>> print(result.m_prop_kg, "kg propellant")
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import differential_evolution, OptimizeResult

from .control import ControlWeights
from .propagator import propagate_earth_phase, PropagationResult


@dataclass
class OptimisationResult:
    """Result of the weight optimisation."""
    optimal_weights: ControlWeights
    m_prop_kg:       float           # propellant consumed [kg]
    delta_v_ms:      float           # ΔV [m/s]
    transfer_days:   float           # transfer time [days]
    converged:       bool
    raw:             OptimizeResult  # scipy result object


def optimise_weights(
    coe_initial: np.ndarray,
    coe_target: np.ndarray,
    mass0: float,
    n_thrusters: int,
    thrust_per_thruster: float,
    isp: float,
    *,
    budget_days: float = 300.0,
    enable_mask: np.ndarray | None = None,
    max_iter: int = 80,
    popsize: int = 8,
    seed: int = 42,
    propagator_kwargs: dict | None = None,
) -> OptimisationResult:
    """
    Minimise propellant mass by tuning Lyapunov control weights.

    Parameters
    ----------
    coe_initial   : (6,) initial classical orbital elements [m, -, rad…].
    coe_target    : (6,) target orbital elements.
    mass0         : initial spacecraft mass [kg].
    n_thrusters   : number of thrusters.
    thrust_per_thruster : thrust per thruster [N].
    isp           : specific impulse [s].
    budget_days   : maximum allowed transfer time [days].
    enable_mask   : (5,) bool array indicating which weights to optimise.
                    Default: [True, True, False, True, False]
                    (semi-major axis, eccentricity, inclination, ω, RAAN).
    max_iter      : differential evolution maximum iterations.
    popsize       : population size multiplier.
    seed          : random seed for reproducibility.
    propagator_kwargs : extra kwargs forwarded to ``propagate_earth_phase``.

    Returns
    -------
    OptimisationResult
    """
    if enable_mask is None:
        enable_mask = np.array([True, True, False, True, False])

    if propagator_kwargs is None:
        propagator_kwargs = {}

    # Fixed weights for disabled elements
    fixed_weights = np.array([1.0, 0.0, 0.0, 1.0, 0.0])

    # Bounds for the free weights (0 to 1)
    n_free  = int(np.sum(enable_mask))
    bounds  = [(0.0, 1.0)] * n_free

    def objective(x_free: np.ndarray) -> float:
        """Return propellant consumed; penalise if transfer exceeds budget."""
        w = fixed_weights.copy()
        w[enable_mask] = x_free
        enable = ControlWeights(*w)

        try:
            res: PropagationResult = propagate_earth_phase(
                coe_initial, coe_target, mass0,
                n_thrusters, thrust_per_thruster, isp,
                enable,
                max_days=budget_days,
                **propagator_kwargs,
            )
        except Exception:
            return 1e9   # numerical failure → very bad

        # Heavy penalty if transfer did not reach SOI within budget
        if not res.converged or res.t_transfer > budget_days * 86400.0:
            return mass0  # worst possible propellant (all fuel)

        return res.m_prop

    raw = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max_iter,
        popsize=popsize,
        seed=seed,
        tol=1e-4,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=True,
        disp=False,
    )

    # Reconstruct the best solution
    w_best = fixed_weights.copy()
    w_best[enable_mask] = raw.x
    best_enable = ControlWeights(*w_best)

    final: PropagationResult = propagate_earth_phase(
        coe_initial, coe_target, mass0,
        n_thrusters, thrust_per_thruster, isp,
        best_enable,
        max_days=budget_days,
        **propagator_kwargs,
    )

    return OptimisationResult(
        optimal_weights=best_enable,
        m_prop_kg=final.m_prop,
        delta_v_ms=final.delta_v,
        transfer_days=final.t_transfer / 86400.0,
        converged=final.converged,
        raw=raw,
    )


def sensitivity_analysis(
    coe_initial: np.ndarray,
    coe_target: np.ndarray,
    mass0: float,
    n_thrusters: int,
    thrust_per_thruster: float,
    isp: float,
    weight_grid: np.ndarray | None = None,
    budget_days: float = 300.0,
    propagator_kwargs: dict | None = None,
) -> dict:
    """
    Grid-search sensitivity of propellant mass to the ka / ke weight pair.

    Returns a dict with 'ka_grid', 'ke_grid', 'mprop_grid' suitable for
    a contour plot.
    """
    if weight_grid is None:
        weight_grid = np.linspace(0.0, 1.0, 8)

    if propagator_kwargs is None:
        propagator_kwargs = {}

    Ka, Ke = np.meshgrid(weight_grid, weight_grid)
    Mprop  = np.full_like(Ka, np.nan)

    for i in range(Ka.shape[0]):
        for j in range(Ka.shape[1]):
            enable = ControlWeights(ka=Ka[i, j], ke=Ke[i, j],
                                    ki=0.0, kw=1.0, kraan=0.0)
            try:
                res = propagate_earth_phase(
                    coe_initial, coe_target, mass0,
                    n_thrusters, thrust_per_thruster, isp,
                    enable,
                    max_days=budget_days,
                    **propagator_kwargs,
                )
                Mprop[i, j] = res.m_prop if res.converged else np.nan
            except Exception:
                Mprop[i, j] = np.nan

    return {"ka_grid": Ka, "ke_grid": Ke, "mprop_grid": Mprop}
