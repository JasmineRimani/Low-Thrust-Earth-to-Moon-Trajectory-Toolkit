# Low-Thrust Earth-to-Moon Mission Analysis Toolkit

Python implementation of a low-thrust trajectory propagator for electric-propulsion
spacecraft transferring from GTO toward the Moon, with `scipy.optimize`
weight tuning.

This repository is a clean-room Python rewrite based on the public paper:

> Rimani et al., *"MAGNETO: Electric Propulsion Tug for Lunar Cargo Delivery"*,
> Acta Astronautica, DOI `10.1016/j.actaastro.2020.05.045`

---

## Scope

Currently, the supported workflow focuses on the first transfer leg:

- **GTO → Moon SOI** low-thrust propagation
- **Lyapunov control-weight optimisation** for the Earth-phase transfer

This is built using:

- **Modified Equinoctial Elements** (MEE) as the state vector (singularity-free)
- **Lyapunov feedback control** (Ruggiero et al. 2011) for thrust direction
- **scipy.integrate.solve_ivp DOP853** adaptive integrator (mass as 7th state)
- Perturbations: J2, third-body (Moon + Sun), solar radiation pressure, drag

And optionally optimises the Lyapunov control weights with `scipy.optimize.differential_evolution`.

The repository also contains Moon-phase and CR3BP / NRHO extension code under
`src/` and `src/cr3bp/`. That work is still in progress and is not yet treated
as a supported end-to-end NRHO transfer workflow.

## Provenance And Boundaries

- This repository is intended to contain only a public-literature-inspired Python implementation.
- Experimental NRHO extension code is kept for ongoing development, but it is not yet the supported product surface.

---

## Improvements over the original MATLAB

| Issue | Fix |
|-------|-----|
| `abs()` on third-body vector destroyed direction | Removed — vector passed directly |
| Solar angle hardcoded to π/2 | Computed from actual Sun–SC geometry |
| Eclipse model was 2D only | Full 3-D conical shadow model |
| Mass updated outside integrator | Mass is 7th ODE state variable |
| Custom RKF7(8) with fixed outer step | `solve_ivp(DOP853)` fully adaptive |
| No optimisation of control weights | `scipy.optimize.differential_evolution` |

---

## Installation

```bash
pip install numpy scipy matplotlib
# Optional for real ephemerides:
pip install spiceypy
```

---

## Quick start

```python
import numpy as np
from src.constants import R_EARTH
from src.control import ControlWeights
from src.propagator import propagate_earth_phase

coe_initial = np.array([24_500e3 + R_EARTH, 0.71, np.radians(7), 0, 0, 0])
coe_target  = np.array([384_400e3, 0.055, np.radians(5.14), 0, 0, 0])
enable      = ControlWeights(ka=1.0, ke=0.0, ki=0.0, kw=1.0, kraan=0.0)

result = propagate_earth_phase(
    coe_initial, coe_target,
    mass0=530.0, n_thrusters=4,
    thrust_per_thruster=0.01, isp=1600.0,
    enable=enable, max_days=400.0,
)
print(f"Transfer: {result.t_transfer/86400:.0f} days, ΔV: {result.delta_v:.0f} m/s")
```

---

## Optimise control weights

```python
from src.optimise import optimise_weights

opt = optimise_weights(
    coe_initial, coe_target, mass0=530.0,
    n_thrusters=4, thrust_per_thruster=0.01, isp=1600.0,
    budget_days=350.0,
)
print(opt.optimal_weights)
print(f"Propellant: {opt.m_prop_kg:.1f} kg")
```

---

## Repository layout

```
src/
  constants.py          Physical constants (SI)
  orbital_elements.py   COE ↔ MEE ↔ ECI conversions
  perturbations.py      J2, third-body, SRP, drag, eclipse
  control.py            Lyapunov guidance law
  equations_of_motion.py  MEE ODEs (Earth + Moon phase)
  propagator.py         solve_ivp wrappers + mock ephemeris
  optimise.py           scipy.optimize weight tuner
  plotting.py           shared plotting helpers for example scripts
  cr3bp/               experimental NRHO extension code
examples/
  mock_mission.py       Supported Earth-phase worked example
tests/
  test_magneto.py       Core propagation / perturbation / optimisation tests
```

Example scripts save plots and summaries into `outputs/` so the repository root
stays tidy.

The NRHO-related dynamics and transfer files are currently experimental extension
work and should not be treated as a finished implemented feature yet.

See [NOTICE.md](/home/jasmi/research_reputation/personal_github/MAGNETO_Core/NOTICE.md) for the repository cleanup intent and scope.

---

## Third-body ephemeris

By default, circular-Keplerian mock functions are used for Moon and Sun positions.
To use real JPL ephemerides, install `spiceypy`, load DE430 kernels, and pass a
`get_third_body` callable to the propagators.

---

## Note on mock data

The example spacecraft data used in the public examples is **completely
fictitious**. No ESA, ESTEC, or proprietary vehicle data is intended to be
included.
