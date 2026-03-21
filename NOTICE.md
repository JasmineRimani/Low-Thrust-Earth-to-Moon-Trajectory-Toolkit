# Repository Notice

This repository is being maintained as a public, clean-room Python rewrite
inspired by published orbital-mechanics literature.

Intended public scope:

- low-thrust Earth-to-Moon mission analysis
- perturbation modelling
- guidance-law experimentation
- optimisation with `scipy.optimize`

Current status:
- The supported workflow is the first transfer leg (`GTO -> Moon SOI`) and its
  optimisation utilities.
- Moon-phase / NRHO extension code is kept in the repository for ongoing work,
  but it should be treated as experimental and not yet a finished supported
  feature.
