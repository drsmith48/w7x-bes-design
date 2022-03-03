# BES design tools for W7-X

- `w7x_bes_tools` is a Python module for BES design tools
  - `beams.py` - class `HeatingBeam` represents the geometry of W7-X heating beams
  - `sightline.py` - class `Sightline` represents the geometry and beam-weighted localization of a
  ray passing through a beam volume
  - `sightline_grid.py` - class `Grid` represents a grid of sightline rays
  - `fida.py` - class `Fida` represents the beam emission spectra from FIDASIM calculations
  - `profiles.py` - class `Profile` represents n_e, T_e, and T_i profile fits for MPTS and CERS data
- `notebooks` contains Jupyter notebooks
