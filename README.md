# BES design tools for W7-X

`w7x_bes_tools` is a Python package for BES diagnostic design tools at W7-X

- `beams.py` - class `HeatingBeam` represents the geometry of W7-X heating beams
- `sightline.py` - class `Sightline` represents the geometry and beam-weighted localization of a ray passing through a beam volume
- `sightline_grid.py` - class `Grid` represents a grid of sightline rays
- `fida.py` - class `Fida` represents beam emission spectra from FIDASIM calculations
- `profiles.py` - class `Profile` represents n_e, T_e, and T_i profile fits for MPTS, CERS, and XICS measurements
- `detectors.py` - classes `PinDiode`, `ApdDiode`, and `TIA` represent operational and noise characteristics for PIN diodes, APDs, and transimpedance amplifiers
- `signals.py`
- `optics.py`
