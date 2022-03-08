# BES design tools for W7-X

`w7x_bes_tools/` is a Python package for BES diagnostic design tools at W7-X

- `beams.py`
    - Class `HeatingBeam` represents the geometry of W7-X heating beams.  Class attributes include a VMEC equilibrium and locations for nearby ports.  Class methods include plotting routines.
- `sightline.py`
    - Class `Sightline` represents the geometry and beam-weighted localization of a ray passing through a beam volume.
- `sightline_grid.py`
    - Class `Grid` represents a grid of sightline rays to capture spatial and k-space coverage.
- `fida.py`
    - Class `Fida` represents beam emission spectra from FIDASIM calculations.
- `profiles.py`
    - Class `Profile` represents n_e, T_e, and T_i profile fits for MPTS, CERS, and XICS measurements.
- `detectors.py`
    - Classes `PinDiode`, `ApdDiode`, and `TIA` represent operational and noise characteristics for PIN diodes, APDs, and transimpedance amplifiers.
- `signals.py`
- `optics.py`
- `utilities/`
    - Helper modules

`notebooks/` contains analysis code and Juptyer notebooks.