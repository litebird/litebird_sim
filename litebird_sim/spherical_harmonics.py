import warnings
from litebird_sim.maps_and_harmonics import SphericalHarmonics

# Raise a warning when this module is accessed
warnings.warn(
    "The 'litebird_sim.spherical_harmonics' module is deprecated and will be removed "
    "in a future version. Please update your scripts or re-save your data to use "
    "'litebird_sim.maps_and_harmonics'.",
    DeprecationWarning,
    stacklevel=2,
)

# TODO: Remove this module once all legacy MDR2 production files are migrated.
# This alias allows pickle (np.load) to find the class in its old location.
SphericalHarmonics = SphericalHarmonics
