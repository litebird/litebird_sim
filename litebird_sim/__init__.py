# -*- encoding: utf-8 -*-

from .distribute import distribute_evenly, distribute_optimally
from .detectors import Detector
from .healpix import (
    nside_to_npix,
    npix_to_nside,
    is_npix_ok,
    map_type,
    get_pixel_format,
    write_healpix_map_to_hdu,
    write_healpix_map_to_file,
)
from .mpi import MPI_COMM_WORLD, MPI_ENABLED, MPI_CONFIGURATION
from .observations import Observation
from .simulations import Simulation
from .version import __author__, __version__

__all__ = [
    "__author__",
    "__version__",
    # healpix.py
    "nside_to_npix",
    "npix_to_nside",
    "is_npix_ok",
    "map_type",
    "get_pixel_format",
    "write_healpix_map_to_hdu",
    "write_healpix_map_to_file",
    # distribute.py
    "distribute_evenly",
    "distribute_optimally",
    # mpi.py
    "MPI_COMM_WORLD",
    "MPI_ENABLED",
    "MPI_CONFIGURATION",
    # observations.py
    "Observation",
    # detectors.py
    "Detector",
    # simulations.py
    "Simulation",
]
