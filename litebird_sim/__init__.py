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
from .mpi import HAVE_MPI4PY, MPI_RANK, MPI_SIZE, MPI_COMM_WORLD, MPI_CONFIGURATION
from .observations import Observation
from .simulations import Simulation

__author__ = "The LiteBIRD simulation team"
__version__ = "0.1.0"
__all__ = [
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
    "HAVE_MPI4PY",
    "MPI_RANK",
    "MPI_SIZE",
    "MPI_CONFIGURATION",
    "MPI_COMM_WORLD",
    # observations.py
    "Observation",
    # detectors.py
    "Detector",
    # simulations.py
    "Simulation",
]
