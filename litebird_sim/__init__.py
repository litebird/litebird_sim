# -*- encoding: utf-8 -*-

from .distribute import distribute_evenly, distribute_optimally
from .detectors import Detector, read_detector_from_dict, read_detector_from_imo
from .healpix import (
    nside_to_npix,
    npix_to_nside,
    is_npix_ok,
    map_type,
    get_pixel_format,
    write_healpix_map_to_hdu,
    write_healpix_map_to_file,
)
from .imo import (
    Imo,
    FormatSpecification,
    Entity,
    Quantity,
    Release,
    ImoFormatError,
    ImoFlatFile,
)
from .mpi import MPI_COMM_WORLD, MPI_ENABLED, MPI_CONFIGURATION
from .observations import Observation
from .scanning import (
    qrotation_x,
    qrotation_y,
    qrotation_z,
    quat_multiply,
    rotate_vector,
    all_rotate_vector,
    rotate_z_vector,
    all_rotate_z_vector,
    boresight_to_sun_earth_axis,
    boresight_to_ecliptic,
    all_boresight_to_ecliptic,
    ScanningStrategy,
)
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
    # imo.py
    "Imo",
    # imofile.py
    "ImoFormatError",
    "ImoFlatFile",
    # imoobjects
    "FormatSpecification",
    "Entity",
    "Quantity",
    "Release",
    # detectors.py
    "Detector",
    "read_detector_from_dict",
    "read_detector_from_imo",
    # mpi.py
    "MPI_COMM_WORLD",
    "MPI_ENABLED",
    "MPI_CONFIGURATION",
    # observations.py
    "Observation",
    # detectors.py
    "Detector",
    # scanning.py
    "qrotation_x",
    "qrotation_y",
    "qrotation_z",
    "quat_multiply",
    "rotate_vector",
    "all_rotate_vector",
    "rotate_z_vector",
    "all_rotate_z_vector",
    "boresight_to_sun_earth_axis",
    "boresight_to_ecliptic",
    "all_boresight_to_ecliptic",
    "ScanningStrategy",
    # simulations.py
    "Simulation",
]
