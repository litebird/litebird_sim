# -*- encoding: utf-8 -*-
from .distribute import distribute_evenly, distribute_optimally
from .detectors import (
    DetectorInfo,
    FreqChannelInfo,
    InstrumentInfo,
    detector_list_from_parameters,
)
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
from .mbs.mbs import Mbs, MbsParameters, MbsSavedMapInfo
from .mpi import MPI_COMM_WORLD, MPI_ENABLED, MPI_CONFIGURATION
from .observations import Observation
from .quaternions import (
    quat_rotation_x,
    quat_rotation_y,
    quat_rotation_z,
    quat_right_multiply,
    quat_left_multiply,
    rotate_vector,
    rotate_x_vector,
    rotate_y_vector,
    rotate_z_vector,
    all_rotate_vectors,
    all_rotate_x_vectors,
    all_rotate_y_vectors,
    all_rotate_z_vectors,
)
from .scanning import (
    compute_pointing_and_polangle,
    all_compute_pointing_and_polangle,
    spin_to_ecliptic,
    all_spin_to_ecliptic,
    calculate_sun_earth_angles_rad,
    Spin2EclipticQuaternions,
    ScanningStrategy,
    SpinningScanningStrategy,
    get_det2ecl_quaternions,
    get_ecl2det_quaternions,
    get_pointings
)
from .mapping import make_bin_map
from .destriper import DestriperParameters, DestriperResult, destripe
from .simulations import Simulation
from .noise import add_noise
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
    "DetectorInfo",
    "FreqChannelInfo",
    "InstrumentInfo",
    "detector_list_from_parameters",
    # mbs.py
    "Mbs",
    "MbsParameters",
    "MbsSavedMapInfo",
    # mpi.py
    "MPI_COMM_WORLD",
    "MPI_ENABLED",
    "MPI_CONFIGURATION",
    # observations.py
    "Observation",
    # quaternions.py
    "quat_rotation_x",
    "quat_rotation_y",
    "quat_rotation_z",
    "quat_right_multiply",
    "quat_left_multiply",
    "rotate_vector",
    "rotate_x_vector",
    "rotate_y_vector",
    "rotate_z_vector",
    "all_rotate_vectors",
    "all_rotate_x_vectors",
    "all_rotate_y_vectors",
    "all_rotate_z_vectors",
    # scanning.py
    "compute_pointing_and_polangle",
    "all_compute_pointing_and_polangle",
    "spin_to_ecliptic",
    "all_spin_to_ecliptic",
    "calculate_sun_earth_angles_rad",
    "Spin2EclipticQuaternions",
    "ScanningStrategy",
    "SpinningScanningStrategy",
    "get_det2ecl_quaternions",
    "get_ecl2det_quaternions",
    "get_pointings",
    # mapping.py
    "make_bin_map",
    # destripe.py
    "DestriperParameters",
    "DestriperResult",
    "destripe",
    # simulations.py
    "Simulation",
    # noise.py
    "add_noise",
]
