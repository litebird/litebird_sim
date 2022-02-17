# -*- encoding: utf-8 -*-

from .compress import (
    rle_compress,
    rle_decompress,
)
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
from .hwp_sys.hwp_sys import HwpSys
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
    get_pointings,
)
from .mapping import make_bin_map
from .destriper import DestriperParameters, DestriperResult, destripe
from .simulations import Simulation
from .noise import (
    add_white_noise,
    add_one_over_f_noise,
    add_noise,
    add_noise_to_observations,
)
from .scan_map import scan_map, scan_map_in_observations
from .coordinates import DEFAULT_COORDINATE_SYSTEM, DEFAULT_TIME_SCALE
from .spacecraft import (
    compute_l2_pos_and_vel,
    compute_lissajous_pos_and_vel,
    spacecraft_pos_and_vel,
    SpacecraftOrbit,
    SpacecraftPositionAndVelocity,
)
from .dipole import add_dipole, add_dipole_to_observations, DipoleType
from .io import (
    write_list_of_observations,
    write_observations,
    read_list_of_observations,
    read_observations,
)
from .version import __author__, __version__

__all__ = [
    "__author__",
    "__version__",
    # compress.py
    "rle_compress",
    "rle_decompress",
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
    "add_white_noise",
    "add_one_over_f_noise",
    "add_noise",
    "add_noise_to_observations",
    # scan_map.py
    "scan_map",
    "scan_map_in_observations",
    # dipole.py
    "add_dipole",
    "add_dipole_to_observations",
    "DipoleType",
    # coordinates.py
    "DEFAULT_COORDINATE_SYSTEM",
    "DEFAULT_TIME_SCALE",
    # spacecraft.py
    "compute_l2_pos_and_vel",
    "compute_lissajous_pos_and_vel",
    "spacecraft_pos_and_vel",
    "SpacecraftOrbit",
    "SpacecraftPositionAndVelocity",
    # io
    "write_list_of_observations",
    "write_observations",
    "read_list_of_observations",
    "read_observations",
]
