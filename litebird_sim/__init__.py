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
from .bandpasses import BandPassInfo
from .healpix import (
    nside_to_npix,
    npix_to_nside,
    is_npix_ok,
    map_type,
    get_pixel_format,
    write_healpix_map_to_hdu,
    write_healpix_map_to_file,
)
from .hwp import (
    HWP,
    IdealHWP,
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
from .madam import save_simulation_for_madam
from .mbs.mbs import Mbs, MbsParameters, MbsSavedMapInfo
from .mpi import MPI_COMM_WORLD, MPI_ENABLED, MPI_CONFIGURATION
from .observations import Observation, TodDescription
from .pointings import (
    apply_hwp_to_obs,
    get_pointing_buffer_shape,
    get_pointings,
    get_pointings_for_observations,
)
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
)
from .mapping import DestriperParameters, DestriperResult, make_bin_map
from .destriper import destripe
from .simulations import (
    Simulation,
    MpiObservationDescr,
    MpiProcessDescr,
    MpiDistributionDescr,
)
from .noise import (
    add_white_noise,
    add_one_over_f_noise,
    add_noise,
    add_noise_to_observations,
)
from .scan_map import scan_map, scan_map_in_observations
from .coordinates import (
    DEFAULT_COORDINATE_SYSTEM,
    DEFAULT_TIME_SCALE,
    ECL_TO_GAL_ROT_MATRIX,
    CoordinateSystem,
)

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

from .gaindrifts import (
    GainDriftType,
    SamplingDist,
    GainDriftParams,
    apply_gaindrift_for_one_detector,
    apply_gaindrift_to_tod,
    apply_gaindrift_to_observations,
)

from .version import __author__, __version__

__all__ = [
    "__author__",
    "__version__",
    # compress.py
    "rle_compress",
    "rle_decompress",
    # bandpasses.py
    "BandPassInfo",
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
    # hwp.py
    "HWP",
    "IdealHWP",
    # madam.py
    "save_simulation_for_madam",
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
    "TodDescription",
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
    # pointings.py
    "apply_hwp_to_obs",
    "get_pointing_buffer_shape",
    "get_pointings",
    # mapping.py
    "make_bin_map",
    # destripe.py
    "DestriperParameters",
    "DestriperResult",
    "destripe",
    # simulations.py
    "Simulation",
    "MpiObservationDescr",
    "MpiProcessDescr",
    "MpiDistributionDescr",
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
    "ECL_TO_GAL_ROT_MATRIX",
    "CoordinateSystem",
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
    # gaindrifts.py
    "GainDriftType",
    "SamplingDist",
    "GainDriftParams",
    "apply_gaindrift_for_one_detector",
    "apply_gaindrift_to_tod",
    "apply_gaindrift_to_observations",
]
