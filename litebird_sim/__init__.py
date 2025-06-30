# -*- encoding: utf-8 -*-

import numba

from litebird_sim.mapmaking import (
    make_binned_map,
    make_brahmap_gls_map,
    check_valid_splits,
    BinnerResult,
    make_destriped_map,
    DestriperParameters,
    DestriperResult,
    ExternalDestriperParameters,
)
from .bandpasses import BandPassInfo
from .beam_convolution import (
    add_convolved_sky_to_one_detector,
    add_convolved_sky,
    add_convolved_sky_to_observations,
    BeamConvolutionParameters,
)
from .beam_synthesis import (
    alm_index,
    allocate_alm,
    gauss_beam_to_alm,
    generate_gauss_beam_alms,
)
from .compress import (
    rle_compress,
    rle_decompress,
)
from .coordinates import (
    DEFAULT_COORDINATE_SYSTEM,
    DEFAULT_TIME_SCALE,
    ECL_TO_GAL_ROT_MATRIX,
    CoordinateSystem,
    coord_sys_to_healpix_string,
    rotate_coordinates_e2g,
)
from .detectors import (
    DetectorInfo,
    FreqChannelInfo,
    InstrumentInfo,
    detector_list_from_parameters,
)
from .dipole import add_dipole, add_dipole_to_observations, DipoleType
from .distribute import distribute_evenly, distribute_optimally
from .gaindrifts import (
    GainDriftType,
    SamplingDist,
    GainDriftParams,
    apply_gaindrift_for_one_detector,
    apply_gaindrift_to_tod,
    apply_gaindrift_to_observations,
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
from .hwp import (
    HWP,
    IdealHWP,
    read_hwp_from_hdf5,
)
from .hwp_sys.hwp_sys import (
    HwpSys,
)
from .imo import (
    PTEP_IMO_LOCATION,
    Imo,
    FormatSpecification,
    Entity,
    Quantity,
    Release,
)
from .io import (
    write_list_of_observations,
    read_list_of_observations,
)
from .madam import save_simulation_for_madam
from .mbs.mbs import FG_MODELS, Mbs, MbsParameters, MbsSavedMapInfo
from .mpi import MPI_COMM_WORLD, MPI_ENABLED, MPI_CONFIGURATION, MPI_COMM_GRID
from .mueller_convolver import MuellerConvolver
from .noise import (
    add_white_noise,
    add_one_over_f_noise,
    add_noise,
    add_noise_to_observations,
)
from .non_linearity import (
    NonLinParams,
    apply_quadratic_nonlin_for_one_detector,
    apply_quadratic_nonlin_to_observations,
)
from .observations import Observation, TodDescription
from .pointing_sys import (
    get_detector_orientation,
    FocalplaneCoord,
    SpacecraftCoord,
    PointingSys,
)
from .pointings import (
    PointingProvider,
)
from .pointings_in_obs import (
    prepare_pointings,
    precompute_pointings,
    apply_hwp_to_obs,
)
from .profiler import TimeProfiler, profile_list_to_speedscope
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
    multiply_quaternions_list_x_list,
    multiply_quaternions_list_x_one,
    multiply_quaternions_one_x_list,
    normalize_quaternions,
    quat_rotation,
    quat_rotation_brdcast,
    quat_rotation_x_brdcast,
    quat_rotation_y_brdcast,
    quat_rotation_z_brdcast,
)
from .scan_map import scan_map, scan_map_in_observations
from .scanning import (
    compute_pointing_and_orientation,
    all_compute_pointing_and_orientation,
    spin_to_ecliptic,
    all_spin_to_ecliptic,
    calculate_sun_earth_angles_rad,
    RotQuaternion,
    ScanningStrategy,
    SpinningScanningStrategy,
    get_det2ecl_quaternions,
    get_ecl2det_quaternions,
)
from .seeding import (
    get_derived_random_generators,
    get_detector_level_generators_from_hierarchy,
    get_generator_from_hierarchy,
    regenerate_or_check_detector_generators,
    RNGHierarchy,
)
from .simulations import (
    NUMBA_NUM_THREADS_ENVVAR,
    Simulation,
    MpiObservationDescr,
    MpiProcessDescr,
    MpiDistributionDescr,
)
from .spacecraft import (
    compute_l2_pos_and_vel,
    compute_lissajous_pos_and_vel,
    spacecraft_pos_and_vel,
    SpacecraftOrbit,
    SpacecraftPositionAndVelocity,
)
from .spherical_harmonics import (
    SphericalHarmonics,
)
from .version import __author__, __version__

# Privilege TBB over OpenPM and the internal Numba implementation of a
# work queue
numba.config.THREADING_LAYER_CONFIG = ["tbb", "omp", "workqueue"]


__all__ = [
    "__author__",
    "__version__",
    "PTEP_IMO_LOCATION",
    # beam_convolution.py
    "add_convolved_sky_to_one_detector",
    "add_convolved_sky",
    "add_convolved_sky_to_observations",
    "BeamConvolutionParameters",
    # beam_synthesis.py
    "alm_index",
    "allocate_alm",
    "gauss_beam_to_alm",
    "generate_gauss_beam_alms",
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
    "read_hwp_from_hdf5",
    # hwp_sys/hwp_sys.py
    "HwpSys",
    # madam.py
    "save_simulation_for_madam",
    # mbs.py
    "FG_MODELS",
    "Mbs",
    "MbsParameters",
    "MbsSavedMapInfo",
    # mpi.py
    "MPI_COMM_WORLD",
    "MPI_ENABLED",
    "MPI_CONFIGURATION",
    "MPI_COMM_GRID",
    # mueller_convolver.py
    "MuellerConvolver",
    # observations.py
    "Observation",
    "TodDescription",
    # profiler.py
    "TimeProfiler",
    "profile_list_to_speedscope",
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
    "multiply_quaternions_list_x_list",
    "multiply_quaternions_list_x_one",
    "multiply_quaternions_one_x_list",
    "normalize_quaternions",
    "quat_rotation",
    "quat_rotation_brdcast",
    "quat_rotation_x_brdcast",
    "quat_rotation_y_brdcast",
    "quat_rotation_z_brdcast",
    # scanning.py
    "compute_pointing_and_orientation",
    "all_compute_pointing_and_orientation",
    "spin_to_ecliptic",
    "all_spin_to_ecliptic",
    "calculate_sun_earth_angles_rad",
    "RotQuaternion",
    "ScanningStrategy",
    "SpinningScanningStrategy",
    "get_det2ecl_quaternions",
    "get_ecl2det_quaternions",
    # seeding.py
    "get_derived_random_generators",
    "get_detector_level_generators_from_hierarchy",
    "get_generator_from_hierarchy",
    "regenerate_or_check_detector_generators",
    "RNGHierarchy",
    # pointings.py
    "PointingProvider",
    # pointings_in_obs.py
    "prepare_pointings",
    "precompute_pointings",
    "apply_hwp_to_obs",
    # mapmaking
    "make_binned_map",
    "make_brahmap_gls_map",
    "check_valid_splits",
    "BinnerResult",
    "make_destriped_map",
    "DestriperParameters",
    "DestriperResult",
    "ExternalDestriperParameters",
    # simulations.py
    "NUMBA_NUM_THREADS_ENVVAR",
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
    "coord_sys_to_healpix_string",
    "rotate_coordinates_e2g",
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
    # gaindrifts.py
    "GainDriftType",
    "SamplingDist",
    "GainDriftParams",
    "apply_gaindrift_for_one_detector",
    "apply_gaindrift_to_tod",
    "apply_gaindrift_to_observations",
    # non_linearity.py
    "NonLinParams",
    "apply_quadratic_nonlin_for_one_detector",
    "apply_quadratic_nonlin_to_observations",
    # pointing_sys.py
    "get_detector_orientation",
    "left_multiply_offset2det",
    "left_multiply_disturb2det",
    "left_multiply_offset2quat",
    "left_multiply_disturb2quat",
    "FocalplaneCoord",
    "SpacecraftCoord",
    "PointingSys",
    # spherical_harmonics.py
    "SphericalHarmonics",
]
