# -*- encoding: utf-8 -*-
import importlib
from pathlib import Path

import numba

from litebird_sim.mapmaking import (
    make_binned_map,
    check_valid_splits,
    BinnerResult,
    make_destriped_map,
    DestriperParameters,
    DestriperResult,
    ExternalDestriperParameters,
)
from .bandpasses import BandPassInfo
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
    Imo,
    FormatSpecification,
    Entity,
    Quantity,
    Release,
)
from .io import (
    write_list_of_observations,
    write_observations,
    read_list_of_observations,
)
from .madam import save_simulation_for_madam
from .mbs.mbs import Mbs, MbsParameters, MbsSavedMapInfo
from .mpi import MPI_COMM_WORLD, MPI_ENABLED, MPI_CONFIGURATION
from .noise import (
    add_white_noise,
    add_one_over_f_noise,
    add_noise,
    add_noise_to_observations,
)
from .observations import Observation, TodDescription
from .pointings import (
    apply_hwp_to_obs,
    PointingProvider,
)
from .pointings_in_obs import (
    prepare_pointings,
    precompute_pointings,
)
from .pointing_sys.pointing_sys import (
    get_detector_orientation,
    left_multiply_offset2det,
    left_multiply_disturb2det,
    left_multiply_offset2quat,
    left_multiply_disturb2quat,
    FocalplaneCoord,
    SpacecraftCoord,
    PointingSys,
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
from .version import __author__, __version__

# Check if the TOAST2 mapmaker is available
TOAST_ENABLED = importlib.util.find_spec("OpMapMaker", "toast.todmap") is not None
if not TOAST_ENABLED:

    def destripe_with_toast2(*args, **kwargs):
        raise ImportError(
            "Install the toast package using `pip` to use destripe_with_toast2"
        )

    TOAST_ENABLED = False


# Privilege TBB over OpenPM and the internal Numba implementation of a
# work queue
numba.config.THREADING_LAYER_CONFIG = ["tbb", "omp", "workqueue"]

PTEP_IMO_LOCATION = Path(__file__).parent.parent / "default_imo"

__all__ = [
    "__author__",
    "__version__",
    "PTEP_IMO_LOCATION",
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
    # pointings.py
    "apply_hwp_to_obs",
    "PointingProvider",
    # pointings_in_obs.py
    "prepare_pointings",
    "precompute_pointings",
    # mapmaking
    "make_binned_map",
    "check_valid_splits",
    "BinnerResult",
    "make_destriped_map",
    "DestriperParameters",
    "DestriperResult",
    "ExternalDestriperParameters",
    # toast_destriper.py
    "TOAST_ENABLED",
    "destripe_with_toast2",
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
    # pointing_sys.py
    "get_detector_orientation",
    "left_multiply_offset2det",
    "left_multiply_disturb2det",
    "left_multiply_offset2quat",
    "left_multiply_disturb2quat",
    "FocalplaneCoord",
    "SpacecraftCoord",
    "PointingSys",
]
