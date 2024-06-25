from dataclasses import dataclass
from typing import Union, List, Tuple, Callable
import numpy as np
import numpy.typing as npt
from numba import njit
import astropy.time

from ducc0.healpix import Healpix_Base

from litebird_sim.coordinates import CoordinateSystem, rotate_coordinates_e2g
from litebird_sim.observations import Observation


# The threshold on the conditioning number used to determine if a pixel
# was really “seen” or not
COND_THRESHOLD = 1e10

# Definition of time splits
t_year_sec = 365 * 24 * 3600
t_survey_sec = 365 * 24 * 3600 / 2

# Definition of detector splits
lft_wafers = ["L00", "L01", "L02", "L03", "L04", "L05", "L06", "L07"]
mft_wafers = ["M00", "M01", "M02", "M03", "M04", "M05", "M06"]
hft_wafers = ["H00", "H01", "H02"]


@dataclass
class ExternalDestriperParameters:
    """Parameters used by the TOAST/Madam mapmakers to produce a map.

    The list of fields in this dataclass is the following:

    - ``nside``: the NSIDE parameter used to create the maps

    - ``coordinate_system``: an instance of the :class:`.CoordinateSystem` enum.
      It specifies if the map must be created in ecliptic (default) or
      galactic coordinates.

    - ``nnz``: number of components per pixel. The default is 3 (I/Q/U).

    - ``baseline_length_s``: length of the baseline for 1/f noise in seconds.
      The default is 60.0 s.

    - ``iter_max``: maximum number of iterations. The default is 100

    - ``output_file_prefix``: prefix to be used for the filenames of the
      Healpix FITS maps saved in the output directory. The default
      is ``lbs_``.

    The following Boolean flags specify which maps should be returned
    by the function :func:`.destripe`:

    - ``return_hit_map``: return the hit map (number of hits per
      pixel)

    - ``return_binned_map``: return the binned map (i.e., the map with
      no baselines removed).

    - ``return_destriped_map``: return the destriped map. If pure
      white noise is present in the timelines, this should be the same
      as the binned map.

    - ``return_npp``: return the map of the white noise covariance per
      pixel. It contains the following fields: ``II``, ``IQ``, ``IU``,
      ``QQ``, ``QU``, and ``UU`` (in this order).

    - ``return_invnpp``: return the map of the inverse covariance per
      pixel. It contains the following fields: ``II``, ``IQ``, ``IU``,
      ``QQ``, ``QU``, and ``UU`` (in this order).

    - ``return_rcond``: return the map of condition numbers.

    The default is to only return the destriped map.

    """

    nside: int = 512
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic
    nnz: int = 3
    baseline_length_s: float = 60.0
    iter_max: int = 100
    output_file_prefix: str = "lbs_"
    return_hit_map: bool = False
    return_binned_map: bool = False
    return_destriped_map: bool = True
    return_npp: bool = False
    return_invnpp: bool = False
    return_rcond: bool = False


def get_map_making_weights(
    observations: Observation, check: bool = True
) -> npt.NDArray:
    """Return a NumPy array containing the weights of each detector in `observations`

    The number of elements in the result is equal to `observations.n_detectors`. If
    `check` is true, verify that the weights are ok for the map-maker to
    proceed; if not, an `assert` is raised.
    """

    try:
        if isinstance(observations.net_ukrts, (float, int)):
            observations.net_ukrts = observations.net_ukrts * np.ones(
                observations.n_detectors
            )
        weights = observations.sampling_rate_hz * observations.net_ukrts**2
    except AttributeError:
        weights = np.ones(observations.n_detectors)

    if check:
        # Check that there are no weird weights
        assert np.all(
            np.isfinite(weights)
        ), f"Not all the detectors' weights are finite numbers: {weights}"
        assert np.all(
            weights > 0.0
        ), f"Not all the detectors' weights are positive: {weights}"

    return weights


def _normalize_observations_and_pointings(
    observations: Union[Observation, List[Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None],
) -> Tuple[List[Observation], List[npt.NDArray], List[npt.NDArray]]:
    # In map-making routines, we always rely on two local variables:
    #
    # - obs_list contains a list of the observations to be used in the
    #   map-making process by the current MPI process. Unlike the `observations`
    #   parameters used in functions like `make_binned_map`, this is
    #   *always* a list, i.e., even if there is just one observation
    #
    # - ptg_list: a list of pointing matrices, one per each observation,
    #   each belonging to the current MPI process
    #
    # This function builds the tuple (obs_list, ptg_list, psi_list) and
    # returns it.

    if pointings is None:
        if isinstance(observations, Observation):
            obs_list = [observations]
            if hasattr(observations, "pointing_matrix"):
                ptg_list = [observations.pointing_matrix]
            else:
                ptg_list = [observations.get_pointings]
        else:
            obs_list = observations
            ptg_list = []
            for ob in observations:
                if hasattr(ob, "pointing_matrix"):
                    ptg_list.append(ob.pointing_matrix)
                else:
                    ptg_list.append(ob.get_pointings)
    else:
        if isinstance(observations, Observation):
            assert isinstance(pointings, np.ndarray), (
                "You must pass a list of observations *and* a list "
                + "of pointing matrices to scan_map_in_observations"
            )
            obs_list = [observations]
            ptg_list = [pointings]
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to scan_map_in_observations, "
                + "you must do the same for `pointings`"
            )
            assert len(observations) == len(pointings), (
                f"The list of observations has {len(observations)} elements, but "
                + f"the list of pointings has {len(pointings)} elements"
            )
            obs_list = observations
            ptg_list = pointings

    return obs_list, ptg_list


def _compute_pixel_indices(
    hpx: Healpix_Base,
    pointings: Union[npt.ArrayLike, Callable],
    num_of_detectors: int,
    num_of_samples: int,
    hwp_angle: npt.ArrayLike,
    output_coordinate_system: CoordinateSystem,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Compute the index of each pixel and its attack angle

    The routine returns a pair of arrays whose size is ``(N_d, N_t)`` each: rows in
    the first array contain the index of the pixels in the Healpix map represented by `hpx`
    for a given detector, while rows in the second array the value of the ψ angle (in
    radians) for each detector. The coordinates used are the ones specified by
    `output_coordinate_system`.

    The code assumes that `pointings` is a tensor of rank ``(N_d, N_t, 2)``, with
    ``N_d`` the number of detectors and ``N_t`` the number of samples in the TOD,
    and the last rank represents the θ and φ angles (in radians) expressed in the
    Ecliptic reference frame.
    """

    pixidx_all = np.empty((num_of_detectors, num_of_samples), dtype=int)
    polang_all = np.empty((num_of_detectors, num_of_samples), dtype=np.float64)

    for idet in range(num_of_detectors):
        if type(pointings) is np.ndarray:
            curr_pointings_det = pointings[idet, :, :]
        else:
            curr_pointings_det, hwp_angle = pointings(idet)
            curr_pointings_det = curr_pointings_det.reshape(-1, 3)

        if hwp_angle is None:
            hwp_angle = 0

        if output_coordinate_system == CoordinateSystem.Galactic:
            curr_pointings_det = rotate_coordinates_e2g(curr_pointings_det)

        polang_all[idet] = curr_pointings_det[:, 2] + hwp_angle

        pixidx_all[idet] = hpx.ang2pix(curr_pointings_det[:, :2])

    if output_coordinate_system == CoordinateSystem.Galactic:
        # Free curr_pointings_det if the output map is already in Galactic coordinates
        del curr_pointings_det

    return pixidx_all, polang_all


def _cholesky_plain(A: npt.ArrayLike, dest_L: npt.ArrayLike) -> None:
    "Store a lower-triangular matrix in L such that A = L·L†"

    # The following function is a standard textbook implementation of
    # the Cholesky algorithm. It works for an arbitrary matrix N×N.
    # "print" statements have been inserted, so that when this is run, it
    # produces the plain list of statements used to compute the matrix.
    # This is used to implement the function _cholesky_explicit that
    # is provided below, which only works on 3×3 matrices.

    N = 3
    for i in range(N):
        for j in range(N):
            print(f"L[{i}][{j}] = 0.0")
            dest_L[i][j] = 0.0

    for i in range(N):
        for j in range(i + 1):
            accum = 0.0
            print("accum = 0.0")
            for k in range(j + 1):
                accum += dest_L[i][k] * dest_L[j][k]
                print(f"accum += L[{i}][{k}] * L[{j}][{k}]")

            if i == j:
                dest_L[i][i] = np.sqrt(A[i][i] - accum)
                print(f"L[{i}][{i}] = np.sqrt(A[{i}][{i}] - accum)")
            else:
                dest_L[i][j] = (1.0 / dest_L[j][j]) * (A[i][j] - accum)
                print(f"L[{i}][{j}] = (1.0 / L[{j}][{j}]) * (A[{i}][{j}] - accum)")


@njit
def _cholesky_explicit(A, dest_L):
    "Store a lower-triangular matrix in L such that A = L·L†"

    # The code below is the result of a manual optimization of the output
    # of the `print` statements in the function `_cholesky_plain` above.
    # If you pass a 3×3 matrix to `_cholesky_plain`, the list of
    # statements produced in the output involves many useless operations,
    # like adding terms that are always equal to zero. By manually
    # removing them, one gets the current implementation of
    # `_cholesky_explicit`, which was in turn used to implement the
    # `cholesky` function provided below.

    dest_L[0][1] = 0.0
    dest_L[0][2] = 0.0
    dest_L[1][2] = 0.0

    dest_L[0][0] = np.sqrt(A[0][0])
    dest_L[1][0] = A[1][0] / dest_L[0][0]
    dest_L[1][1] = np.sqrt(A[1][1] - dest_L[1][0] * dest_L[1][0])
    dest_L[2][0] = A[2][0] / dest_L[0][0]
    dest_L[2][1] = (A[2][1] - dest_L[2][0] * dest_L[1][0]) / dest_L[1][1]
    dest_L[2][2] = np.sqrt(
        A[2][2] - dest_L[2][0] * dest_L[2][0] - dest_L[2][1] * dest_L[2][1]
    )


@njit
def cholesky(
    a00: float,
    a10: float,
    a11: float,
    a20: float,
    a21: float,
    a22: float,
    dest_L: npt.ArrayLike,
) -> None:
    """Store a 3×3 lower-triangular matrix in L such that A = L·L†

    Matrix A must be a 3×3 symmetric and positive definite matrix. Only the
    elements a₀₀, a₁₀, a₁₁, a₂₀, a₂₁, a₂₂ are needed,
    i.e., the ones on the lower-triangular part of the matrix.

    We ask the caller to pass the coefficients explicitly so that there
    is no need to allocate a real 3×3 matrix.

    The vector L must have room for 6 elements:
    L[0] = l₀₀
    L[1] = l₁₀
    L[2] = l₁₁
    L[3] = l₂₀
    L[4] = l₂₁
    L[5] = l₂₂

    To understand how this function was implemented, have a look at the comments
    in the bodies of `_cholesky_plain` and `_cholesky_explicit` above.
    """

    dest_L[0] = np.sqrt(a00)
    dest_L[1] = a10 / dest_L[0]
    dest_L[2] = np.sqrt(a11 - dest_L[1] * dest_L[1])
    dest_L[3] = a20 / dest_L[0]
    dest_L[4] = (a21 - dest_L[3] * dest_L[1]) / dest_L[2]
    dest_L[5] = np.sqrt(a22 - dest_L[3] * dest_L[3] - dest_L[4] * dest_L[4])


@njit
def solve_cholesky(
    L: npt.ArrayLike, v0: float, v1: float, v2: float
) -> Tuple[float, float, float]:
    """Solve Ax = b if A is a 3×3 symmetric positive definite matrix.

    Instead of providing the matrix A, the caller is expected to provide its
    Cholesky decomposition: the parameter `L` is the lower-triangular matrix
    such that A = L·L†
    """

    # We solve L·L† = v by rewriting it as the system
    #
    #    L† x = y     (upper-triangular system)
    #    L y = v      (lower-triangular system)
    #
    # and solving the two systems in the order shown above. This is just
    # plain algebra!

    # First get y…
    y0 = v0 / L[0]
    y1 = (v1 - L[1] * y0) / L[2]
    y2 = (v2 - L[3] * y0 - L[4] * y1) / L[5]

    # …then get x
    dest_u = y2 / L[5]
    dest_q = (y1 - L[4] * dest_u) / L[2]
    dest_i = (y0 - L[1] * dest_q - L[3] * dest_u) / L[0]

    return (dest_i, dest_q, dest_u)


@njit
def estimate_cond_number(
    a00: float,
    a10: float,
    a11: float,
    a20: float,
    a21: float,
    a22: float,
) -> Tuple[float, bool]:
    """Estimate the condition number for a symmetric 3×3 matrix A

    The result is a tuple containing the condition number and a Boolean flag
    telling if the matrix is non-singular (``True``) or singular (``False``).

    The code is a conversion of a C++ template class by David Eberly, see
    <https://www.geometrictools.com/Documentation/RobustEigenSymmetric3×3.pdf>
    """

    # Precondition the matrix by dividing each member by the largest
    max_abs_element = max(
        np.abs(a00),
        np.abs(a10),
        np.abs(a20),
        np.abs(a11),
        np.abs(a21),
        np.abs(a22),
    )

    if max_abs_element == 0.0:
        # All the elements are 0.0, quit immediately
        return (0.0, False)

    inv_max_abs_element = 1.0 / max_abs_element
    a00 *= inv_max_abs_element
    a10 *= inv_max_abs_element
    a20 *= inv_max_abs_element
    a11 *= inv_max_abs_element
    a21 *= inv_max_abs_element
    a22 *= inv_max_abs_element

    norm = a10 * a10 + a20 * a20 + a21 * a21

    if norm > 0:
        q = (a00 + a11 + a22) / 3
        b00 = a00 - q
        b11 = a11 - q
        b22 = a22 - q

        p = np.sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6)
        c00 = b11 * b22 - a21 * a21
        c01 = a10 * b22 - a21 * a20
        c02 = a10 * a21 - b11 * a20
        det = (b00 * c00 - a10 * c01 + a20 * c02) / (p * p * p)

        halfDet = det * 0.5
        halfDet = min(max(halfDet, -1), 1)

        angle = np.arccos(halfDet) / 3
        twoThirdsPi = 2 * np.pi / 3
        beta2 = np.cos(angle) * 2
        beta0 = np.cos(angle + twoThirdsPi) * 2
        beta1 = -(beta0 + beta2)

        eval0 = q + p * beta0
        eval1 = q + p * beta1
        eval2 = q + p * beta2

    else:
        eval0 = a00
        eval1 = a11
        eval2 = a22

    # Rescale the eigenvalues, as we preconditioned the matrix,
    # and throw away the signs, as we are not interested in them
    eval0 = np.abs(eval0 * max_abs_element)
    eval1 = np.abs(eval1 * max_abs_element)
    eval2 = np.abs(eval2 * max_abs_element)

    min_abs_eval = min(eval0, eval1, eval2)
    if min_abs_eval < 1e-10:
        # The matrix is singular (detA = 0)
        return (0.0, False)

    max_abs_eval = max(max(eval0, eval1), eval2)

    return (max_abs_eval / min_abs_eval, True)


def _build_mask_time_split(
    time_split: str,
    obs_list: List[Observation],
):
    time_mask = []

    for cur_obs in obs_list:
        mask = np.zeros(cur_obs.n_samples, dtype=bool)

        if time_split == "full":
            time_mask.append(np.ones(cur_obs.n_samples, dtype=bool))
        elif time_split == "odd":
            mask[0::2] = True
            time_mask.append(mask)
        elif time_split == "even":
            mask[1::2] = True
            time_mask.append(mask)
        elif time_split == "first_half":
            mask[0 : cur_obs.n_samples // 2] = True
            time_mask.append(mask)
        elif time_split == "second_half":
            mask[cur_obs.n_samples // 2 :] = True
            time_mask.append(mask)
        max_years = 3
        for i in range(1, max_years + 1):
            if time_split == f"year{i}":
                t_i = _get_initial_time(cur_obs)
                time_mask.append(
                    ((cur_obs.get_times() - t_i) >= (i - 1) * t_year_sec)
                    * ((cur_obs.get_times() - t_i) < i * t_year_sec)
                )
        max_surveys = 6
        for i in range(1, max_surveys + 1):
            if time_split == f"survey{i}":
                t_i = _get_initial_time(cur_obs)
                time_mask.append(
                    ((cur_obs.get_times() - t_i) >= (i - 1) * t_survey_sec)
                    * ((cur_obs.get_times() - t_i) < i * t_survey_sec)
                )
    return time_mask


def _get_initial_time(
    observations: Observation,
):
    if isinstance(observations.start_time_global, astropy.time.Time):
        t_i = observations.start_time_global.cxcsec
    else:
        t_i = observations.start_time_global
    return t_i


def _get_end_time(
    observations: Observation,
):
    if isinstance(observations.end_time_global, astropy.time.Time):
        t_f = observations.end_time_global.cxcsec
    else:
        t_f = observations.end_time_global
    return t_f


def _build_mask_detector_split(
    detector_split: str,
    obs_list: List[Observation],
):
    detector_mask = []

    if detector_split == "full":
        for cur_obs in obs_list:
            detector_mask.append(np.ones(cur_obs.n_detectors, dtype=bool))
    elif "wafer" in detector_split:
        for cur_obs in obs_list:
            detector_mask.append(cur_obs.wafer == detector_split.replace("wafer", ""))

    return detector_mask


def _check_valid_splits(
    observations: Union[Observation, List[Observation]],
    detector_splits: Union[str, List[str]] = "full",
    time_splits: Union[str, List[str]] = "full",
):
    valid_detector_splits = ["full"]
    valid_detector_splits.extend(
        [f"wafer{wafer}" for wafer in lft_wafers + mft_wafers + hft_wafers]
    )
    valid_time_splits = [
        "full",
        "first_half",
        "second_half",
        "odd",
        "even",
    ]
    max_years = 3
    max_surveys = 6
    valid_time_splits.extend([f"year{i}" for i in range(1, max_years + 1)])
    valid_time_splits.extend([f"survey{i}" for i in range(1, max_surveys + 1)])

    if isinstance(observations, Observation):
        observations = [observations]
    if isinstance(detector_splits, str):
        detector_splits = [detector_splits]
    if isinstance(time_splits, str):
        time_splits = [time_splits]

    _validate_detector_splits(observations, detector_splits, valid_detector_splits)
    _validate_time_splits(observations, time_splits, valid_time_splits)
    print("Splits are valid!")


def _validate_detector_splits(observations, detector_splits, valid_detector_splits):
    for ds in detector_splits:
        if ds not in valid_detector_splits:
            msg = f"Detector split '{ds}' not recognized!\nValid detector splits are {valid_detector_splits}"
            raise ValueError(msg)
        for cur_obs in observations:
            if "wafer" in ds:
                requested_wafer = ds.replace("wafer", "")
                if requested_wafer not in cur_obs.wafer:
                    msg = f"The requested wafer '{ds}' is not part of the requested observation with wafers {cur_obs.wafer}!"
                    raise AssertionError(msg)


def _validate_time_splits(observations, time_splits, valid_time_splits):
    for ts in time_splits:
        if ts not in valid_time_splits:
            msg = f"Time split '{ts}' not recognized!\nValid time splits are {valid_time_splits}"
            raise ValueError(msg)
        if "year" in ts:
            for cur_obs in observations:
                duration = round(_get_end_time(cur_obs) - _get_initial_time(cur_obs), 0)
                max_years = duration // t_year_sec
                requested_years = int(ts.replace("year", ""))
                if requested_years > max_years:
                    msg = f"Time split '{ts}' not possible for observation with a duration of {round(duration / t_year_sec, 1)} years!"
                    raise AssertionError(msg)
