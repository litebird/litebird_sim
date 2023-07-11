# -*- encoding: utf-8 -*-

from typing import Union, List, Tuple
import numpy as np
import numpy.typing as npt
from numba import njit

from ducc0.healpix import Healpix_Base

from litebird_sim.coordinates import CoordinateSystem, rotate_coordinates_e2g
from litebird_sim.observations import Observation


# The threshold on the conditioning number used to determine if a pixel
# was really “seen” or not
COND_THRESHOLD = 1e10


def get_map_making_weights(obs: Observation, check: bool) -> npt.NDArray:
    """Return a NumPy array containing the weights of each detector in `obs`

    The number of elements in the result is equal to `obs.n_detectors`. If
    `check` is true, verifies that the weights are ok for the map-maker to
    proceed: if not, an `assert` is raised.
    """

    try:
        weights = obs.sampling_rate_hz * obs.net_ukrts**2
    except AttributeError:
        weights = np.ones(obs.n_detectors)

    if check:
        # Check that there are no weird weights
        assert np.alltrue(
            np.isfinite(weights)
        ), f"Not all the detectors' weights are finite numbers: {weights}"
        assert np.alltrue(
            weights > 0.0
        ), f"Not all the detectors' weights are positive: {weights}"

    return weights


def _normalize_observations_and_pointings(
    obs: Union[Observation, List[Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None],
) -> Tuple[List[Observation], List[npt.NDArray], List[npt.NDArray]]:
    # In map-making routines, we always rely on three local variables:
    #
    # - obs_list contains a list of the observations to be used in the
    #   map-making process by the current MPI process. Unlike the `obs`
    #   parameters used in functions like `make_bin_map`, this is
    #   *always* a list, i.e., even if there is just one observation
    #
    # - ptg_list: a list of pointing matrices, one per each observation,
    #   each belonging to the current MPI process
    #
    # - psi_list: a list of pointing angle vectors, one per each observation,
    #   each belonging to the current MPI process
    #
    # This function builds the tuple (obs_list, ptg_list, psi_list) and
    # returns it.

    if pointings is None:
        if isinstance(obs, Observation):
            obs_list = [obs]
            ptg_list = [obs.pointings]
            psi_list = [obs.psi]
        else:
            obs_list = obs
            ptg_list = [ob.pointings for ob in obs]
            psi_list = [ob.psi for ob in obs]
    else:
        if isinstance(obs, Observation):
            assert isinstance(pointings, np.ndarray), (
                "You must pass a list of observations *and* a list "
                + "of pointing matrices to scan_map_in_observations"
            )
            obs_list = [obs]
            ptg_list = [pointings[:, :, 0:2]]
            psi_list = [pointings[:, :, 2]]
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to make_bin_map, "
                + "you must do the same for `pointings`"
            )
            assert len(obs) == len(pointings), (
                f"The list of observations has {len(obs)} elements, but "
                + f"the list of pointings has {len(pointings)} elements"
            )
            obs_list = obs
            ptg_list = [point[:, :, 0:2] for point in pointings]
            psi_list = [point[:, :, 2] for point in pointings]

    return obs_list, ptg_list, psi_list


def _compute_pixel_indices(
    hpx: Healpix_Base,
    pointings: npt.ArrayLike,
    psi: npt.ArrayLike,
    output_coordinate_system: CoordinateSystem,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Compute the index of each pixel and its attack angle

    The routine returns a pair of arrays whose size is `num_of_sample`: the first
    one contains the index of the pixels in the Healpix map represented by `hpx`,
    and the second the value of the ψ angle (in radians). The coordinates used are
    the ones specified by `output_coordinate_system`.

    The code assumes that `pointings` is a tensor of rank ``(N_d, N_t, 2)``, with
    ``N_d`` the number of detectors and ``N_t`` the number of samples in the TOD,
    and the last rank represents the θ and φ angles (in radians) expressed in the
    Ecliptic reference frame.
    """
    num_of_detectors, num_of_samples, _ = pointings.shape
    pixidx_all = np.empty((num_of_detectors, num_of_samples), dtype=int)
    polang_all = np.empty((num_of_detectors, num_of_samples), dtype=np.float64)

    for idet in range(num_of_detectors):
        if output_coordinate_system == CoordinateSystem.Galactic:
            curr_pointings_det, polang_all[idet] = rotate_coordinates_e2g(
                pointings[idet, :, :], psi[idet, :]
            )
        else:
            curr_pointings_det = pointings[idet, :, :]
            polang_all[idet] = psi[idet, :]

        pixidx_all[idet] = hpx.ang2pix(curr_pointings_det)

    if output_coordinate_system == CoordinateSystem.Galactic:
        # free curr_pointings_det if the output map is already in Galactic coordinates
        del curr_pointings_det

    return pixidx_all, polang_all


def _cholesky_plain(A: npt.ArrayLike, dest_L: npt.ArrayLike) -> None:
    "Store a lower-triangular matrix in L such that A = L·L†"

    # The following function is a standard textbook implementation of
    # the Cholesky algorithm. It works for an arbitrary matrix N×N.
    # I have inserted "print" statements so that when this is run, it
    # produces the plain list of statements used to compute the matrix.
    # This is used to implement the function _cholesky_explicit that
    # is provided below

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
    # If you run `_cholesky_plain` passing a 3×3 matrix, the list of
    # statements produced in the output involve useless operations, like
    # adding terms that are always equal to zero. By manually removing
    # them, one gets the current implementation of `_cholesky_explicit`,
    # which was in turn used to implement the `cholesky` function
    # provided below.

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
def solve_cholesky(L: npt.ArrayLike, v: npt.ArrayLike, dest_x: npt.ArrayLike):
    """Solve Ax = b if A is a 3×3 symmetric positive definite matrix.

    Instead of providing the matrix A, the caller is expected to provide its
    Cholesky decomposition:  the parameter `L` is the lower-triangular matrix
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
    y0 = v[0] / L[0]
    y1 = (v[1] - L[1] * y0) / L[2]
    y2 = (v[2] - L[3] * y0 - L[4] * y1) / L[5]

    # …then get x
    dest_x[2] = y2 / L[5]
    dest_x[1] = (y1 - L[4] * dest_x[2]) / L[2]
    dest_x[0] = (y0 - L[1] * dest_x[1] - L[3] * dest_x[2]) / L[0]


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
    <https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf>
    """

    # Precondition the matrix by dividing each member by the largest
    max0 = max(np.abs(a00), np.abs(a10))
    max1 = max(np.abs(a20), np.abs(a11))
    max2 = max(np.abs(a21), np.abs(a22))
    max_abs_element = max(max(max0, max1), max2)

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
        twoThirdsPi = 2.09439510239319549
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

    min_abs_eval = min(min(eval0, eval1), eval2)
    if min_abs_eval < 1e-10:
        # The matrix is singular (detA = 0)
        return (0.0, False)

    max_abs_eval = max(max(eval0, eval1), eval2)

    return (max_abs_eval / min_abs_eval, True)
