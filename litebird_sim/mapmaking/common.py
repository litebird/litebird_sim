from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.observations import Observation
from litebird_sim.utilities import cos_sin

# The threshold on the conditioning number used to determine if a pixel
# was really “seen” or not
COND_THRESHOLD = 1e10


@njit
def pointing_row(psi: float, pol_eff: float) -> tuple[float, float, float]:
    """The row of the pointing matrix of one sample: (1, γ cos 2ψ, γ sin 2ψ)

    A sample of a detector with polarization angle ψ and polarization
    efficiency γ measures I + γ Q cos 2ψ + γ U sin 2ψ. All the map-makers
    take the model of a sample from here."""
    cos_2psi, sin_2psi = cos_sin(2 * psi)
    return 1.0, pol_eff * cos_2psi, pol_eff * sin_2psi


@dataclass
class ExternalDestriperParameters:
    """Parameters used by the Madam mapmakers to produce a map.

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

    The weight is the variance of one sample, ``f_samp · NET²``: the map-makers
    divide by it. It is 1 if `observations` has no ``net_ukrts`` attribute.
    `observations` is not modified.
    """

    try:
        net_ukrts = np.broadcast_to(
            np.asarray(observations.net_ukrts, dtype=np.float64),
            (observations.n_detectors,),
        )
        weights = observations.sampling_rate_hz * net_ukrts**2
    except AttributeError:
        weights = np.ones(observations.n_detectors)

    if check:
        # Check that there are no weird weights
        assert np.all(np.isfinite(weights)), (
            f"Not all the detectors' weights are finite numbers: {weights}"
        )
        assert np.all(weights > 0.0), (
            f"Not all the detectors' weights are positive: {weights}"
        )

    return weights


def get_pol_efficiency(observations: Observation) -> npt.NDArray:
    """Return a NumPy array with the polarization efficiency γ of each detector

    The number of elements in the result is equal to `observations.n_detectors`.
    If `observations` has no ``pol_efficiency`` attribute, γ = 1 is assumed.
    """

    try:
        return np.broadcast_to(
            np.asarray(observations.pol_efficiency, dtype=np.float64),
            (observations.n_detectors,),
        )
    except AttributeError:
        return np.ones(observations.n_detectors)


def _cholesky_plain(A: npt.NDArray, dest_L: npt.NDArray) -> None:
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
    dest_L: npt.NDArray,
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
    L: npt.NDArray, v0: float, v1: float, v2: float
) -> tuple[float, float, float]:
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
def _is_well_conditioned(matrix: npt.NDArray) -> bool:
    """Tell if a symmetric positive semi-definite matrix can be inverted

    The condition number is the ratio between the largest and the smallest
    eigenvalue, which are computed with ``np.linalg.eigvalsh``. Closed
    formulas lose too much precision on the smallest eigenvalue when the
    matrix is singular (e.g., 1–2 hits per pixel), and the test does not
    depend on the scale of the matrix (i.e., on the units of the detector
    weights).
    """
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(eigvals[-1] < COND_THRESHOLD * eigvals[0])


def _apply_flags_to_time_masks(
    time_mask_list: list[npt.NDArray],
    obs_list: list[Observation],
    flags: str | list[str] | None,
    flag_mask: int | None,
) -> list[npt.NDArray]:
    """Combine the time masks with the flags of the observations

    Return one Boolean matrix ``(n_detectors, n_samples)`` per observation,
    True for the samples to use: those in the time mask without flags.
    `flags` are the names of the attributes holding the flags, either per
    detector (e.g. ``"local_flags"``) or common to all the detectors (e.g.
    ``"global_flags"``); attributes set to None mean that all the samples are
    good. A sample is flagged if it has any of the bits in `flag_mask`, or any
    nonzero value if `flag_mask` is None. Without flags the matrices are
    read-only views of the time masks, which need no memory.
    """
    # ponytail: with flags this allocates one byte per sample for all the
    # observations at once; test the bits inside the kernels if it matters
    names = [flags] if isinstance(flags, str) else list(flags or [])

    result = []
    for cur_obs, cur_time_mask in zip(obs_list, time_mask_list):
        mask = np.broadcast_to(
            cur_time_mask, (cur_obs.n_detectors, cur_obs.n_samples_process)
        )
        for name in names:
            cur_flags = getattr(cur_obs, name)
            if cur_flags is None:
                continue
            if flag_mask is not None:
                cur_flags = cur_flags & flag_mask
            # Global flags (n_samples,) are broadcast over the detectors
            mask = mask & (cur_flags == 0)
        result.append(mask)

    return result
