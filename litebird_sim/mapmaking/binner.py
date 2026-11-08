# The implementation of the binning algorithm provided here is derived
# from the more general destriping equation presented in the paper
# «Destriping CMB temperature and polarization maps» by Kurki-Suonio et al. 2009,
# A&A 506, 1511–1539 (2009), https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code, as many
# functions and variable defined here use the same letters and symbols of that
# paper. We refer to it in code comments and docstrings as "KurkiSuonio2009".

from dataclasses import dataclass
from typing import Any

import healpy as hp
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.hwp import HWP
from litebird_sim.observations import Observation
from litebird_sim.observation_utilities import (
    ObservationPointings,
    observation_pointings,
)
from litebird_sim.maps_and_harmonics import HealpixMap

from ..utilities import resolve_nthreads

from .reduction import reduction_for_observations
from .splits import _build_mask_detector_split, _build_mask_time_split
from .common import (
    pointing_row,
    _apply_flags_to_time_masks,
    _is_well_conditioned,
    get_map_making_weights,
    get_pol_efficiency,
)


@dataclass
class BinnerResult:
    """Result of a call to the :func:`.make_binned_map` function

    This dataclass has the following fields:

    - ``binned_map``: Healpix map containing the binned value for each pixel

    - ``invnpp``: inverse of the covariance matrix element for each
      pixel in the map. It is an array of shape `(12 * nside * nside, 3, 3)`

    - ``coordinate_system``: the coordinate system of the output maps
      (a :class:`.CoordinateSistem` object)

    - ``components``: list of components included in the map, by default
      only the field ``tod`` is used

    - ``detector_split``: detector split of the binned map

    - ``time_split``: time split of the binned map
    """

    binned_map: Any = None
    invnpp: Any = None
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic
    components: list | None = None
    detector_split: str = "full"
    time_split: str = "full"


@njit(parallel=True)
def _solve_binning(nobs_matrix, atd):
    # Solve the map-making equation
    #
    # This method alters the parameter `nobs_matrix`, so that after its completion
    # each 3×3 matrix in nobs_matrix[idx, :, :] will be the *inverse*.

    # Expected shape:
    # - `nobs_matrix`: (N_p, 3, 3) is an array of N_p symmetric 3×3 matrices, where
    #   N_p is the number of pixels in the map
    # - `atd`: (N_p, 3)
    #
    # The 3×3 matrices are inverted analytically (adjugate over determinant),
    # and `np.linalg.eigvalsh` is only used for the eigenvalues: calling
    # `np.linalg` three times per pixel (cond, solve, inv) was ~5× slower
    npix = atd.shape[0]

    for ipix in prange(npix):  # type: ignore[not-iterable]
        m = nobs_matrix[ipix]
        a00, a10, a11 = m[0, 0], m[1, 0], m[1, 1]
        a20, a21, a22 = m[2, 0], m[2, 1], m[2, 2]

        # Cofactors of the symmetric matrix
        c00 = a11 * a22 - a21 * a21
        c10 = a20 * a21 - a10 * a22
        c20 = a10 * a21 - a20 * a11
        c11 = a00 * a22 - a20 * a20
        c21 = a10 * a20 - a00 * a21
        c22 = a00 * a11 - a10 * a10
        det = a00 * c00 + a10 * c10 + a20 * c20

        if not _is_well_conditioned(m):
            m.fill(hp.UNSEEN)
            atd[ipix].fill(hp.UNSEEN)
            continue

        inv_det = 1.0 / det
        m[0, 0] = c00 * inv_det
        m[1, 1] = c11 * inv_det
        m[2, 2] = c22 * inv_det
        m[0, 1] = m[1, 0] = c10 * inv_det
        m[0, 2] = m[2, 0] = c20 * inv_det
        m[1, 2] = m[2, 1] = c21 * inv_det

        b0, b1, b2 = atd[ipix, 0], atd[ipix, 1], atd[ipix, 2]
        for i in range(3):
            atd[ipix, i] = m[i, 0] * b0 + m[i, 1] * b1 + m[i, 2] * b2


@njit
def _accumulate_samples_and_build_nobs_matrix(
    tod: npt.NDArray,
    pix: npt.NDArray,
    psi: npt.NDArray,
    weights: npt.NDArray,
    pol_eff: npt.NDArray,
    d_mask: npt.NDArray,
    t_mask: npt.NDArray,
    nobs_matrix: npt.NDArray,
    *,
    additional_component: bool,
) -> None:
    # Fill the upper triangle of the N_obs matrix and use the lower
    # triangle for the RHS of the map-making equation:
    #
    # 1. The upper triangle and the diagonal contains the coefficients in
    #    Eq. (10) of KurkiSuonio2009. This must be set just once, as it only
    #    depends on the pointing information. The flag `additional_component`
    #    tells if this part must be calculated (``False``) or not
    # 2. The lower triangle contains the weighted sum of I/Q/U, i.e.,
    #
    #       (I + γ·Q·cos(2ψ) + γ·U·sin(2ψ)) / σ²
    #
    #    where γ is the polarization efficiency of the detector

    assert tod.shape == pix.shape == psi.shape

    assert tod.shape[0] == d_mask.shape[0]

    num_of_detectors = tod.shape[0]

    for idet in range(num_of_detectors):
        if not d_mask[idet]:
            continue

        inv_sigma2 = 1.0 / weights[idet]

        # One pass for both triangles, so that cos/sin are computed once
        for cur_sample, cur_pix_idx, cur_psi, cur_t_mask in zip(
            tod[idet, :], pix[idet, :], psi[idet, :], t_mask[idet, :]
        ):
            if cur_t_mask:
                r_i, r_q, r_u = pointing_row(cur_psi, pol_eff[idet])
                info_pix = nobs_matrix[cur_pix_idx]

                if not additional_component:
                    # Upper triangle
                    info_pix[0, 0] += r_i * r_i * inv_sigma2
                    info_pix[0, 1] += r_i * r_q * inv_sigma2
                    info_pix[0, 2] += r_i * r_u * inv_sigma2
                    info_pix[1, 1] += r_q * r_q * inv_sigma2
                    info_pix[1, 2] += r_q * r_u * inv_sigma2
                    info_pix[2, 2] += r_u * r_u * inv_sigma2

                # Lower triangle
                weighted_sample = cur_sample * inv_sigma2
                info_pix[1, 0] += r_i * weighted_sample
                info_pix[2, 0] += r_q * weighted_sample
                info_pix[2, 1] += r_u * weighted_sample


@njit
def _numba_extract_map_and_fill_nobs_matrix(
    nobs_matrix: npt.NDArray, rhs: npt.NDArray
) -> None:
    # This is used internally by _extract_map_and_fill_info. The function
    # modifies both `info` and `rhs`; the first parameter would be an `inout`
    # parameter in Fortran (it is both used as input and output), while `rhs`
    # is an `out` parameter
    for idx in range(nobs_matrix.shape[0]):
        # Extract the vector from the lower left triangle of the 3×3 matrix
        # nobs_matrix[idx, :, :]
        rhs[idx, 0] = nobs_matrix[idx, 1, 0]
        rhs[idx, 1] = nobs_matrix[idx, 2, 0]
        rhs[idx, 2] = nobs_matrix[idx, 2, 1]

        # Make each 3×3 matrix in nobs_matrix[idx, :, :] symmetric
        nobs_matrix[idx, 1, 0] = nobs_matrix[idx, 0, 1]
        nobs_matrix[idx, 2, 0] = nobs_matrix[idx, 0, 2]
        nobs_matrix[idx, 2, 1] = nobs_matrix[idx, 1, 2]


def _extract_map_and_fill_info(info: npt.NDArray) -> npt.NDArray:
    # Extract the RHS of the mapmaking equation from the lower triangle of info
    # and fill the lower triangle with the upper triangle, thus making each
    # matrix in "info" symmetric
    rhs = np.empty((info.shape[0], 3), dtype=info.dtype)

    # The implementation in Numba of this code is ~5 times faster than the older
    # implementation that used NumPy.
    _numba_extract_map_and_fill_nobs_matrix(info, rhs)

    return rhs


def _build_nobs_matrix(
    nside: int,
    obs_pointings: list[tuple[Observation, ObservationPointings]],
    dm_list: list[npt.NDArray],
    tm_list: list[npt.NDArray],
    components: list[str],
) -> npt.NDArray:
    n_pix = HealpixMap.nside_to_npix(nside)

    nobs_matrix = np.zeros((n_pix, 3, 3))

    for (cur_obs, cur_ptg), cur_d_mask, cur_t_mask in zip(
        obs_pointings, dm_list, tm_list
    ):
        cur_weights = get_map_making_weights(cur_obs, check=True)
        cur_pol_eff = get_pol_efficiency(cur_obs)

        # In the coordinate system of the map
        pixidx_all, polang_all = cur_ptg.pixel_index_and_pol_angle("all", nside)

        first_component = getattr(cur_obs, components[0])
        for idx, cur_component_name in enumerate(components):
            cur_component = getattr(cur_obs, cur_component_name)
            assert cur_component.shape == first_component.shape, (
                'The two TODs "{}" and "{}" do not have a matching shape'.format(
                    components[0], cur_component_name
                )
            )
            _accumulate_samples_and_build_nobs_matrix(
                cur_component,
                pixidx_all,
                polang_all,
                cur_weights,
                cur_pol_eff,
                cur_d_mask,
                cur_t_mask,
                nobs_matrix,
                additional_component=idx > 0,
            )

        del pixidx_all, polang_all

    assert obs_pointings, "No observations provided"
    reduction_for_observations([cur_obs for cur_obs, _ in obs_pointings]).sum_in_place(
        nobs_matrix
    )

    return nobs_matrix


def make_binned_map(
    nside: int,
    observations: Observation | list[Observation],
    pointings: np.ndarray | list[np.ndarray] | None = None,
    hwp: HWP | None = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    components: str | list[str] = "tod",
    detector_split: str = "full",
    time_split: str = "full",
    pointings_dtype=np.float64,
    nthreads: int | None = None,
    flags: str | list[str] | None = None,
    flag_mask: int | None = None,
) -> BinnerResult:
    """Bin Map-maker

    Map a list of observations

    Args:
        observations (list of :class:`Observations`): observations to be mapped.
            They are required to have the following attributes as arrays

            * `pointings`: the pointing information (in radians) for each tod
               sample. It must be a tensor with shape ``(N_d, N_t, 3)``,
               with ``N_d`` number of detectors and ``N_t`` number of
               samples in the TOD.
            * any attribute listed in `components` (by default, `tod`) and
              containing the TOD(s) to be binned together.

            If the observations are distributed over some communicator(s), they
            must share the same group processes.
            If pointings and psi are not included in the observations, they can
            be provided through an array (or a list of arrays) of dimension
            (Ndetectors x Nsamples x 3), containing theta, phi and psi
        nside (int): HEALPix nside of the output map
        pointings (array or list of arrays): optional, external pointing
            information, if not included in the observations
        hwp (HWP, optional): An instance of the :class:`.HWP` class (optional)
        output_coordinate_system (:class:`.CoordinateSystem`): the coordinates
            to use for the output map
        components (list[str] or str): components to include in the map-making.
            The default is just to use the field ``tod`` of each
            :class:`.Observation` object
        detector_split (str): select the detector split to use in the map-making
        time_split (str): select the time split to use in the map-making.
        flags (str or list[str]): names of the :class:`.Observation`
            attributes holding the flags, either per detector (e.g.
            ``"local_flags"``) or common to all the detectors (e.g.
            ``"global_flags"``). Flagged samples are excluded from the map;
            attributes set to None mean that all the samples are good
        flag_mask (int): bits of the flags that exclude a sample (e.g.
            ``FLAG_PLANET``). If None, any nonzero flag excludes it
        pointings_dtype(dtype): data type for pointings generated on the fly. If
            the pointing is passed or already precomputed this parameter is
            ineffective. Default is `np.float64`.
        nthreads : int, default=None
            Number of threads to use in ducc's Healpix methods. If None, the
            function reads from the `OMP_NUM_THREADS` environment variable.

    Returns:
        An instance of the class :class:`.MapMakerResult`. If the observations are
            distributed over MPI Processes, all of them get a copy of the same object.
    """

    if isinstance(components, str):
        components = [components]

    # Set number of threads
    nthreads = resolve_nthreads(nthreads)

    obs_pointings = observation_pointings(
        observations,
        pointings=pointings,
        hwp=hwp,
        output_coordinate_system=output_coordinate_system,
        pointings_dtype=pointings_dtype,
        nthreads=nthreads,
    )
    assert obs_pointings, "No observations provided"
    obs_list = [cur_obs for cur_obs, _ in obs_pointings]

    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)

    time_mask_list = _apply_flags_to_time_masks(
        _build_mask_time_split(time_split, obs_list), obs_list, flags, flag_mask
    )

    nobs_matrix = _build_nobs_matrix(
        nside=nside,
        obs_pointings=obs_pointings,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
        components=components,
    )

    rhs = _extract_map_and_fill_info(nobs_matrix)

    _solve_binning(nobs_matrix, rhs)

    return BinnerResult(
        binned_map=rhs.T,
        invnpp=nobs_matrix,
        coordinate_system=output_coordinate_system,
        components=components,
        detector_split=detector_split,
        time_split=time_split,
    )
