# The implementation of the binning algorithm provided here is derived
# from the more general destriping equation presented in the paper
# «Destriping CMB temperature and polarization maps» by Kurki-Suonio et al. 2009,
# A&A 506, 1511–1539 (2009), https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code, as many
# functions and variable defined here use the same letters and symbols of that
# paper. We refer to it in code comments and docstrings as "KurkiSuonio2009".

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
import healpy as hp

from typing import Union, List, Any, Optional, Callable
from litebird_sim.observations import Observation
from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.pointings import get_hwp_angle
from litebird_sim.hwp import HWP
from litebird_sim import mpi
from ducc0.healpix import Healpix_Base
from litebird_sim.healpix import nside_to_npix

import logging

from .common import (
    _compute_pixel_indices,
    _normalize_observations_and_pointings,
    COND_THRESHOLD,
    get_map_making_weights,
    _build_mask_detector_split,
    _build_mask_time_split,
    _check_valid_splits,
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
    components: List = None
    detector_split: str = "full"
    time_split: str = "full"


@njit
def _solve_binning(nobs_matrix, atd):
    # Sove the map-making equation
    #
    # This method alters the parameter `nobs_matrix`, so that after its completion
    # each 3×3 matrix in nobs_matrix[idx, :, :] will be the *inverse*.

    # Expected shape:
    # - `nobs_matrix`: (N_p, 3, 3) is an array of N_p 3×3 matrices, where
    #   N_p is the number of pixels in the map
    # - `atd`: (N_p, 3)
    npix = atd.shape[0]

    for ipix in range(npix):
        if np.linalg.cond(nobs_matrix[ipix]) < COND_THRESHOLD:
            atd[ipix] = np.linalg.solve(nobs_matrix[ipix], atd[ipix])
            nobs_matrix[ipix] = np.linalg.inv(nobs_matrix[ipix])
        else:
            nobs_matrix[ipix].fill(hp.UNSEEN)
            atd[ipix].fill(hp.UNSEEN)


@njit
def _accumulate_samples_and_build_nobs_matrix(
    tod: npt.ArrayLike,
    pix: npt.ArrayLike,
    psi: npt.ArrayLike,
    weights: npt.ArrayLike,
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    nobs_matrix: npt.ArrayLike,
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
    #       (I + Q·cos(2ψ) + U·sin(2ψ)) / σ²

    assert tod.shape == pix.shape == psi.shape

    assert tod.shape[0] == d_mask.shape[0]

    num_of_detectors = tod.shape[0]

    for idet in range(num_of_detectors):
        if not d_mask[idet]:
            continue

        inv_sigma = 1.0 / np.sqrt(weights[idet])
        inv_sigma2 = inv_sigma * inv_sigma

        if not additional_component:
            # Fill the upper triangle
            for cur_pix_idx, cur_psi, cur_t_mask in zip(pix[idet], psi[idet], t_mask):
                if cur_t_mask:
                    cos_over_sigma = np.cos(2 * cur_psi) * inv_sigma
                    sin_over_sigma = np.sin(2 * cur_psi) * inv_sigma
                    info_pix = nobs_matrix[cur_pix_idx]

                    # Upper triangle
                    info_pix[0, 0] += inv_sigma2
                    info_pix[0, 1] += inv_sigma * cos_over_sigma
                    info_pix[0, 2] += inv_sigma * sin_over_sigma
                    info_pix[1, 1] += cos_over_sigma * cos_over_sigma
                    info_pix[1, 2] += sin_over_sigma * cos_over_sigma
                    info_pix[2, 2] += sin_over_sigma * sin_over_sigma

        # Fill the lower triangle
        for cur_sample, cur_pix_idx, cur_psi, cur_t_mask in zip(
            tod[idet, :], pix[idet, :], psi[idet, :], t_mask
        ):
            if cur_t_mask:
                cos_over_sigma = np.cos(2 * cur_psi) * inv_sigma
                sin_over_sigma = np.sin(2 * cur_psi) * inv_sigma
                info_pix = nobs_matrix[cur_pix_idx]

                info_pix[1, 0] += cur_sample * inv_sigma2
                info_pix[2, 0] += cur_sample * cos_over_sigma * inv_sigma
                info_pix[2, 1] += cur_sample * sin_over_sigma * inv_sigma


@njit
def _numba_extract_map_and_fill_nobs_matrix(
    nobs_matrix: npt.ArrayLike, rhs: npt.ArrayLike
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


def _extract_map_and_fill_info(info: npt.ArrayLike) -> npt.ArrayLike:
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
    obs_list: List[Observation],
    ptg_list: Union[List[npt.ArrayLike], List[Callable]],
    hwp: Union[HWP, None],
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
    output_coordinate_system: CoordinateSystem,
    components: List[str],
) -> npt.ArrayLike:
    hpx = Healpix_Base(nside, "RING")
    n_pix = nside_to_npix(nside)

    nobs_matrix = np.zeros((n_pix, 3, 3))

    for obs_idx, (cur_obs, cur_ptg, cur_d_mask, cur_t_mask) in enumerate(
        zip(obs_list, ptg_list, dm_list, tm_list)
    ):
        cur_weights = get_map_making_weights(cur_obs, check=True)

        if hwp is None:
            if hasattr(cur_obs, "hwp_angle"):
                hwp_angle = cur_obs.hwp_angle
            else:
                hwp_angle = None
        else:
            if type(cur_ptg) is np.ndarray:
                hwp_angle = get_hwp_angle(cur_obs, hwp)
            else:
                logging.warning(
                    "For using an external HWP object also pass a pre-calculated pointing"
                )

        pixidx_all, polang_all = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            num_of_detectors=cur_obs.n_detectors,
            num_of_samples=cur_obs.n_samples,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
        )

        first_component = getattr(cur_obs, components[0])
        for idx, cur_component_name in enumerate(components):
            cur_component = getattr(cur_obs, cur_component_name)
            assert (
                cur_component.shape == first_component.shape
            ), 'The two TODs "{}" and "{}" do not have a matching shape'.format(
                components[0], cur_component_name
            )
            _accumulate_samples_and_build_nobs_matrix(
                cur_component,
                pixidx_all,
                polang_all,
                cur_weights,
                cur_d_mask,
                cur_t_mask,
                nobs_matrix,
                additional_component=idx > 0,
            )

        del pixidx_all, polang_all

    if all([obs.comm is None for obs in obs_list]) or not mpi.MPI_ENABLED:
        # Serial call
        pass
    elif all(
        [
            mpi.MPI.Comm.Compare(obs_list[i].comm, obs_list[i + 1].comm) < 2
            for i in range(len(obs_list) - 1)
        ]
    ):
        nobs_matrix = obs_list[0].comm.allreduce(nobs_matrix, mpi.MPI.SUM)
    else:
        raise NotImplementedError(
            "All observations must be distributed over the same MPI groups"
        )

    return nobs_matrix


def make_binned_map(
    nside: int,
    observations: Union[Observation, List[Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[HWP] = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    components: List[str] = None,
    detector_split: str = "full",
    time_split: str = "full",
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
        output_coordinate_system (:class:`.CoordinateSystem`): the coordinates
            to use for the output map
        components (list[str]): list of components to include in the map-making.
            The default is just to use the field ``tod`` of each
            :class:`.Observation` object
        detector_split (str): select the detector split to use in the map-making
        time_split (str): select the time split to use in the map-making.

    Returns:
        An instance of the class :class:`.MapMakerResult`. If the observations are
            distributed over MPI Processes, all of them get a copy of the same object.
    """

    if not components:
        components = ["tod"]

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )

    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)

    time_mask_list = _build_mask_time_split(time_split, obs_list)

    nobs_matrix = _build_nobs_matrix(
        nside=nside,
        obs_list=obs_list,
        ptg_list=ptg_list,
        hwp=hwp,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
        output_coordinate_system=output_coordinate_system,
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


def check_valid_splits(
    observations: Union[Observation, List[Observation]],
    detector_splits: Union[str, List[str]] = "full",
    time_splits: Union[str, List[str]] = "full",
):
    """Check if the splits are valid

    For each observation in the list, check if the detector and time splits
    are valid.
    In particular, the compatibility between the detectors in each observation
    and the desired split in detector domain is checked. On the other hand, this
    assess whether the desired time split fits inside the duration of the
    observation (when this applies).
    If the splits are not compatible with the input data, an error is raised.

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
        detector_splits (Union[str, List[str]], optional): detector-domain splits
            used to produce maps.

            * "full": every detector in the observation will be used;
            * "waferXXX": the mapmaking will be performed on the intersection
                of the detectors specified in the input and the detectors specified
                in the detector_split.
                The wafer must be specified in the format "waferXXX". The valid values
                for "XXX" are all the 3-digits strings corresponding to the wafers
                in the LITEBIRD focal plane (e.g. L00, M01, H02).

        time_splits (Union[str, List[str]], optional): time-domain splits
            used to produce maps. This defaults to "full" indicating that every
            sample in the observation will be used. In addition, the user can specify
            a string, or a list of strings, to indicate a subsample of the observation
            to be used:

            * "full": every sample in the observation will be used;
            * "first_half" and/or "second_half": the first and/or second half of the
                observation will be used;
            * "odd" and/or "even": the odd and/or even samples in the observation
                will be used;
            * "yearX": the samples in the observation will be
                used according to the year they belong to (relative to the
                starting time). The valid values for "X" are ["1", "2", "3"].
            * "surveyX": the samples in the observation will be used according
                to the requested survey. In this context, a survey is taken to
                be complete in 6 months, thus the valid values for "X" are
                ["1", "2", "3", "4", "5", "6"].

    """
    _check_valid_splits(observations, detector_splits, time_splits)
