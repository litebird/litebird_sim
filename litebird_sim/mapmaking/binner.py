# -*- encoding: utf-8 -*-

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

from typing import Union, List, Any, Optional
from litebird_sim.observations import Observation
from litebird_sim.coordinates import rotate_coordinates_e2g, CoordinateSystem
from litebird_sim import mpi
from ducc0.healpix import Healpix_Base
from litebird_sim.healpix import nside_to_npix

from .common import (
    _compute_pixel_indices,
    _normalize_observations_and_pointings,
    COND_THRESHOLD,
    get_map_making_weights,
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
    """

    binned_map: Any = None
    invnpp: Any = None
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic


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
    nobs_matrix: npt.ArrayLike,
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

    num_of_detectors = tod.shape[0]

    for idet in range(num_of_detectors):
        inv_sigma = 1.0 / np.sqrt(weights[idet])
        inv_sigma2 = inv_sigma * inv_sigma

        if not additional_component:
            # Fill the upper triangle
            for cur_pix_idx, cur_psi in zip(pix[idet], psi[idet]):
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
        for cur_sample, cur_pix_idx, cur_psi in zip(
            tod[idet, :], pix[idet, :], psi[idet, :]
        ):
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
    ptg_list: List[npt.ArrayLike],
    psi_list: List[npt.ArrayLike],
    output_coordinate_system: CoordinateSystem,
    components: List[str],
) -> npt.ArrayLike:
    hpx = Healpix_Base(nside, "RING")
    n_pix = nside_to_npix(nside)

    nobs_matrix = np.zeros((n_pix, 3, 3))

    for obs_idx, (cur_obs, cur_ptg, cur_psi) in enumerate(
        zip(obs_list, ptg_list, psi_list)
    ):
        cur_weights = get_map_making_weights(cur_obs, check=True)

        pixidx_all, polang_all = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            psi=cur_psi,
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
    obs: Union[Observation, List[Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    components: List[str] = None,
) -> BinnerResult:
    """Bin Map-maker

    Map a list of observations

    Args:
        obs (list of :class:`Observations`): observations to be mapped. They
            are required to have the following attributes as arrays

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
            :class:`.Observation` object.

    Returns:
        An instance of the class :class:`.MapMakerResult`. If the observations are
            distributed over MPI Processes, all of them get a copy of the same object.
    """

    if not components:
        components = ["tod"]

    obs_list, ptg_list, psi_list = _normalize_observations_and_pointings(
        obs=obs, pointings=pointings
    )

    nobs_matrix = _build_nobs_matrix(
        nside=nside,
        obs_list=obs_list,
        ptg_list=ptg_list,
        psi_list=psi_list,
        output_coordinate_system=output_coordinate_system,
        components=components,
    )

    rhs = _extract_map_and_fill_info(nobs_matrix)

    _solve_binning(nobs_matrix, rhs)

    return BinnerResult(
        binned_map=rhs.T,
        invnpp=nobs_matrix,
        coordinate_system=output_coordinate_system,
    )
