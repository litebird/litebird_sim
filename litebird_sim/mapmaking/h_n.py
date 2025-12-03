# WIP
# -------
# -------

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
import healpy as hp

from typing import Any
from collections.abc import Callable
from litebird_sim.observations import Observation
from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.pointings_in_obs import (
    _get_hwp_angle,
    _normalize_observations_and_pointings,
)
from litebird_sim.hwp import HWP
from litebird_sim import mpi
from ducc0.healpix import Healpix_Base
from litebird_sim.healpix import nside_to_npix


from .common import (
    _compute_pixel_indices,
    COND_THRESHOLD,
    get_map_making_weights,
    _build_mask_detector_split,
    _build_mask_time_split,
    _check_valid_splits,
)


@dataclass
class HnMapResult:
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
    detector_split: str = "full"
    time_split: str = "full"


@njit
def _solve_binning(nobs_matrix, atd):
    # Solve the map-making equation
    #
    # This method alters the parameter `nobs_matrix`, so that after its completion
    # each 3×3 matrix in nobs_matrix[idx, :, :] will be the *inverse*.

    # Expected shape:
    # - `nobs_matrix`: (N_p,3) is an array of N_p×3 matrices, where
    #   N_p is the number of pixels in the map
    # - `atd`: (N_p, 2)
    npix = atd.shape[0]
    for ipix in range(npix):
        if nobs_matrix[ipix,0] != 0:
            atd[ipix,0] = atd[ipix,0]/nobs_matrix[ipix,0]
            atd[ipix,1] = atd[ipix,1]/nobs_matrix[ipix,0] 
            nobs_matrix[ipix,0] =1/nobs_matrix[ipix,0]
        else:
            nobs_matrix[ipix]=hp.UNSEEN
            atd[ipix]= hp.UNSEEN


@njit
def _accumulate_samples_and_build_nobs_matrix(
    n: int,
    pix: npt.ArrayLike,
    psi: npt.ArrayLike,
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    nobs_matrix: npt.ArrayLike,
    num_of_detectors: int,

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

    assert pix.shape == psi.shape



    for idet in range(num_of_detectors):
        if not d_mask[idet]:
            continue


        # Fill the upper triangle
        for cur_pix_idx, cur_psi, cur_t_mask in zip(pix[idet], psi[idet], t_mask):

                info_pix = nobs_matrix[cur_pix_idx]

                cos_n = np.cos(n * cur_psi) 
                sin_n = np.sin(n * cur_psi)
                
                info_pix[0] += 1
                info_pix[1] += cos_n 
                info_pix[2] += sin_n


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
        rhs[idx, 0] = nobs_matrix[idx, 1]
        rhs[idx, 1] = nobs_matrix[idx, 2]



def _extract_map_and_fill_info(info: npt.ArrayLike) -> npt.ArrayLike:
    # Extract the RHS of the mapmaking equation from the lower triangle of info
    # and fill the lower triangle with the upper triangle, thus making each
    # matrix in "info" symmetric
    rhs = np.empty((info.shape[0], 2), dtype=info.dtype)

    # The implementation in Numba of this code is ~5 times faster than the older
    # implementation that used NumPy.
    _numba_extract_map_and_fill_nobs_matrix(info, rhs)

    return rhs


def _build_nobs_matrix(
    n:int,
    nside: int,
    obs_list: list[Observation],
    ptg_list: list[npt.ArrayLike] | list[Callable],
    hwp: HWP | None,
    dm_list: list[npt.ArrayLike],
    tm_list: list[npt.ArrayLike],
    output_coordinate_system: CoordinateSystem,
    pointings_dtype=np.float64,
) -> npt.ArrayLike:
    hpx = Healpix_Base(nside, "RING")
    n_pix = nside_to_npix(nside)

    nobs_matrix = np.zeros((n_pix,3))

    for obs_idx, (cur_obs, cur_ptg, cur_d_mask, cur_t_mask) in enumerate(
        zip(obs_list, ptg_list, dm_list, tm_list)
    ):

        # Determine the HWP angle to use:
        # - If an external HWP object is provided, compute the angle from it
        # - If not, compute or retrieve the HWP angle from the observation, depending on availability
        hwp_angle = _get_hwp_angle(obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype)

        pixidx_all, polang_all = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            pol_angle_detectors=np.zeros(cur_obs.n_detectors), #cur_obs.pol_angle_rad
            num_of_detectors=cur_obs.n_detectors,
            num_of_samples=cur_obs.n_samples,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
            pointings_dtype=pointings_dtype,
        )
        num_of_detectors = len(cur_obs.detectors_global)

        _accumulate_samples_and_build_nobs_matrix(
            n,
            pixidx_all,
            polang_all,
            cur_d_mask,
            cur_t_mask,
            nobs_matrix,
            num_of_detectors,
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
        obs_list[0].comm.Allreduce(mpi.MPI.IN_PLACE, nobs_matrix, mpi.MPI.SUM)

    else:
        raise NotImplementedError(
            "All observations must be distributed over the same MPI groups"
        )

    return nobs_matrix

def get_exp_i_pi_n_psi(obs,det_num,n):
    pointings=obs.get_pointings()[0][det_num][:,2]
    cos_n=np.cos(n*pointings)
    sin_n=np.sin(n*pointings)
    return cos_n,sin_n

def make_h_map(
    n,
    nside: int,
    observations: Observation | list[Observation],
    hwp: HWP | None = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    detector_split: str = "full",
    time_split: str = "full",
    pointings_dtype=np.float64,
) -> HnMapResult:
    """
    Compute h_n maps 
    """

    # if isinstance(observations, Observation):
    #     obs_list = [observations]
    # else:
    #     obs_list = observations
    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=None
    )
    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)

    time_mask_list = _build_mask_time_split(time_split, obs_list)

    nobs_matrix = _build_nobs_matrix(
        n,
        nside=nside,
        obs_list=obs_list,
        ptg_list=ptg_list,
        hwp=hwp,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
        output_coordinate_system=output_coordinate_system,
        pointings_dtype=pointings_dtype,
    )

    rhs = _extract_map_and_fill_info(nobs_matrix)

    _solve_binning(nobs_matrix, rhs)

    return HnMapResult(
        binned_map=rhs.T,
        invnpp=nobs_matrix,
        coordinate_system=output_coordinate_system,
        detector_split=detector_split,
        time_split=time_split,
    )


def check_valid_splits(
    observations: Observation | list[Observation],
    detector_splits: str | list[str] = "full",
    time_splits: str | list[str] = "full",
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
               samples in the TOD. @paganol paganol linked an issue 3 days ago that may be closed by this pull request 
            * any attribute listed in `components` (by default, `tod`) @paganol paganol linked an issue 3 days ago that may be closed by this pull request  and
              containing the TOD(s) to be binned together.

            If the observations are distributed over some communicator(s), they
            must share the same group processes.
            If pointings and psi are not included in the observations, they can
            be provided through an array (or a list of arrays) of dimension
            (Ndetectors x Nsamples x 3), containing theta, phi and psi
        detector_splits (str | list[str], optional): detector-domain splits
            used to produce maps.

            * "full": every detector in the observation will be used;
            * "waferXXX": the mapmaking will be performed on the intersection
                of the detectors specified in the input and the detectors specified
                in the detector_split.
                The wafer must be specified in the format "waferXXX". The valid values
                for "XXX" are all the 3-digits strings corresponding to the wafers
                in the LITEBIRD focal plane (e.g. L00, M01, H02).

        time_splits (str | list[str], optional): time-domain splits
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