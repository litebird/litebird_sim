# WIP
# -------
# -------

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
import h5py
import os
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
from litebird_sim.healpix import UNSEEN_PIXEL_VALUE
from litebird_sim.maps_and_harmonics import HealpixMap
from litebird_sim.profiler import TimeProfiler
from litebird_sim.detectors import DetectorInfo
from .common import (
    _compute_pixel_indices,
    _build_mask_detector_split,
    _build_mask_time_split,
)


@dataclass
class h_map_Re_and_Im:
    """A single h_n,m map component for one detector"""

    real: npt.NDArray
    imag: npt.NDArray
    det_info: DetectorInfo

    @property
    def norm(self):
        return np.sqrt(self.real**2 + self.imag**2)


# @dataclass
# class h_n_m_list:
#     """All h_n,m maps for a single (n,m) pair"""
#     maps: list[h_map_Re_and_Im]  # One per detector


@dataclass
class HnMapResult:
    """Result of a call to the :func:`make_hn_maps` function

    This dataclass has the following fields:

    - ``h_maps``: Dictionnary containing the h_n maps for each spin order n,m and each detector.

    - ``coordinate_system``: the coordinate system of the output maps
      (a :class:`.CoordinateSistem` object)

    - ``detector_split``: detector split of the h map

    - ``time_split``: time split of the h map
    """

    h_maps: dict[tuple[int, int], list[h_map_Re_and_Im]]
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
    # - `nobs_matrix`: array of shape (Ndet,N_p,3), where
    #   N_p is the number of pixels in the map and Ndet the number of detectors
    # - `atd`: (Ndet,N_p, 2)
    ndet = atd.shape[0]
    npix = atd.shape[1]
    for idet in range(ndet):
        for ipix in range(npix):
            if nobs_matrix[idet, ipix, 0] != 0:
                atd[idet, ipix, 0] = atd[idet, ipix, 0] / nobs_matrix[idet, ipix, 0]
                atd[idet, ipix, 1] = atd[idet, ipix, 1] / nobs_matrix[idet, ipix, 0]
                nobs_matrix[idet, ipix, 0] = 1 / nobs_matrix[idet, ipix, 0]
            else:
                nobs_matrix[idet, ipix] = UNSEEN_PIXEL_VALUE
                atd[idet, ipix] = UNSEEN_PIXEL_VALUE


@njit
def _accumulate_spin_terms_and_build_nobs_matrix(
    n: int,
    m: int,
    pix: npt.ArrayLike,
    psi: npt.ArrayLike,
    hwp_angle: npt.ArrayLike | None,
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    nobs_matrix: npt.ArrayLike,
    num_of_detectors: int,
) -> None:
    assert pix.shape == psi.shape

    for idet in range(num_of_detectors):
        if not d_mask[idet]:
            continue

        print(np.shape(pix[idet]), "   ", np.shape(hwp_angle))
        for cur_pix_idx, cur_psi, cur_hwp_angle in zip(pix[idet], psi[idet], hwp_angle):
            info_pix = nobs_matrix[idet, cur_pix_idx]
            if hwp_angle != None:
                cos_n_m = np.cos(n * cur_psi + m * cur_hwp_angle)
                sin_n_m = np.sin(n * cur_psi + m * cur_hwp_angle)
            else:
                cos_n_m = np.cos(n * cur_psi)
                sin_n_m = np.sin(n * cur_psi)
            if (n, m) == (0, 0):
                info_pix[0] = (
                    1  # if n=m=0 we compute the hit count, beceause it is useful to combine h maps later on
                )
            else:
                info_pix[0] += 1

            info_pix[1] += cos_n_m
            info_pix[2] += sin_n_m


@njit
def _numba_extract_rhs(nobs_matrix: npt.ArrayLike, rhs: npt.ArrayLike) -> None:
    # This is used internally by _extract_map_and_fill_info.
    for idet in range(nobs_matrix.shape[0]):
        for idx in range(nobs_matrix.shape[1]):
            # Extract the vector from the lower left triangle of the 3×3 matrix
            # nobs_matrix[idx, :, :]
            rhs[idet, idx, 0] = nobs_matrix[idet, idx, 1]
            rhs[idet, idx, 1] = nobs_matrix[idet, idx, 2]


def _extract_rhs(info: npt.ArrayLike) -> npt.ArrayLike:
    # Extract the RHS of the mapmaking equation from the lower triangle of info
    # The RHS has a shape (Ndet,Np,2)
    rhs = np.empty((info.shape[0], info.shape[1], 2), dtype=info.dtype)

    # The implementation in Numba of this code is ~5 times faster than the older
    # implementation that used NumPy.
    _numba_extract_rhs(info, rhs)

    return rhs


def _compute_pixel_indices_for_all_obs(
    nside: int,
    obs_list: list[Observation],
    ptg_list: list[npt.ArrayLike] | list[Callable],
    hwp: HWP | None,
    output_coordinate_system: CoordinateSystem,
    pointings_dtype=np.float64,
) -> tuple[list[npt.ArrayLike], list[npt.ArrayLike], list[npt.ArrayLike]]:
    """Precompute pixel indices and angles for all observations.

    Returns:
        Tuple of (pixidx_list, psi_list, hwp_angle_list) where each is a list indexed by observation.
    """
    hpx = Healpix_Base(nside, "RING")

    pixidx_list = []
    psi_list = []
    hwp_angle_list = []

    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        hwp_angle = _get_hwp_angle(obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype)
        hwp_angle_list.append(hwp_angle)
        print(hwp_angle)
        pixidx_all, psi_all = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            pol_angle_detectors=np.zeros(cur_obs.n_detectors),
            num_of_detectors=cur_obs.n_detectors,
            num_of_samples=cur_obs.n_samples,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
            pointings_dtype=pointings_dtype,
            hmap_generation=True,
        )

        pixidx_list.append(pixidx_all)
        psi_list.append(psi_all)

    return pixidx_list, psi_list, hwp_angle_list


def _build_nobs_matrix(
    n: int,
    m: int,
    nside: int,
    obs_list: list[Observation],
    pixidx_list: list[npt.ArrayLike],
    psi_list: list[npt.ArrayLike],
    hwp_angle_list: list[npt.ArrayLike | None],
    dm_list: list[npt.ArrayLike],
    tm_list: list[npt.ArrayLike],
) -> npt.ArrayLike:
    """Build the nobs matrix for all detectors and pixels, it has shape (Ndet,Npix,3) and contains the accumulated spin terms and hit counts of each detector and pixel."""
    n_pix = HealpixMap.nside_to_npix(nside)

    tot_num_of_detectors = sum([len(obs.detectors_global) for obs in obs_list])
    nobs_matrix = np.zeros((tot_num_of_detectors, n_pix, 3))

    for obs_idx, (
        cur_obs,
        cur_d_mask,
        cur_t_mask,
        pixidx_all,
        psi_all,
        hwp_angle,
    ) in enumerate(
        zip(obs_list, dm_list, tm_list, pixidx_list, psi_list, hwp_angle_list)
    ):
        cur_num_of_detectors = len(cur_obs.detectors_global)

        _accumulate_spin_terms_and_build_nobs_matrix(
            n,
            m,
            pixidx_all,
            psi_all,
            hwp_angle,
            cur_d_mask,
            cur_t_mask,
            nobs_matrix,
            cur_num_of_detectors,
        )

    return nobs_matrix


def make_h_maps(
    observations: Observation | list[Observation],
    nside: int,
    pointings: np.ndarray | list[np.ndarray] | None = None,
    n_list: list[int] = [0, 2, 4],
    hwp: HWP | None = None,
    m_list: list[int] = [0],
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    detector_split: str = "full",
    time_split: str = "full",
    pointings_dtype=np.float64,
    save_to_file: bool = True,
    output_directory: str = "./h_n_maps",
) -> HnMapResult:
    """
    This function generates complex harmonic maps h_n_m for the supplied observations.
    The map h_0_0 contains the hit counts per pixel.

    Args:
        observations (list of :class:`Observations`):
            If the observations are distributed over some communicator(s), they
            must share the same group processes.
            If pointings and psi are not included in the observations, they can
            be provided through an array (or a list of arrays) of dimension
            (Ndetectors x Nsamples x 3), containing theta, phi and psi
        nside (int): HEALPix nside of the output map
        pointings (array or list of arrays): optional, external pointing
            information, if not included in the observations
        hwp (HWP, optional): An instance of the :class:`.HWP` class (optional)
        n_list(list[int]): list of the spin order which are computed
        output_coordinate_system (:class:`.CoordinateSystem`): the coordinates
            to use for the output map
        detector_split (str): select the detector split to use in the map-making
        time_split (str): select the time split to use in the map-making.
        pointings_dtype(dtype): data type for pointings generated on the fly. If
            the pointing is passed or already precomputed this parameter is
            ineffective. Default is `np.float64`.
        save_to_file(bool): If true, the h_n_maps are saved in the hd5f file format
        output_directory(str): path to directory where the h_n_maps are saved
    Returns:
        An instance of the class HnMapResult.
    """

    h_maps = {}
    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )
    all_dets_list = np.concatenate([obs.detectors_global for obs in obs_list])
    tot_num_of_detectors = np.shape(all_dets_list)[0]

    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)

    time_mask_list = _build_mask_time_split(time_split, obs_list)

    # Precompute pixel indices and angles once for all (n, m) pairs
    pixidx_list, psi_list, hwp_angle_list = _compute_pixel_indices_for_all_obs(
        nside=nside,
        obs_list=obs_list,
        ptg_list=ptg_list,
        hwp=hwp,
        output_coordinate_system=output_coordinate_system,
        pointings_dtype=pointings_dtype,
    )

    for n in n_list:
        for m in m_list:
            h_maps[(n, m)] = []

            nobs_matrix = _build_nobs_matrix(
                n,
                m,
                nside=nside,
                obs_list=obs_list,
                pixidx_list=pixidx_list,
                psi_list=psi_list,
                hwp_angle_list=hwp_angle_list,
                dm_list=detector_mask_list,
                tm_list=time_mask_list,
            )

            rhs = _extract_rhs(nobs_matrix)

            _solve_binning(nobs_matrix, rhs)

            for idet in range(tot_num_of_detectors):
                h_maps[(n, m)].append(
                    h_map_Re_and_Im(
                        real=rhs[idet].T[0],
                        imag=rhs[idet].T[1],
                        det_info=all_dets_list[idet],
                    )
                )
    result = HnMapResult(
        h_maps=h_maps,
        coordinate_system=output_coordinate_system,
        detector_split=detector_split,
        time_split=time_split,
    )
    if save_to_file:
        save_hn_maps(result, output_directory, all_dets_list)
    return result


def save_hn_maps(hn_maps, output_directory: str, dets_list) -> None:
    """Save the h_n maps to the specified output directory

    Parameters
    ----------
    output_directory : str
        Path to the output directory where to save the maps
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for n, m in hn_maps.h_maps.keys():
        with h5py.File(os.path.join(output_directory, f"h_{n, m}.h5"), "w") as f:
            f.attrs["coordinate_system"] = hn_maps.coordinate_system.name
            f.attrs["n"] = int(n)
            f.attrs["m"] = int(m)
            for _, det_map in enumerate(hn_maps.h_maps[n, m]):
                grp = f.create_group(det_map.det_info["name"])
                grp.create_dataset("Re", data=det_map.real)
                grp.create_dataset("Im", data=det_map.imag)
                for key in det_map.det_info.keys():
                    grp.attrs[key] = str(det_map.det_info[key])
