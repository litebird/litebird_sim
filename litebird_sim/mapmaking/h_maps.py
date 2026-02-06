# WIP
# -------
# -------

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
import h5py
import os
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
from litebird_sim.detectors import DetectorInfo
from .common import (
    _compute_pixel_indices_single_detector,
    _build_mask_detector_split,
    _build_mask_time_split,
)
import gc
import logging
log = logging.getLogger(__name__)

@dataclass
class h_map_Re_and_Im:
    """A single h_n,m map component for one detector and time split"""

    real: npt.NDArray
    imag: npt.NDArray
    n: int
    m: int
    det_info: str

    @property
    def norm(self):
        output=np.full(np.shape(self.real),UNSEEN_PIXEL_VALUE)
        mask=self.real !=UNSEEN_PIXEL_VALUE
        output[mask]=np.sqrt(self.real[mask]**2 + self.imag[mask]**2)
        return output


@dataclass
class HnMapResult:
    """Result of a call to the :func:`make_hn_maps` function

    This dataclass has the following fields:

    - ``h_maps``: Dictionnary containing the h_n maps for each spin order n,m and each detector.

    - ``coordinate_system``: the coordinate system of the output maps
      (a :class:`.CoordinateSystem` object)

    - ``detector_split``: detector split of the h map

    - ``time_split``: time split of the h map
    """

    h_maps: dict[list[str], list[h_map_Re_and_Im]]
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic
    detector_split: str = "full"
    time_split: str = "full"

def load_h_map_from_file(
    filepath: str,
) -> HnMapResult:
    """Load h_n maps from an HDF5 file"""
    h_maps = {}
    log.info(f"Loading h maps from file:{filepath}")
    with h5py.File(filepath, "r") as f:
        det_info = f.attrs["det"]
        h_maps[det_info] = {}
        for hn_map_key in f.keys():
            n, m = map(int, hn_map_key.split(","))
            log.debug(f"Loading h map n,m ={n},{m}")
            group = f[hn_map_key]
            h_maps[det_info][n,m] = h_map_Re_and_Im(
                real=np.array(group["Re"]),
                imag=np.array(group["Im"]),
                det_info=det_info,
                n=n,
                m=m,
            )

        return HnMapResult(
            h_maps=h_maps,
            coordinate_system=CoordinateSystem[f.attrs["coordinate_system"]],
            detector_split=f.attrs["detector_split"],
            time_split=f.attrs["time_split"],
        )


@njit
def _solve_binning(nobs_matrix, atd,n,m):
    # Solve the map-making equation
    #
    # This method alters the parameter `nobs_matrix`, so that after its completion
    # each 3×3 matrix in nobs_matrix[idx, :, :] will be the *inverse*.

    # Expected shape:
    # - `nobs_matrix`: array of shape (Ndet,N_p,3), where
    #   N_p is the number of pixels in the map and Ndet the number of detectors
    # - `atd`: (Ndet,N_p, 2)
    npix = atd.shape[0]
    for ipix in range(npix):
        if nobs_matrix[ipix, 0] != 0:
            atd[ipix, 0] = atd[ipix, 0] / nobs_matrix[ipix, 0]
            atd[ipix, 1] = atd[ipix, 1] / nobs_matrix[ipix, 0]
            nobs_matrix[ipix, 0] = 1 / nobs_matrix[ipix, 0]
        else:
            if (n, m) == (0, 0):
                nobs_matrix[ipix] = 0
                atd[ipix] = 0
            else:
                nobs_matrix[ipix] = UNSEEN_PIXEL_VALUE
                atd[ ipix] = UNSEEN_PIXEL_VALUE


@njit
def _accumulate_spin_terms_and_build_nobs_matrix(
    n: int,
    m: int,
    pix: npt.ArrayLike,
    psi: npt.ArrayLike,
    hwp_angle: npt.ArrayLike | None,
    t_mask: npt.ArrayLike,
    nobs_matrix: npt.ArrayLike,
) -> None:
    assert pix.shape == psi.shape


    for cur_pix_idx, cur_psi, cur_hwp_angle in zip(pix, psi, hwp_angle):
        info_pix = nobs_matrix[cur_pix_idx]
        if hwp_angle is not None:
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
    for idx in range(nobs_matrix.shape[0]):
            # Extract the vector from the lower left triangle of the 3×3 matrix
            # nobs_matrix[idx, :, :]
        rhs[idx, 0] = nobs_matrix[idx, 1]
        rhs[idx, 1] = nobs_matrix[idx, 2]


def _extract_rhs(n_obs: npt.ArrayLike) -> npt.ArrayLike:
    # Extract the RHS of the mapmaking equation from the lower triangle of info
    # The RHS has a shape (Ndet,Np,2)
    rhs = np.empty((n_obs.shape[0], 2), dtype=n_obs.dtype)

    # The implementation in Numba of this code is ~5 times faster than the older
    # implementation that used NumPy.
    _numba_extract_rhs(n_obs, rhs)

    return rhs



def _build_nobs_matrix(
    n: int,
    m: int,
    nside: int,
    obs: Observation,
    pointings: npt.ArrayLike,
    pixidx: npt.ArrayLike,
    polang: npt.ArrayLike,
    detector_index: int,
    hwp_angle: npt.ArrayLike,
    time_mask: npt.ArrayLike,
    output_coordinate_system: CoordinateSystem,
    pointings_dtype=np.float64,
    
) -> npt.ArrayLike:
    """Build the nobs matrix for all detectors and pixels, it has shape (Npix,3) and contains the accumulated spin terms and hit counts of each pixel for the considered detector."""
    n_pix = HealpixMap.nside_to_npix(nside)
    
    nobs_matrix = np.zeros((n_pix, 3))
    
    _accumulate_spin_terms_and_build_nobs_matrix(
            n,
            m,
            pixidx,
            polang,
            hwp_angle,
            time_mask,
            nobs_matrix,
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
    all_dets_list = np.concatenate([obs.name for obs in obs_list])

    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)

    time_mask_list = _build_mask_time_split(time_split, obs_list)

    # Precompute pixel indices and angles for all observations
    # pixidx_list, psi_list, hwp_angle_list = _compute_pixel_indices_for_all_obs(
    #     nside=nside,
    #     obs_list=obs_list,
    #     ptg_list=ptg_list,
    #     hwp=hwp,
    #     output_coordinate_system=output_coordinate_system,
    #     pointings_dtype=pointings_dtype,
    # )

    h_maps = {det: {} for det in all_dets_list}

    for (cur_obs, cur_ptg, cur_d_mask, cur_t_mask) in zip(
        obs_list, ptg_list, detector_mask_list, time_mask_list):

        for idet in range(cur_obs.n_detectors):
            if not cur_d_mask[idet]:
                continue
            log.info(f" Computing pixel indices and angles for detector {all_dets_list[idet]}" )
            
            hpx = Healpix_Base(nside, "RING")
            hwp_angle = _get_hwp_angle(obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype)

            pixidx, polang= _compute_pixel_indices_single_detector(
            hpx=hpx,
            pointings=cur_ptg,
            pol_angle_detector=cur_obs.pol_angle_rad,
            num_of_samples=cur_obs.n_samples,
            detector_index=idet,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
            pointings_dtype=pointings_dtype,
            hmap_generation=True,
    )
            for n in n_list:
                for m in m_list:
                    nobs_matrix = _build_nobs_matrix(
                        n,
                        m,
                        nside=nside,
                        obs=cur_obs,
                        pointings=cur_ptg,
                        pixidx=pixidx,
                        polang=polang,
                        detector_index=idet,
                        hwp_angle=hwp_angle,
                        time_mask=cur_t_mask,
                        output_coordinate_system=output_coordinate_system,
                        pointings_dtype=pointings_dtype,
                    )
                    rhs = _extract_rhs(nobs_matrix)
                    _solve_binning(nobs_matrix, rhs,n,m)
                    h_maps[all_dets_list[idet]][n, m] = h_map_Re_and_Im(
                            real=rhs.T[0].copy(),
                            imag=rhs.T[1].copy(),
                            det_info=all_dets_list[idet],
                            n=n,
                            m=m,
                        )
                    log.info(f"  h_map n={n} m={m} for detector {all_dets_list[idet]} computed." )
                    del rhs
                    del nobs_matrix
                    gc.collect()

    result = HnMapResult(
        h_maps=h_maps,
        coordinate_system=output_coordinate_system,
        detector_split=detector_split,
        time_split=time_split,
    )
    if save_to_file:
        save_hn_maps(result, output_directory)
        log.info(f"h_n maps saved to directory: {output_directory}")
    return result


def save_hn_maps(result, output_directory: str) -> None:
    """Save the h_n maps to the specified output directory

    Parameters
    ----------
    output_directory : str
        Path to the output directory where to save the maps
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for det in result.h_maps.keys():
        with h5py.File(
            os.path.join(output_directory, f"h_maps_det_{det}.h5"), "w"
        ) as f:
            f.attrs["coordinate_system"] = result.coordinate_system.name
            f.attrs["det"] = str(det)
            f.attrs["detector_split"] = result.detector_split
            f.attrs["time_split"] = result.time_split
            for hn_map in result.h_maps[det].values():
                grp = f.create_group(f"{hn_map.n},{hn_map.m}")
                grp.create_dataset("Re", data=hn_map.real)
                grp.create_dataset("Im", data=hn_map.imag)
