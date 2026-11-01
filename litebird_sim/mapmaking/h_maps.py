import gc
import logging
import os
from dataclasses import dataclass

import h5py
import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base
from numba import njit, prange
from astropy.time import TimeDelta

from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.healpix import UNSEEN_PIXEL_VALUE
from litebird_sim.hwp import HWP
from litebird_sim.maps_and_harmonics import HealpixMap
from litebird_sim.observations import Observation
from litebird_sim.observation_utilities import (
    _get_hwp_angle,
    _normalize_observations_and_pointings,
)
from litebird_sim.utilities import cos_sin, resolve_nthreads

from .common import (
    _build_mask_detector_split,
    _build_mask_time_split,
    _compute_pixel_indices_single_detector,
    _get_reduction_comm,
)

log = logging.getLogger(__name__)


@dataclass
class h_map_Re_and_Im:
    """Single :math:`h_{n,m}` map component for one detector.

    :ivar real: Real part of :math:`h_{n,m}`.
    :type real: npt.NDArray
    :ivar imag: Imaginary part of :math:`h_{n,m}`.
    :type imag: npt.NDArray
    :ivar n: Spin index :math:`n`.
    :type n: int
    :ivar m: Spin index :math:`m`.
    :type m: int
    :ivar det_info: Detector name associated with this map.
    :type det_info: str
    """

    real: npt.NDArray
    imag: npt.NDArray
    n: int
    m: int
    det_info: str

    @property
    def norm(self):
        """Return :math:`|h_{n,m}|` per pixel.

        :returns: Pixel-wise magnitude computed from ``real`` and ``imag``.
        :rtype: npt.NDArray
        """
        output = np.full(np.shape(self.real), UNSEEN_PIXEL_VALUE)
        mask = self.real != UNSEEN_PIXEL_VALUE
        output[mask] = np.sqrt(self.real[mask] ** 2 + self.imag[mask] ** 2)
        return output


@dataclass
class HMapsResult:
    """Result of :func:`make_h_maps`.

    :ivar h_maps: Mapping ``detector -> {(n, m): h_map_Re_and_Im}`` containing
        all computed :math:`h_{n,m}` maps.
    :type h_maps: dict[str, dict[tuple[int, int], h_map_Re_and_Im]]
    :ivar duration_s: Duration of the simulated observation in seconds.
    :type duration_s: float
    :ivar sampling_rate_Hz: Sampling rate in hertz.
    :type sampling_rate_Hz: float
    :ivar coordinate_system: Coordinate system of the output maps.
    :type coordinate_system: CoordinateSystem
    :ivar detector_split: Detector split used to build the maps.
    :type detector_split: str
    :ivar time_split: Time split used to build the maps.
    :type time_split: str
    """

    h_maps: dict[str, dict[tuple[int, int], h_map_Re_and_Im]]
    duration_s: float
    sampling_rate_Hz: float
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic

    detector_split: str = "full"
    time_split: str = "full"


def load_h_maps_from_file(
    filepath: str,
) -> HMapsResult:
    """Load :math:`h_{n,m}` maps from an HDF5 file.

    :param filepath: Path to an HDF5 file produced by :func:`save_hn_maps`.
    :type filepath: str
    :returns: Loaded maps and metadata.
    :rtype: HnMapResult
    """
    h_maps = {}
    log.info(f"Loading h maps from file:{filepath}")
    with h5py.File(filepath, "r") as f:
        det_info = f.attrs["det"]
        h_maps[det_info] = {}
        for hn_map_key in f.keys():
            n, m = map(int, hn_map_key.split(","))
            log.debug(f"Loading h map n,m ={n},{m}")
            group = f[hn_map_key]
            h_maps[det_info][n, m] = h_map_Re_and_Im(
                real=np.array(group["Re"]),
                imag=np.array(group["Im"]),
                det_info=det_info,
                n=n,
                m=m,
            )

        return HMapsResult(
            h_maps=h_maps,
            coordinate_system=CoordinateSystem[f.attrs["coordinate_system"]],
            detector_split=f.attrs["detector_split"],
            time_split=f.attrs["time_split"],
            duration_s=f.attrs["duration_s"],
            sampling_rate_Hz=f.attrs["sampling_rate_Hz"],
        )


@njit(parallel=True)
def _solve_binning(nobs_matrix, atd):
    # Normalize the accumulated cosine/sine sums, in place in `atd`.
    # Shapes are (Npix, 3) for nobs_matrix and (Npix, 2) for atd.
    npix = atd.shape[0]
    for ipix in prange(npix):  # type: ignore[not-iterable]
        if nobs_matrix[ipix, 0] != 0:
            atd[ipix, 0] = atd[ipix, 0] / nobs_matrix[ipix, 0]
            atd[ipix, 1] = atd[ipix, 1] / nobs_matrix[ipix, 0]
        else:
            atd[ipix] = 0  # If there is no hit we set the pixel to 0


@njit
def _accumulate_spin_terms_and_build_nobs_matrix(
    n: int,
    m: int,
    pix: npt.NDArray,
    psi: npt.NDArray,
    hwp_angle: npt.NDArray,
    t_mask: npt.NDArray,
    nobs_matrix: npt.NDArray,
) -> None:
    assert pix.shape == psi.shape

    for cur_pix_idx, cur_psi, cur_hwp_angle, cur_t_mask in zip(
        pix, psi, hwp_angle, t_mask
    ):
        if cur_t_mask:
            info_pix = nobs_matrix[cur_pix_idx]

            cos_n_m, sin_n_m = cos_sin(n * cur_psi + m * cur_hwp_angle)
            if (n, m) == (0, 0):
                # Keep the normalization at one across all observations so
                # the cosine sum remains a hit count after solving.
                info_pix[0] = 1
            else:
                info_pix[0] += 1
            info_pix[1] += cos_n_m
            info_pix[2] += sin_n_m


@njit
def _numba_extract_rhs(nobs_matrix: npt.NDArray, rhs: npt.NDArray) -> None:
    for idx in range(nobs_matrix.shape[0]):
        # Extract the accumulated cosine and sine sums.
        rhs[idx, 0] = nobs_matrix[idx, 1]
        rhs[idx, 1] = nobs_matrix[idx, 2]


def _extract_rhs(n_obs: npt.NDArray) -> npt.NDArray:
    # The RHS contains the spin sums and has shape (Npix, 2).
    rhs = np.empty((n_obs.shape[0], 2), dtype=n_obs.dtype)

    # The implementation in Numba of this code is ~5 times faster than the older
    # implementation that used NumPy.
    _numba_extract_rhs(n_obs, rhs)

    return rhs


def _build_nobs_matrix(
    n: int,
    m: int,
    nside: int,
    pixidx: npt.NDArray,
    polang: npt.NDArray,
    hwp_angle: npt.NDArray | None,
    time_mask: npt.NDArray,
    nobs_matrix: npt.NDArray | None = None,
) -> npt.NDArray:
    """Accumulate one detector's spin terms and normalization in a (Npix, 3)
    matrix. Pass an existing matrix to include another observation before
    solving; otherwise a zero-filled matrix is allocated.
    """
    if nobs_matrix is None:
        n_pix = HealpixMap.nside_to_npix(nside)
        nobs_matrix = np.zeros((n_pix, 3))

    if hwp_angle is None:
        hwp_angle = np.zeros_like(polang)
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
    n_m_couples: np.ndarray = np.array(np.meshgrid([0, 2, 4], [0])).T.reshape(-1, 2),
    hwp: HWP | None = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    detector_split: str = "full",
    time_split: str = "full",
    pointings_dtype=np.float64,
    save_to_file: bool = True,
    output_directory: str = "./h_n_maps",
    nthreads: int | None = None,
) -> HMapsResult:
    """Generate complex harmonic maps :math:`h_{n,m}` from observations.

    The map for ``(n, m) = (0, 0)`` contains hit counts per pixel.
    Observations with the same detector name contribute to one map: their
    spin terms and normalizations are accumulated before solving.

    :param observations: Observation object or list of observations used to build
        the maps.
    :type observations: Observation | list[Observation]
    :param nside: HEALPix ``nside`` of the output maps.
    :type nside: int
    :param pointings: Optional external pointings. Expected shape is
        ``(n_detectors, n_samples, 3)`` per observation.
    :type pointings: np.ndarray | list[np.ndarray] | None
    :param n_m_couples: Array of ``(n, m)`` pairs with shape ``(N, 2)``.
    :type n_m_couples: np.ndarray
    :param hwp: Half-wave plate model.
    :type hwp: HWP | None
    :param output_coordinate_system: Coordinate system of the output maps.
    :type output_coordinate_system: CoordinateSystem
    :param detector_split: Detector split to apply.
    :type detector_split: str
    :param time_split: Time split to apply.
    :type time_split: str
    :param pointings_dtype: Data type used when pointings are computed on the fly.
    :type pointings_dtype: type
    :param save_to_file: If ``True``, save maps to HDF5 files.
    :type save_to_file: bool
    :param output_directory: Output directory for generated HDF5 files.
    :type output_directory: str
    :param nthreads: Number of threads to use for ducc0 calls. If None,
        resolved via :func:`.resolve_nthreads`.
    :type nthreads: int | None
    :returns: Result container with all computed maps and metadata.
    :rtype: HnMapResult
    """

    nthreads = resolve_nthreads(nthreads)

    assert (
        n_m_couples.shape[1] == 2
    ), """the n,m couples should be passed in this shape: array([[n1, m1],
       [n2, m2],
       [n2, m3]])"""

    assert detector_split == "full", "detectors split over wafers is not implemented"

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )
    if not obs_list:
        raise ValueError("At least one observation is required to compute h maps")

    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)
    time_mask_list = _build_mask_time_split(time_split, obs_list)

    # Store local indices with their observation; detector order and membership
    # may change between observations. Grouping also bounds the working matrices
    # to one detector, rather than keeping matrices for the whole focal plane.
    detector_observations = {}
    for obs_idx, (cur_obs, cur_d_mask) in enumerate(zip(obs_list, detector_mask_list)):
        assert cur_d_mask.ravel().any(), (
            "No detectors is selected by the current detector split"
        )
        assert cur_obs.n_blocks_time == 1, (
            "The current implementation of make_h_maps only supports one time block per observation, you can set several detector blocks though."
        )
        duration_s = cur_obs.get_time_span()
        if isinstance(duration_s, TimeDelta):
            duration_s = duration_s.to_value("sec")
        assert isinstance(duration_s, float)
        sampling_rate = cur_obs.sampling_rate_hz
        for idet in range(cur_obs.n_detectors):
            if not cur_d_mask[idet]:
                log.info(f"skipping detector {idet}")
                continue
            det = cur_obs.name[idet]
            detector_observations.setdefault(det, []).append((obs_idx, idet))

    # Observations split in time over the MPI processes put the same detector
    # on several processes. Each detector is owned by the lowest rank that
    # observes it: its partial sums are reduced there, and only the owner
    # solves, keeps and saves the maps of that detector.
    comm = _get_reduction_comm(obs_list)
    rank = comm.rank if comm is not None else 0
    if comm is not None and comm.size > 1:
        dets_per_rank = comm.allgather(list(detector_observations))
    else:
        dets_per_rank = [list(detector_observations)]
    owner = {}
    shared_dets = set()
    for cur_rank, dets in enumerate(dets_per_rank):
        for det in dets:
            if det in owner:
                shared_dets.add(det)
            else:
                owner[det] = cur_rank

    # Repeated spin pairs must not accumulate the same samples twice.
    spin_pairs = list(dict.fromkeys(map(tuple, n_m_couples)))
    h_maps = {det: {} for det in owner if owner[det] == rank}
    hpx = Healpix_Base(nside, "RING")
    n_pix = HealpixMap.nside_to_npix(nside)
    # Same order on all processes, as the reductions are collective
    for det in sorted(owner, key=str):
        is_shared = det in shared_dets
        if owner[det] != rank and not is_shared:
            continue

        nobs_matrices = {pair: np.zeros((n_pix, 3)) for pair in spin_pairs}
        for obs_idx, idet in detector_observations.get(det, []):
            cur_obs = obs_list[obs_idx]
            cur_ptg = ptg_list[obs_idx]
            cur_t_mask = time_mask_list[obs_idx]
            log.info(f"Computing pixel indices and angles for detector {det}")

            hwp_angle = _get_hwp_angle(
                obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype
            )
            pixidx, polang = _compute_pixel_indices_single_detector(
                hpx=hpx,
                pointings=cur_ptg,
                pol_angle_detector=cur_obs.pol_angle_rad[idet],
                num_of_samples=cur_obs.n_samples,
                detector_index=idet,
                hwp_angle=hwp_angle,
                output_coordinate_system=output_coordinate_system,
                pointings_dtype=pointings_dtype,
                hmap_generation=True,
                nthreads=nthreads,
            )
            log.info(
                f"Pixel indices and angles for detector {det} computed, now accumulating nobs matrices"
            )

            for n, m in spin_pairs:
                nobs_matrices[n, m] = _build_nobs_matrix(
                    n,
                    m,
                    nside=nside,
                    pixidx=pixidx,
                    polang=polang,
                    hwp_angle=hwp_angle,
                    time_mask=cur_t_mask,
                    nobs_matrix=nobs_matrices[n, m],
                )
            del pixidx
            del polang
            gc.collect()

        if is_shared:
            from litebird_sim.mpi import MPI

            for (n, m), nobs_matrix in nobs_matrices.items():
                if rank == owner[det]:
                    comm.Reduce(MPI.IN_PLACE, nobs_matrix, op=MPI.SUM, root=rank)
                    if (n, m) == (0, 0):
                        # Each process set the normalization to one in the
                        # pixels it observed: keep it at one after the sum
                        np.minimum(nobs_matrix[:, 0], 1.0, out=nobs_matrix[:, 0])
                else:
                    comm.Reduce(nobs_matrix, None, op=MPI.SUM, root=owner[det])

        if owner[det] != rank:
            continue

        # Solve once, after every observation of this detector has contributed.
        for n, m in spin_pairs:
            nobs_matrix = nobs_matrices.pop((n, m))
            rhs = _extract_rhs(nobs_matrix)
            _solve_binning(nobs_matrix, rhs)
            h_maps[det][n, m] = h_map_Re_and_Im(
                real=rhs.T[0].copy(),
                imag=rhs.T[1].copy(),
                det_info=det,
                n=n,
                m=m,
            )
            log.info(f"h_map n={n} m={m} for detector {det} computed.")
            del rhs
            del nobs_matrix
            gc.collect()

    result = HMapsResult(
        h_maps=h_maps,
        coordinate_system=output_coordinate_system,
        duration_s=duration_s,
        sampling_rate_Hz=sampling_rate,
        detector_split=detector_split,
        time_split=time_split,
    )
    if save_to_file:
        save_h_maps(result, output_directory)
        log.info(f"h_n maps saved to directory: {output_directory}")
    return result


def save_h_maps(result, output_directory: str) -> None:
    """Save :math:`h_n` maps to HDF5 files.

    One file is produced per detector in ``output_directory``. The directory is
    created if it does not exist.

    :param result: Container with maps and metadata to save.
    :type result: HnMapResult
    :param output_directory: Destination directory for output HDF5 files.
    :type output_directory: str
    :returns: ``None``
    :rtype: None
    """

    # Several MPI processes may create the directory at the same time
    os.makedirs(output_directory, exist_ok=True)
    for det in result.h_maps.keys():
        with h5py.File(
            os.path.join(output_directory, f"h_maps_det_{det}.h5"), "w"
        ) as f:
            f.attrs["coordinate_system"] = result.coordinate_system.name
            f.attrs["det"] = str(det)
            f.attrs["detector_split"] = result.detector_split
            f.attrs["time_split"] = result.time_split
            f.attrs["duration_s"] = result.duration_s
            f.attrs["sampling_rate_Hz"] = result.sampling_rate_Hz
            for hn_map in result.h_maps[det].values():
                grp = f.create_group(f"{hn_map.n},{hn_map.m}")
                grp.create_dataset("Re", data=hn_map.real)
                grp.create_dataset("Im", data=hn_map.imag)
