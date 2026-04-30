# The implementation of the binning algorithm provided here is derived
# from the more general destriping equation presented in the paper
# «Destriping CMB temperature and polarization maps» by Kurki-Suonio et al. 2009,
# A&A 506, 1511–1539 (2009), https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code, as many
# functions and variable defined here use the same letters and symbols of that
# paper. We refer to it in code comments and docstrings as "KurkiSuonio2009".

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import healpy as hp
import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base
from numba import njit

from litebird_sim import mpi
from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.hwp import HWP
from litebird_sim.observations import Observation
from litebird_sim.pointings_in_obs import (
    _get_hwp_angle,
    _normalize_observations_and_pointings,
)
from litebird_sim.maps_and_harmonics import HealpixMap


from .common import (
    COND_THRESHOLD,
    _build_mask_detector_split,
    _build_mask_time_split,
    _check_valid_splits,
    _compute_pixel_indices,
    get_map_making_weights,
)


@dataclass
class PairDifferencingResult:
    """Result of a call to the :func:`.make_pair_differenced_map` function

    This dataclass has the following fields:

        - ``binned_map``: Healpix map containing the binned value for each pixel

        - ``invnpp``: inverse of the covariance matrix element for each
            pixel in the map. It is an array of shape `(12 * nside * nside, 2, 2)`

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


@njit
def _solve_binning(nobs_matrix, atd):
    # Solve the map-making equation
    #
    # This method alters the parameter `nobs_matrix`, so that after its completion
    # each matrix in nobs_matrix[idx, :, :] will be the *inverse*.

    # Expected shape:
    # - `nobs_matrix`: (N_p, N_c, N_c) is an array of N_p square matrices, where
    #   N_p is the number of pixels in the map
    # - `atd`: (N_p, N_c)
    npix = atd.shape[0]

    for ipix in range(npix):
        if np.linalg.cond(nobs_matrix[ipix]) < COND_THRESHOLD:
            atd[ipix] = np.linalg.solve(nobs_matrix[ipix], atd[ipix])
            nobs_matrix[ipix] = np.linalg.inv(nobs_matrix[ipix])
        else:
            nobs_matrix[ipix].fill(hp.UNSEEN)
            atd[ipix].fill(hp.UNSEEN)


@njit
def _accumulate_pair_differenced_samples_and_build_nobs_matrix(
    tod_t: npt.NDArray,
    tod_b: npt.NDArray,
    pix_t: npt.NDArray,
    psi_t: npt.NDArray,
    psi_b: npt.NDArray,
    weight_t: float,
    weight_b: float,
    t_mask: npt.NDArray,
    nobs_matrix: npt.NDArray,
    rhs: npt.NDArray,
    *,
    additional_component: bool,
) -> None:
    # Fill the QU pair-differencing normal matrix and RHS.
    #
    # For a T/B pair, the differenced TOD is modeled as
    #
    #   d_T - d_B = Q·(cos(2ψ_T) - cos(2ψ_B)) + U·(sin(2ψ_T) - sin(2ψ_B))
    #
    # The pair weight is defined as the average of the two detector weights
    # ψ is the proper sum of the detector polarization angle and the HWP angle (if present)

    assert (
        tod_t.shape
        == tod_b.shape
        == pix_t.shape
        == psi_t.shape
        == psi_b.shape
        == t_mask.shape
    )

    pair_weight = 0.5 * (weight_t + weight_b)
    inv_sigma = 1.0 / np.sqrt(pair_weight)
    inv_sigma2 = inv_sigma * inv_sigma

    if not additional_component:
        for cur_pix_idx, cur_psi_t, cur_psi_b, cur_t_mask in zip(
            pix_t, psi_t, psi_b, t_mask
        ):
            if cur_t_mask:
                pair_cos = np.cos(2 * cur_psi_t) - np.cos(2 * cur_psi_b)
                pair_sin = np.sin(2 * cur_psi_t) - np.sin(2 * cur_psi_b)
                info_pix = nobs_matrix[cur_pix_idx]

                info_pix[0, 0] += pair_cos * pair_cos * inv_sigma2
                info_pix[0, 1] += pair_cos * pair_sin * inv_sigma2
                info_pix[1, 0] += pair_cos * pair_sin * inv_sigma2
                info_pix[1, 1] += pair_sin * pair_sin * inv_sigma2

    for (
        cur_sample_t,
        cur_sample_b,
        cur_pix_idx,
        cur_psi_t,
        cur_psi_b,
        cur_t_mask,
    ) in zip(tod_t, tod_b, pix_t, psi_t, psi_b, t_mask):
        if cur_t_mask:
            pair_sample = cur_sample_t - cur_sample_b
            pair_cos = np.cos(2 * cur_psi_t) - np.cos(2 * cur_psi_b)
            pair_sin = np.sin(2 * cur_psi_t) - np.sin(2 * cur_psi_b)
            rhs_pix = rhs[cur_pix_idx]

            rhs_pix[0] += pair_sample * pair_cos * inv_sigma2
            rhs_pix[1] += pair_sample * pair_sin * inv_sigma2


def _build_nobs_matrix(
    nside: int,
    obs_list: list[Observation],
    ptg_list: list[npt.NDArray | Callable],
    hwp: HWP | None,
    dm_list: list[npt.NDArray],
    tm_list: list[npt.NDArray],
    output_coordinate_system: CoordinateSystem,
    components: list[str],
    pointings_dtype=np.float64,
) -> tuple[npt.NDArray, npt.NDArray]:
    hpx = Healpix_Base(nside, "RING")
    n_pix = HealpixMap.nside_to_npix(nside)

    nobs_matrix = np.zeros((n_pix, 2, 2))
    rhs = np.zeros((n_pix, 2))

    for obs_idx, (cur_obs, cur_ptg, cur_d_mask, cur_t_mask) in enumerate(
        zip(obs_list, ptg_list, dm_list, tm_list)
    ):
        cur_weights = get_map_making_weights(cur_obs, check=True)

        # Determine the HWP angle to use:
        # - If an external HWP object is provided, compute the angle from it
        # - If not, compute or retrieve the HWP angle from the observation, depending on availability
        hwp_angle = _get_hwp_angle(obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype)

        pixidx_all, polang_all = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            pol_angle_detectors=cur_obs.pol_angle_rad,
            num_of_detectors=cur_obs.n_detectors,
            num_of_samples=cur_obs.n_samples,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
            pointings_dtype=pointings_dtype,
        )

        detector_pairs = _get_detector_pairs(
            cur_obs, detector_mask=cur_d_mask, obs_idx=obs_idx
        )

        for t_idx, b_idx in detector_pairs:
            if not np.array_equal(
                pixidx_all[t_idx, cur_t_mask], pixidx_all[b_idx, cur_t_mask]
            ):
                raise ValueError(
                    f"Observation {obs_idx}: detectors {t_idx} and {b_idx} do not "
                    "observe the same map pixels for the selected time samples."
                )

        first_component = getattr(cur_obs, components[0])
        for idx, cur_component_name in enumerate(components):
            cur_component = getattr(cur_obs, cur_component_name)
            assert cur_component.shape == first_component.shape, (
                'The two TODs "{}" and "{}" do not have a matching shape'.format(
                    components[0], cur_component_name
                )
            )
            for t_idx, b_idx in detector_pairs:
                _accumulate_pair_differenced_samples_and_build_nobs_matrix(
                    cur_component[t_idx],
                    cur_component[b_idx],
                    pixidx_all[t_idx],
                    polang_all[t_idx],
                    polang_all[b_idx],
                    cur_weights[t_idx],
                    cur_weights[b_idx],
                    cur_t_mask,
                    nobs_matrix,
                    rhs,
                    additional_component=idx > 0,
                )

        del pixidx_all, polang_all

    assert obs_list, "No observations provided"
    if mpi.MPI_ENABLED:
        from litebird_sim.mpi import MPI
    if all([obs.comm is None for obs in obs_list]) or not mpi.MPI_ENABLED:
        # Serial call
        pass
    elif all(
        [
            MPI.Comm.Compare(obs_list[i].comm, obs_list[i + 1].comm) < 2
            for i in range(len(obs_list) - 1)
        ]
    ):
        obs_list[0].comm.Allreduce(MPI.IN_PLACE, nobs_matrix, MPI.SUM)
        obs_list[0].comm.Allreduce(MPI.IN_PLACE, rhs, MPI.SUM)

    else:
        raise NotImplementedError(
            "All observations must be distributed over the same MPI groups"
        )

    return nobs_matrix, rhs


def _get_detector_pairs(
    obs: Observation,
    detector_mask: npt.NDArray | None = None,
    *,
    obs_idx: int | None = None,
) -> list[tuple[int, int]]:
    pol_values = getattr(obs, "pol", None)
    pixel_values = getattr(obs, "pixel", None)
    wafer_values = getattr(obs, "wafer", None)

    for attr_name, attr_values in (
        ("pol", pol_values),
        ("pixel", pixel_values),
        ("wafer", wafer_values),
    ):
        if attr_values is None:
            obs_label = "Observation" if obs_idx is None else f"Observation {obs_idx}"
            raise ValueError(
                f"{obs_label} is missing the '{attr_name}' detector attribute required "
                "for pair-differencing map-making."
            )

    assert pol_values is not None
    assert pixel_values is not None
    assert wafer_values is not None

    pairs: dict[tuple[Any, Any], dict[str, int]] = {}
    for det_idx in range(obs.n_detectors):
        if detector_mask is not None and not detector_mask[det_idx]:
            continue

        wafer = wafer_values[det_idx]
        pixel = pixel_values[det_idx]
        pol = pol_values[det_idx]

        if wafer is None or pixel is None or pol is None:
            obs_label = "observation" if obs_idx is None else f"observation {obs_idx}"
            raise ValueError(
                f"Detector {det_idx} in {obs_label} has unset 'wafer', 'pixel', "
                "or 'pol' attribute."
            )

        if pol not in ("T", "B"):
            obs_label = "Observation" if obs_idx is None else f"Observation {obs_idx}"
            raise ValueError(
                f"{obs_label}: detector {det_idx} has unsupported polarization '{pol}'."
            )

        pair_key = (wafer, pixel)
        if pair_key not in pairs:
            pairs[pair_key] = {}

        if pol in pairs[pair_key]:
            obs_label = "Observation" if obs_idx is None else f"Observation {obs_idx}"
            raise ValueError(
                f"{obs_label}: wafer '{wafer}', pixel {pixel} has more than one '{pol}' detector."
            )

        pairs[pair_key][pol] = det_idx

    detector_pairs = []
    for (wafer, pixel), pol_to_det in pairs.items():
        if set(pol_to_det) != {"T", "B"}:
            obs_label = "Observation" if obs_idx is None else f"Observation {obs_idx}"
            found_pols = sorted(pol_to_det)
            raise ValueError(
                f"{obs_label}: detectors in wafer '{wafer}', pixel {pixel} do not form "
                f"a valid T/B pair (found polarizations: {found_pols})."
            )

        detector_pairs.append((pol_to_det["T"], pol_to_det["B"]))

    return detector_pairs


def _check_tb_detector_pairs(
    obs_list: list[Observation],
    detector_mask_list: list[npt.NDArray | None] | None = None,
) -> None:
    """Verify that detectors form valid T/B pairs sharing the same wafer and pixel.

    For each observation, every unique (wafer, pixel) combination must contain
    exactly two detectors: one with ``pol='T'`` and one with ``pol='B'``.

    Raises:
        ValueError: if any observation is missing the ``pol``, ``pixel``, or
            ``wafer`` detector attribute, or if any (wafer, pixel) group does
            not contain exactly one T and one B detector.
    """
    if detector_mask_list is None:
        detector_mask_list = [None] * len(obs_list)

    assert detector_mask_list is not None

    for obs_idx, (obs, det_mask) in enumerate(zip(obs_list, detector_mask_list)):
        _get_detector_pairs(obs, detector_mask=det_mask, obs_idx=obs_idx)


def make_pair_differenced_map(
    nside: int,
    observations: Observation | list[Observation],
    pointings: np.ndarray | list[np.ndarray] | None = None,
    hwp: HWP | None = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    components: str | list[str] = "tod",
    detector_split: str = "full",
    time_split: str = "full",
    pointings_dtype=np.float64,
) -> PairDifferencingResult:
    """QU pair-differencing map-maker

    Map a list of observations

    Args:
        observations (list of :class:`Observations`): observations to be mapped.
            They are required to have the following attributes as arrays

            * `pointings`: the pointing information (in radians) for each tod
              sample. It must be a tensor with shape ``(N_d, N_t, 3)``,
              with ``N_d`` number of detectors and ``N_t`` number of
              samples in the TOD.
            * any attribute listed in `components` (by default, `tod`) and
              containing the TOD(s) to be differenced and binned together.

            The local detector set is required to be arranged in T/B pairs sharing
            the same wafer and focal-plane pixel. The map-maker differences the TOD
            of each pair and solves only for the Q/U Stokes components.

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
        pointings_dtype(dtype): data type for pointings generated on the fly. If
            the pointing is passed or already precomputed this parameter is
            ineffective. Default is `np.float64`.

    Returns:
        An instance of the class :class:`.PairDifferencingResult`. If the observations are
            distributed over MPI Processes, all of them get a copy of the same object.
    """

    if isinstance(components, str):
        components = [components]

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )
    assert ptg_list, "No observations provided"

    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)

    _check_tb_detector_pairs(obs_list, detector_mask_list)

    time_mask_list = _build_mask_time_split(time_split, obs_list)

    nobs_matrix, rhs = _build_nobs_matrix(
        nside=nside,
        obs_list=obs_list,
        ptg_list=ptg_list,
        hwp=hwp,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
        output_coordinate_system=output_coordinate_system,
        components=components,
        pointings_dtype=pointings_dtype,
    )

    _solve_binning(nobs_matrix, rhs)

    return PairDifferencingResult(
        binned_map=rhs.T,
        invnpp=nobs_matrix,
        coordinate_system=output_coordinate_system,
        components=components,
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
               samples in the TOD.
            * any attribute listed in `components` (by default, `tod`) and
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
