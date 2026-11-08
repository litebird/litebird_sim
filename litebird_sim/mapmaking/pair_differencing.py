# The implementation of the binning algorithm provided here is derived
# from the more general destriping equation presented in the paper
# «Destriping CMB temperature and polarization maps» by Kurki-Suonio et al. 2009,
# A&A 506, 1511–1539 (2009), https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code, as many
# functions and variable defined here use the same letters and symbols of that
# paper. We refer to it in code comments and docstrings as "KurkiSuonio2009".

from collections.abc import Sequence
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
    _is_well_conditioned,
    pointing_row,
    _apply_flags_to_time_masks,
    get_map_making_weights,
    get_pol_efficiency,
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


@njit(parallel=True)
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

    for ipix in prange(npix):  # type: ignore[not-iterable]
        m = nobs_matrix[ipix]
        if _is_well_conditioned(m):
            # Analytic inverse of the symmetric 2×2 matrix
            inv_det = 1.0 / (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
            m00 = m[1, 1] * inv_det
            m11 = m[0, 0] * inv_det
            m01 = -m[0, 1] * inv_det
            b0, b1 = atd[ipix, 0], atd[ipix, 1]
            atd[ipix, 0] = m00 * b0 + m01 * b1
            atd[ipix, 1] = m01 * b0 + m11 * b1
            m[0, 0], m[0, 1], m[1, 0], m[1, 1] = m00, m01, m01, m11
        else:
            m.fill(hp.UNSEEN)
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
    pol_eff_t: float,
    pol_eff_b: float,
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
    #   d_T - d_B = Q·(γ_T·cos(2ψ_T) - γ_B·cos(2ψ_B)) + U·(γ_T·sin(2ψ_T) - γ_B·sin(2ψ_B))
    #
    # where γ is the polarization efficiency of each detector.
    # Detector weights are sample variances (sampling_rate_hz * net_ukrts**2).
    # For independent detector noise, Var(d_T - d_B) = weight_t + weight_b.
    # ψ is the proper sum of the detector polarization angle and the HWP angle (if present)

    assert (
        tod_t.shape
        == tod_b.shape
        == pix_t.shape
        == psi_t.shape
        == psi_b.shape
        == t_mask.shape
    )

    pair_weight = weight_t + weight_b
    inv_sigma = 1.0 / np.sqrt(pair_weight)
    inv_sigma2 = inv_sigma * inv_sigma

    # One pass for the matrix and the RHS, so that cos/sin are computed once
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
            # The difference of the two rows: the I terms cancel
            _, q_t, u_t = pointing_row(cur_psi_t, pol_eff_t)
            _, q_b, u_b = pointing_row(cur_psi_b, pol_eff_b)
            pair_cos = q_t - q_b
            pair_sin = u_t - u_b

            if not additional_component:
                info_pix = nobs_matrix[cur_pix_idx]
                info_pix[0, 0] += pair_cos * pair_cos * inv_sigma2
                info_pix[0, 1] += pair_cos * pair_sin * inv_sigma2
                info_pix[1, 0] += pair_cos * pair_sin * inv_sigma2
                info_pix[1, 1] += pair_sin * pair_sin * inv_sigma2

            rhs_pix = rhs[cur_pix_idx]
            rhs_pix[0] += pair_sample * pair_cos * inv_sigma2
            rhs_pix[1] += pair_sample * pair_sin * inv_sigma2


def _build_nobs_matrix(
    nside: int,
    obs_pointings: list[tuple[Observation, ObservationPointings]],
    dm_list: list[npt.NDArray],
    tm_list: list[npt.NDArray],
    components: list[str],
) -> tuple[npt.NDArray, npt.NDArray]:
    n_pix = HealpixMap.nside_to_npix(nside)

    nobs_matrix = np.zeros((n_pix, 2, 2))
    rhs = np.zeros((n_pix, 2))

    for obs_idx, ((cur_obs, cur_ptg), cur_d_mask, cur_t_mask) in enumerate(
        zip(obs_pointings, dm_list, tm_list)
    ):
        cur_weights = get_map_making_weights(cur_obs, check=True)
        cur_pol_eff = get_pol_efficiency(cur_obs)

        # In the coordinate system of the map
        pixidx_all, polang_all = cur_ptg.pixel_index_and_pol_angle("all", nside)

        detector_pairs = _get_detector_pairs(
            cur_obs, detector_mask=cur_d_mask, obs_idx=obs_idx
        )

        # A pair difference uses the samples good for both detectors
        pair_masks = {
            (t_idx, b_idx): np.ascontiguousarray(cur_t_mask[t_idx] & cur_t_mask[b_idx])
            for t_idx, b_idx in detector_pairs
        }

        for t_idx, b_idx in detector_pairs:
            pair_mask = pair_masks[t_idx, b_idx]
            if not np.array_equal(
                pixidx_all[t_idx, pair_mask], pixidx_all[b_idx, pair_mask]
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
                    cur_pol_eff[t_idx],
                    cur_pol_eff[b_idx],
                    pair_masks[t_idx, b_idx],
                    nobs_matrix,
                    rhs,
                    additional_component=idx > 0,
                )

        del pixidx_all, polang_all

    assert obs_pointings, "No observations provided"
    reduction_for_observations([cur_obs for cur_obs, _ in obs_pointings]).sum_in_place(
        nobs_matrix, rhs
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
    detector_mask_list: Sequence[npt.NDArray | None] | None = None,
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
    nthreads: int | None = None,
    flags: str | list[str] | None = None,
    flag_mask: int | None = None,
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
        nthreads : int, default=None
            Number of threads to use in ducc's Healpix methods. If None,
            the function reads from the `OMP_NUM_THREADS` environment variable.
        flags (str or list[str]): names of the :class:`.Observation`
            attributes holding the flags, either per detector (e.g.
            ``"local_flags"``) or common to all the detectors (e.g.
            ``"global_flags"``). A sample enters the difference of a pair only
            if it is not flagged in either detector.
        flag_mask (int): bits of the flags that exclude a sample (e.g.
            ``0b101``). If None, any nonzero flag excludes the sample.

    Returns:
        An instance of the class :class:`.PairDifferencingResult`. If the observations are
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

    _check_tb_detector_pairs(obs_list, detector_mask_list)

    time_mask_list = _apply_flags_to_time_masks(
        _build_mask_time_split(time_split, obs_list), obs_list, flags, flag_mask
    )

    nobs_matrix, rhs = _build_nobs_matrix(
        nside=nside,
        obs_pointings=obs_pointings,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
        components=components,
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
