import logging

import numpy as np
import numpy.typing as npt
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from ducc0.healpix import Healpix_Base
from numba import njit

from litebird_sim.hwp_jones_parameters import HWPJonesParams

from ..bandpass_template_module import bandpass_profile
from ..coordinates import CoordinateSystem
from ..hwp_non_ideal import HWPFormalism, NonIdealHWP
from ..input_sky import SkyInput
from ..maps_and_harmonics import (
    HealpixMap,
    SphericalHarmonics,
    _interpolation_epsilon_for_tod,
    interpolate_alm,
)
from ..observations import Observation
from ..observation_utilities import (
    ObservationPointings,
    _check_detector_fields,
    observation_pointings,
)
from ..utilities import resolve_nthreads
from .jones_methods import (
    compute_signal_for_one_detector as compute_signal_for_one_detector_jones,
)
from .jones_methods import (
    integrate_inband_signal_for_one_detector as integrate_inband_signal_for_one_detector_jones,
)
from .mueller_methods import (
    compute_signal_for_one_detector as compute_signal_for_one_detector_mueller,
)

log = logging.getLogger(__name__)

k_B = getattr(const, "k_B").value  # Boltzmann constant in J/K
c = getattr(const, "c").value  # Speed of light in m/s
h = getattr(const, "h").value  # Planck constant in J s
Tcmb0 = getattr(cosmo, "Tcmb0").value  # CMB temperature today in K


def _dBodTrj(nu: npt.NDArray[np.float64] | np.float64):
    return 2 * k_B * nu * nu * 1e18 / c / c


def _dBodTth(nu: npt.NDArray[np.float64] | np.float64):
    x = h * nu * 1e9 / k_B / Tcmb0
    ex = np.exp(x)
    exm1 = ex - 1.0e0
    return 2 * h * nu * nu * nu * 1e27 / c / c / exm1 / exm1 * ex * x / Tcmb0


@njit
def compute_orientation_from_detquat(quat: npt.NDArray[np.float64]) -> float:
    if quat[2] == 0:
        polang = 0.0
    else:
        polang = 2 * np.arctan2(
            np.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2), quat[3]
        )
        if quat[2] < 0:
            polang = -polang

    return polang


def set_band_params_for_one_detector(
    hwp: NonIdealHWP,
    band_params: HWPJonesParams,
    bandcenter_ghz: float,
    bandwidth_ghz: float,
    bandpass: dict[str, object] | None,
    include_beam_throughput: bool,
) -> tuple[HWPJonesParams, npt.NDArray[np.float64]]:
    """
    Load frequency-dependent parameters and normalized bandpass profiles
    for a single detector based on HWP Jones coefficients.

    This function retrieves the necessary HWP coefficients, filters them by
    the detector's frequency range, and calculates the integrated bandpass
    profile (bpi) weighted by the derivative of the Planck function.

    Args:
        hwp: The HWP object
        bandcenter_ghz: Central frequency of the detector band.
        bandwidth_ghz: Width of the detector band.
        bandpass: Dictionary containing bandpass profile metadata.
        include_beam_throughput: Whether to include beam effects in the profile.

    Returns:
        A tuple containing:
            - An instance of :class:`HWPJonesParams`.
            - A numpy array representing the normalized bandpass profile (bpi).

    Raises:
        NotImplementedError: If the HWP calculus is set to Mueller instead of Jones.
    """

    if hwp.calculus != HWPFormalism.JONES:
        raise NotImplementedError(
            "Band integration is currently only implemented for the Jones formalism."
        )

    # Create a copy of `band_params` that only refers to the frequencies
    # of this detector
    det_params = band_params.clip_frequencies(
        bandcenter_ghz=bandcenter_ghz, bandwidth_ghz=bandwidth_ghz
    )

    #
    # For now, if there is no bandpass, this function will return
    # a simple array of ones, because unit conversions are being done
    # at the input map level. This is left in here as a placeholder
    # for future developments, in which the term to multiply to
    # the bandpass can be further developed.
    #
    bpi = np.ones_like(det_params.freq_ghz)
    if not bandpass:
        return (det_params, bpi)
    else:
        band_params.freq_ghz, bandpass_prof = bandpass_profile(
            band_params.freq_ghz, bandpass, include_beam_throughput
        )
        bpi *= bandpass_prof
        return (det_params, bpi)


def _harmonics_fields(hwp: NonIdealHWP) -> list[str]:
    """The fields of the detectors read by :func:`.fill_tod_with_hwp_harmonics`"""
    return [
        "g_one_over_k",
        "amplitude_2f_k",
        "pol_angle_rad",
        "pointing_theta_phi_psi_deg",
        "bandcenter_ghz",
        "bandwidth_ghz",
        "mueller_hwp" if hwp.calculus is HWPFormalism.MUELLER else "jones_hwp",
    ]


def fill_tod_with_hwp_harmonics(
    hwp: NonIdealHWP,
    observation: Observation,
    tod: np.ndarray,
    maps: SkyInput | None = None,
    pointings: np.ndarray | list[np.ndarray] | ObservationPointings | None = None,
    hwp_angle: np.ndarray | None = None,
    input_names: list[str] | None = None,
    pointings_dtype=np.float64,
    apply_non_linearity: bool = False,
    add_2f_hwpss: bool = False,
    mueller_phases: dict[str, np.ndarray] | None = None,
    integrate_in_band: bool = False,
    nthreads: int | None = None,
    include_beam_throughput: bool = False,
):
    r"""Fill a TOD for one observation, using HWP rotation speed
    harmonics calculus.

    Args:
        hwp : HWP, optional
            Half-wave plate (HWP) model. If None, HWP effects are ignored unless
            the `Observation` object contains HWP data.

        observation (:class:`Observation`) : container for the
            TOD,detectors and pointings. If the TOD is not required,
            you can avoid allocating ``observations.tod`` by setting
            ``allocate_tod=False`` in :class:`.Observation`.

        tod : np.ndarray
            Time-ordered data (TOD) array of shape (n_detectors, n_samples) that will be
            filled with the simulated sky signal.

        maps : HealpixMap | SphericalHarmonics | dict[str, HealpixMap] |
            dict[str, SphericalHarmonics]
            Sky model to be scanned. In dictionary form, the keys must
            match the entries of `input_names`.

        pointings (optional) : if not present, it is either computed
            on the fly (generated by :func:`lbs.get_pointings` per
            detector), or read from ``observations.pointing_matrix``
            (if present).
            ``pointings`` must be a np.array of shape ``(N_det, N_samples, 3)``
            in Ecliptic coordinates, or the :class:`.ObservationPointings`
            of `observation`.

        hwp_angle (optional) : `2ωt`, hwp rotation angles
            (radians). If ``pointings`` is an array, ``hwp_angle``
            must be passed as well, otherwise both must be
            ``None``. If not passed, it is taken from the
            :class:`.ObservationPointings` (the angle of `hwp`).
            ``hwp_angle`` must be a np.array of dimensions (N_samples).

        input_names : array-like of str or None, default=None
            Per-detector keys to select entries in `maps` when `maps` is a dictionary.

        pointings_dtype : if ``pointings`` is None and is computed
            within ``fill_tod``, this is the dtype for
            pointings and tod (default: np.float32).

        apply_non_linearity (bool) : applies the coupling of the non-linearity
            systematics with hwp_sys

        add_2f_hwpss (bool) : adds the 2f hwpss signal to the TOD

        mueller_phases (dict) : the non-ideal phases of the 2f and 4f
            Mueller matrices, as ``{"2f": ..., "4f": ...}``, the same for all
            the detectors. Required with the Mueller formalism.

        integrate_in_band : bool, default=False
            Whether to integrate the signal over the detector's frequency band.
            Only implemented for the Jones formalism.

        nthreads (int) : number of threads to use in ducc's Healpix methods.
            If None, the function reads from the `OMP_NUM_THREADS` environment variable.

        include_beam_throughput: bool, default=False
            Whether to include beam throughput in the bandpass in the bandpass profile.

    Raises:
        NotImplementedError : If `integrate_in_band` is True and the HWP calculus
            is set to Mueller.

    """
    if maps is None:
        if observation is None:
            raise ValueError(
                "'maps' is None and 'observation' is None. "
                "You should either pass the maps here, or pass an observation."
            )
        elif isinstance(observation, list):
            maps = observation[0].sky
        elif isinstance(observation, Observation):
            maps = observation.sky
        else:
            raise ValueError(
                f"'maps' is None and type 'observation' is {type(observation)}. "
                "You should either pass the maps here, or pass an observation, or a list of observations that contain the maps."
            )
    assert maps is not None, "You need to pass input maps to fill_tod."

    nthreads = resolve_nthreads(nthreads)

    if isinstance(pointings, ObservationPointings):
        assert isinstance(observation, Observation) and (
            pointings.observation is observation
        ), "The pointings must be those of the observation"
        obs_pointings = [(observation, pointings)]
        hwp_angle_list = [hwp_angle if hwp_angle is not None else pointings.hwp_angle]
    elif pointings is None:
        if hwp_angle is not None:
            raise Warning(
                "You passed hwp_angle, but you did not pass pointings, "
                + "so hwp_angle will be ignored and re-computed on the fly."
            )
        obs_pointings = observation_pointings(
            observation,
            hwp=hwp,
            pointings_dtype=pointings_dtype,
            nthreads=nthreads,
        )
        hwp_angle_list = [cur_ptg.hwp_angle for _, cur_ptg in obs_pointings]
    else:
        if callable(pointings):
            pointings, hwp_angle = pointings("all", pointings_dtype=pointings_dtype)
        if hwp_angle is None:
            raise ValueError("If you pass pointings, you must also pass hwp_angle.")
        if isinstance(observation, Observation):
            assert isinstance(hwp_angle, np.ndarray), (
                "For one observation, hwp_angle must be passed "
                + "as a np.array to fill_tod"
            )
            assert observation.n_samples_process == hwp_angle.shape[0], (
                "You need to pass a hwp_angle np.array with shape"
                + "N_samples for the observation"
            )
            hwp_angle_list = [hwp_angle]
        else:
            assert len(observation) == len(hwp_angle), (
                f"The list of observations has {len(observation)} elements, but "
                + f"the list of hwp_angle has {len(hwp_angle)} elements"
            )
            hwp_angle_list = list(hwp_angle)
        # The shapes of the pointings are checked here
        obs_pointings = observation_pointings(
            observation,
            pointings=pointings,
            hwp=hwp,
            pointings_dtype=pointings_dtype,
            nthreads=nthreads,
        )

    _check_detector_fields(
        [cur_obs for cur_obs, _ in obs_pointings], _harmonics_fields(hwp)
    )

    for idx_obs, (cur_obs, cur_ptg) in enumerate(obs_pointings):
        # use getattr to avoid ty errors for dynamically
        # generated cur_obs class attributes
        g_one_over_k = getattr(cur_obs, "g_one_over_k")
        amplitude_2f_k = getattr(cur_obs, "amplitude_2f_k")
        pol_angle_rad = getattr(cur_obs, "pol_angle_rad")
        pointing_theta_phi_psi_deg = getattr(cur_obs, "pointing_theta_phi_psi_deg")
        bandcenter_ghz = getattr(cur_obs, "bandcenter_ghz")
        bandwidth_ghz = getattr(cur_obs, "bandwidth_ghz")

        assert hasattr(cur_obs, "tod"), "cur_obs must have 'tod' initialized"

        if integrate_in_band:
            if hwp.jones_parameters is None:
                raise AssertionError(
                    "integrate_in_band set to True but no csv file containing the jones parameters \
                    per frequency given to the HWP object."
                )
            else:
                band_params = hwp.jones_parameters

        for idet in range(cur_obs.n_detectors):
            # ----------------------------------------------------------
            # Select per-detector sky object
            # ----------------------------------------------------------
            if isinstance(maps, dict):
                if input_names is None:
                    raise ValueError(
                        "scan_map: maps is a dict, but input_names is None. "
                        "You must provide per-detector keys."
                    )
                key = input_names[idet]
                try:
                    maps_det = maps[key]
                except KeyError as exc:
                    raise KeyError(
                        f"scan_map: maps does not contain an entry for '{key}' "
                        f"(detector index {idet})"
                    ) from exc
            else:
                maps_det = maps

            # ----------------------------------------------------------
            # Determine coordinate system from the object
            # ----------------------------------------------------------
            coordinates = getattr(maps_det, "coordinates", None)
            if coordinates is None:
                log.warning(
                    "scan_map: maps_det.coordinates is None — assuming "
                    "CoordinateSystem.Galactic"
                )
                coordinates = CoordinateSystem.Galactic

            # Pointings in the coordinate system of the map
            curr_pointings_det = cur_ptg.detector(
                idet, output_coordinate_system=coordinates
            )
            hwp_angle = hwp_angle_list[idx_obs]

            tod_det = tod[idet, :]

            xi = pol_angle_rad[idet]
            psi = curr_pointings_det[:, 2]

            phi = np.deg2rad(pointing_theta_phi_psi_deg[idet][1])

            cos2Xi2Phi = np.cos(2 * xi - 2 * phi)
            sin2Xi2Phi = np.sin(2 * xi - 2 * phi)

            # ----------------------------------------------------------
            # REAL SPACE (HealpixMap)
            # ----------------------------------------------------------
            if isinstance(maps_det, HealpixMap):
                pixmap = maps_det.values  # (nstokes, Npix)
                scheme = "NESTED" if maps_det.nest else "RING"
                hpx = Healpix_Base(maps_det.nside, scheme)

                pixel_ind_det = hpx.ang2pix(
                    curr_pointings_det[:, 0:2], nthreads=nthreads
                )

                if pixmap.ndim == 2:
                    # Shape: (nstokes, Npix)
                    if maps_det.nstokes == 1:
                        input_T = pixmap[0, pixel_ind_det]
                        input_Q = np.zeros_like(input_T)
                        input_U = np.zeros_like(input_T)
                    else:
                        input_T = pixmap[0, pixel_ind_det]
                        input_Q = pixmap[1, pixel_ind_det]
                        input_U = pixmap[2, pixel_ind_det]

                elif pixmap.ndim == 3:
                    frequencies = np.array(maps_det.frequencies_ghz)

                    # Find indices for the frequency band range for this detector
                    indices = np.where(
                        (frequencies >= bandcenter_ghz[idet] - bandwidth_ghz[idet] / 2)
                        & (
                            frequencies
                            <= bandcenter_ghz[idet] + bandwidth_ghz[idet] / 2
                        )
                    )[0]
                    start_index = indices[0]
                    end_index = indices[-1]

                    # Shape: (N, nstokes, Npix)
                    if maps_det.nstokes == 1:
                        input_T = pixmap[start_index:end_index, 0, pixel_ind_det]
                        input_Q = np.zeros_like(input_T)
                        input_U = input_Q
                    else:
                        input_T = pixmap[start_index : end_index + 1, 0, pixel_ind_det]
                        input_Q = pixmap[start_index : end_index + 1, 1, pixel_ind_det]
                        input_U = pixmap[start_index : end_index + 1, 2, pixel_ind_det]

                del pixel_ind_det

            # ----------------------------------------------------------
            # HARMONIC SPACE (SphericalHarmonics)
            # ----------------------------------------------------------
            elif isinstance(maps_det, SphericalHarmonics):
                interp = interpolate_alm(
                    alms=maps_det,
                    locations=curr_pointings_det[:, 0:2],
                    epsilon=_interpolation_epsilon_for_tod(maps_det, tod.dtype),
                    nthreads=nthreads,
                )

                if maps_det.nstokes == 1:
                    input_T = interp
                    input_Q = np.zeros_like(input_T)
                    input_U = np.zeros_like(input_T)
                else:
                    input_T, input_Q, input_U = interp
            else:
                raise TypeError(
                    "scan_map: maps_det must be HealpixMap or SphericalHarmonics "
                    f"(got {type(maps_det)!r})"
                )
            if integrate_in_band:
                if hwp.calculus is HWPFormalism.MUELLER:
                    raise NotImplementedError(
                        "Band integration is only implemented for Jones Formalism"
                    )

                cur_det_params, cur_det_bpi = set_band_params_for_one_detector(
                    hwp=hwp,
                    band_params=band_params,
                    bandcenter_ghz=bandcenter_ghz[idet],
                    bandwidth_ghz=bandwidth_ghz[idet],
                    bandpass=None,
                    include_beam_throughput=include_beam_throughput,
                )

                deltas_j0f = np.zeros(
                    (len(cur_det_params.freq_ghz), 2, 2), dtype=np.complex128
                )
                deltas_j2f = np.zeros(
                    (len(cur_det_params.freq_ghz), 2, 2), dtype=np.complex128
                )

                for nu in range(len(cur_det_params.freq_ghz)):
                    deltas_j0f[nu] = np.array(
                        [
                            [
                                (
                                    cur_det_params.Jxx_0f[nu]
                                    * np.exp(
                                        1j * np.deg2rad(cur_det_params.Phxx_0f[nu])
                                    )
                                )
                                - 1,
                                cur_det_params.Jxy_0f[nu]
                                * np.exp(1j * np.deg2rad(cur_det_params.Phxy_0f[nu])),
                            ],
                            [
                                cur_det_params.Jyx_0f[nu]
                                * np.exp(1j * np.deg2rad(cur_det_params.Phyx_0f[nu])),
                                (
                                    cur_det_params.Jyy_0f[nu]
                                    * np.exp(
                                        1j * np.deg2rad(cur_det_params.Phyy_0f[nu])
                                    )
                                )
                                + 1,
                            ],
                        ],
                        dtype=np.complex128,
                    )
                    deltas_j2f[nu] = np.array(
                        [
                            [
                                (
                                    cur_det_params.Jxx_2f[nu]
                                    * np.exp(
                                        1j * np.deg2rad(cur_det_params.Phxx_2f[nu])
                                    )
                                ),
                                cur_det_params.Jxy_2f[nu]
                                * np.exp(1j * np.deg2rad(cur_det_params.Phxy_2f[nu])),
                            ],
                            [
                                cur_det_params.Jyx_2f[nu]
                                * np.exp(1j * np.deg2rad(cur_det_params.Phyx_2f[nu])),
                                (
                                    cur_det_params.Jyy_2f[nu]
                                    * np.exp(
                                        1j * np.deg2rad(cur_det_params.Phyy_2f[nu])
                                    )
                                ),
                            ],
                        ],
                        dtype=np.complex128,
                    )

                integrate_inband_signal_for_one_detector_jones(
                    tod_det=tod_det,
                    freqs=cur_det_params.freq_ghz,
                    band=cur_det_bpi,
                    deltas_j0f=deltas_j0f,
                    deltas_j2f=deltas_j2f,
                    mapT=input_T,
                    mapQ=input_Q,
                    mapU=input_U,
                    rho=np.array(hwp_angle, dtype=np.float64),
                    psi=np.array(psi, dtype=np.float64),
                    phi=phi,
                    xi=xi,
                    cos2Xi2Phi=cos2Xi2Phi,
                    sin2Xi2Phi=sin2Xi2Phi,
                    apply_non_linearity=apply_non_linearity,
                    g_one_over_k=g_one_over_k[idet],
                    add_2f_hwpss=add_2f_hwpss,
                    amplitude_2f_k=amplitude_2f_k[idet],
                )

                # normalize to the bandwidth after integration
                if len(cur_det_params.freq_ghz) > 1:
                    tod_det /= bandwidth_ghz[idet] * 1e9

            else:
                if hwp.calculus is HWPFormalism.MUELLER:
                    if mueller_phases is None:
                        raise AssertionError(
                            "HWP Formalism set to Mueller but no mueller_phases given."
                        )
                    compute_signal_for_one_detector_mueller(
                        tod_det=tod_det,
                        m0f=cur_obs.mueller_hwp[idet]["0f"],
                        m2f=cur_obs.mueller_hwp[idet]["2f"],
                        m4f=cur_obs.mueller_hwp[idet]["4f"],
                        rho=np.array(hwp_angle, dtype=np.float64),
                        psi=np.array(psi, dtype=np.float64),
                        mapT=input_T,
                        mapQ=input_Q,
                        mapU=input_U,
                        cos2Xi2Phi=cos2Xi2Phi,
                        sin2Xi2Phi=sin2Xi2Phi,
                        phi=phi,
                        xi=xi,
                        apply_non_linearity=apply_non_linearity,
                        g_one_over_k=g_one_over_k[idet],
                        add_2f_hwpss=add_2f_hwpss,
                        amplitude_2f_k=amplitude_2f_k[idet],
                        phases_2f=mueller_phases["2f"],
                        phases_4f=mueller_phases["4f"],
                    )

                elif hwp.calculus is HWPFormalism.JONES:
                    jones_0f = cur_obs.jones_hwp[idet]["0f"]
                    jones_2f = cur_obs.jones_hwp[idet]["2f"]
                    deltas_j0f = jones_0f.copy()
                    deltas_j2f = jones_2f.copy()
                    deltas_j0f[0, 0] = jones_0f[0, 0] - 1
                    deltas_j0f[1, 1] = jones_0f[1, 1] + 1
                    deltas_j2f[0, 0] = jones_2f[0, 0]
                    deltas_j2f[1, 1] = jones_2f[1, 1]

                    compute_signal_for_one_detector_jones(
                        tod_det=tod_det,
                        deltas_j0f=deltas_j0f,
                        deltas_j2f=deltas_j2f,
                        rho=np.array(hwp_angle, dtype=np.float64),
                        psi=np.array(psi, dtype=np.float64),
                        mapT=input_T,
                        mapQ=input_Q,
                        mapU=input_U,
                        cos2Xi2Phi=cos2Xi2Phi,
                        sin2Xi2Phi=sin2Xi2Phi,
                        phi=phi,
                        xi=xi,
                        apply_non_linearity=apply_non_linearity,
                        g_one_over_k=g_one_over_k[idet],
                        add_2f_hwpss=add_2f_hwpss,
                        amplitude_2f_k=amplitude_2f_k[idet],
                    )

        del input_T, input_Q, input_U
