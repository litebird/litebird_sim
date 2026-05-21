import numpy as np
import numpy.typing as npt
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from ducc0.healpix import Healpix_Base
from numba import njit
import os
import logging

from litebird_sim.hwp_jones_parameters import HWPJonesParams
from .jones_methods import (
    compute_signal_for_one_detector as compute_signal_for_one_detector_jones,
    integrate_inband_signal_for_one_detector as integrate_inband_signal_for_one_detector_jones,
)
from .mueller_methods import (
    compute_signal_for_one_detector as compute_signal_for_one_detector_mueller,
)
from ..bandpass_template_module import bandpass_profile
from ..coordinates import CoordinateSystem
from ..hwp_non_ideal import HWPFormalism, NonIdealHWP
from ..input_sky import SkyGenerationParams
from ..maps_and_harmonics import HealpixMap, SphericalHarmonics, interpolate_alm
from ..observations import Observation
from ..pointings_in_obs import (
    _get_pointings_array,
)
from ..constants import NUM_THREADS_ENVVAR

COND_THRESHOLD = 1e10

k_B = getattr(const, "k_B").value  # Boltzmann constant in J/K
c = getattr(const, "c").value  # Speed of light in m/s
h = getattr(const, "h").value  # Planck constant in J s
Tcmb0 = getattr(cosmo, "Tcmb0").value  # CMB temperature today in K


def _dBodTrj(nu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 2 * k_B * nu * nu * 1e18 / c / c


def _dBodTth(nu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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

    # bpi : bandpass profile * intensity conversion
    # if no bandpass, only apply the frequency dependent
    # T to I conversion
    if not bandpass:
        bpi = _dBodTth(det_params.freq_ghz)
    else:
        band_params.freq_ghz, bandpass_prof = bandpass_profile(
            band_params.freq_ghz, bandpass, include_beam_throughput
        )
        bpi = _dBodTth(det_params.freq_ghz) * bandpass_prof

    # Normalize the band
    bpi /= np.trapz(bpi, det_params.freq_ghz)

    return (det_params, bpi)


def fill_tod_with_hwp_harmonics(
    hwp: NonIdealHWP,
    observation: Observation,
    tod: np.ndarray,
    maps: (
        HealpixMap
        | dict[str, HealpixMap | SkyGenerationParams]
        | SphericalHarmonics
        | dict[str, SphericalHarmonics | SkyGenerationParams]
        | None
    ) = None,
    pointings: np.ndarray | None = None,
    hwp_angle: np.ndarray | None = None,
    save_tod: bool = True,
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
            ``pointings`` must be a np.array of shape ``(N_det, N_samples, 3)``.

        hwp_angle (optional) : `2ωt`, hwp rotation angles
            (radians). If ``pointings`` is passed, ``hwp_angle``
            must be passed as well, otherwise both must be
            ``None``. If not passed, it is computed on the fly
            (generated by :func:`lbs.get_pointings` per detector).
            ``hwp_angle`` must be a np.array of dimensions (N_samples).

        save_tod (bool) : if True, ``obs.tod`` is saved in
            ``observations.tod`` and locally as a .npy file;
            if False, ``obs.tod`` gets deleted.

        input_names : array-like of str or None, default=None
            Per-detector keys to select entries in `maps` when `maps` is a dictionary.

        pointings_dtype : if ``pointings`` is None and is computed
            within ``fill_tod``, this is the dtype for
            pointings and tod (default: np.float32).

        apply_non_linearity (bool) : applies the coupling of the non-linearity
            systematics with hwp_sys

        add_2f_hwpss (bool) : adds the 2f hwpss signal to the TOD

        mueller_phases (dict) : the non ideal phases for the mueller
            matrix elements. When None is given, the temporary values from
            Patanchon et al. [2021] are used.

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

    # Set number of threads
    if nthreads is None:
        nthreads = int(os.environ.get(NUM_THREADS_ENVVAR, 0))

    if pointings is None:
        if hwp_angle is not None:
            raise Warning(
                "You passed hwp_angle, but you did not pass pointings, "
                + "so hwp_angle will be ignored and re-computed on the fly."
            )

        if isinstance(observation, Observation):
            obs_list = [observation]
            if hasattr(observation, "pointing_matrix"):
                ptg_list = [observation.pointing_matrix]
            else:
                ptg_list = []
            if hasattr(observation, "hwp_angle"):
                hwp_angle_list = [observation.hwp_angle]
            else:
                hwp_angle_list = []

        else:
            obs_list = observation
            ptg_list = []
            hwp_angle_list = []
            for ob in observation:
                if hasattr(ob, "pointing_matrix"):
                    ptg_list.append(ob.pointing_matrix)
                if hasattr(ob, "hwp_angle"):
                    hwp_angle_list.append(ob.hwp_angle)

    else:
        if callable(pointings):
            pointings, hwp_angle = pointings("all", pointings_dtype=pointings_dtype)
        if isinstance(observation, Observation):
            assert isinstance(pointings, np.ndarray), (
                "For one observation you need to pass a np.array "
                + "of pointings to fill_tod"
            )
            assert (
                observation.n_detectors == pointings.shape[0]
                and observation.n_samples == pointings.shape[1]
                and pointings.shape[2] == 3
            ), (
                "You need to pass a pointing np.array with shape"
                + "(N_det, N_samples, 3) for the observation"
            )
            obs_list = [observation]
            ptg_list = [pointings]
            if hwp_angle is not None:
                assert isinstance(hwp_angle, np.ndarray), (
                    "For one observation, hwp_angle must be passed "
                    + "as a np.array to fill_tod"
                )
                assert observation.n_samples == hwp_angle.shape[0], (
                    "You need to pass a hwp_angle np.array with shape"
                    + "N_samples for the observation"
                )
                hwp_angle_list = [hwp_angle]
            else:
                raise ValueError("If you pass pointings, you must also pass hwp_angle.")
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to fill_tod, "
                + "you must a list of `pointings`"
            )
            assert len(observation) == len(pointings), (
                f"The list of observations has {len(observation)} elements, but "
                + f"the list of pointings has {len(pointings)} elements"
            )
            obs_list = observation
            ptg_list = pointings
            if hwp_angle is not None:
                assert len(observation) == len(hwp_angle), (
                    f"The list of observations has {len(observation)} elements, but "
                    + f"the list of hwp_angle has {len(hwp_angle)} elements"
                )
                hwp_angle_list = hwp_angle
            else:
                raise ValueError("If you pass pointings, you must also pass hwp_angle.")

    for idx_obs, cur_obs in enumerate(obs_list):
        if not hasattr(cur_obs, "pointing_matrix"):
            cur_obs.pointing_matrix = np.empty(
                (cur_obs.n_detectors, cur_obs.n_samples, 3),
                dtype=pointings_dtype,
            )

        # use getattr to avoid ty errors for dynamically
        # generated cur_obs class attributes
        g_one_over_k = getattr(cur_obs, "g_one_over_k")
        amplitude_2f_k = getattr(cur_obs, "amplitude_2f_k")
        pol_angle_rad = getattr(cur_obs, "pol_angle_rad")
        pointing_theta_phi_psi_deg = getattr(cur_obs, "pointing_theta_phi_psi_deg")
        bandcenter_ghz = getattr(cur_obs, "bandcenter_ghz")
        bandwidth_ghz = getattr(cur_obs, "bandwidth_ghz")

        assert hasattr(cur_obs, "tod"), "cur_obs must have 'tod' initialized"

        if type(pointings) is np.ndarray:
            assert tod.shape == pointings.shape[0:2]

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

            if pointings is None and ((not ptg_list) or (not hwp_angle_list)):
                cur_point, cur_hwp_angle = cur_obs.get_pointings(
                    detector_idx=idet, pointings_dtype=pointings_dtype
                )
                cur_point = cur_point.reshape(-1, 3)
            else:
                cur_point = ptg_list[idx_obs][:, :, :]
                cur_hwp_angle = hwp_angle_list[idx_obs]

            # ----------------------------------------------------------
            # Determine coordinate system from the object
            # ----------------------------------------------------------
            coordinates = getattr(maps_det, "coordinates", None)
            if coordinates is None:
                logging.warning(
                    "scan_map: maps_det.coordinates is None — assuming "
                    "CoordinateSystem.Galactic"
                )
                coordinates = CoordinateSystem.Galactic

            # ----------------------------------------------------------
            # Get pointings in the correct coordinate system
            # ----------------------------------------------------------
            curr_pointings_det, hwp_angle = _get_pointings_array(
                detector_idx=idet,
                pointings=cur_point,
                hwp_angle=cur_hwp_angle,
                output_coordinate_system=coordinates,
                pointings_dtype=pointings_dtype,
            )

            tod_det = tod[idet, :]

            xi = pol_angle_rad[idet]
            psi = curr_pointings_det[:, 2]

            phi = np.deg2rad(pointing_theta_phi_psi_deg[idet][1])

            cos2Xi2Phi = np.cos(2 * xi - 2 * phi)
            sin2Xi2Phi = np.sin(2 * xi - 2 * phi)

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
            # REAL SPACE (HealpixMap)
            # ----------------------------------------------------------
            if isinstance(maps_det, HealpixMap):
                pixmap = maps_det.values  # (nstokes, Npix)
                scheme = "NESTED" if maps_det.nest else "RING"
                hpx = Healpix_Base(maps_det.nside, scheme)

                pixel_ind_det = hpx.ang2pix(
                    curr_pointings_det[:, 0:2], nthreads=nthreads
                )

                if maps_det.nstokes == 1:
                    input_T = pixmap[0, pixel_ind_det]
                    input_Q = np.zeros_like(input_T)
                    input_U = np.zeros_like(input_T)
                else:
                    input_T = pixmap[0, pixel_ind_det]
                    input_Q = pixmap[1, pixel_ind_det]
                    input_U = pixmap[2, pixel_ind_det]

                del pixel_ind_det

            # ----------------------------------------------------------
            # HARMONIC SPACE (SphericalHarmonics)
            # ----------------------------------------------------------
            elif isinstance(maps_det, SphericalHarmonics):
                interp = interpolate_alm(
                    alms=maps_det,
                    locations=curr_pointings_det[:, 0:2],
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
        if not save_tod:
            del tod
