import numpy as np
import numpy.typing as npt
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from ducc0.healpix import Healpix_Base
from numba import njit

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
from ..maps_and_harmonics import HealpixMap, SphericalHarmonics
from ..observations import Observation
from ..pointings_in_obs import (
    _get_pointings_array,
)

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
    observation: Observation | list[Observation],
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
            If ``observations`` is not a list, ``pointings`` must be a
            np.array of shape ``(N_det, N_samples, 3)``. If ``observations``
            is a list, ``pointings`` must be a list of same length.

        hwp_angle (optional) : `2ωt`, hwp rotation angles
            (radians). If ``pointings`` is passed, ``hwp_angle``
            must be passed as well, otherwise both must be
            ``None``. If not passed, it is computed on the fly
            (generated by :func:`lbs.get_pointings` per detector).
            If ``observations`` is not a list, ``hwp_angle`` must
            be a np.array of dimensions (N_samples).

            If ``observations`` is a list, ``hwp_angle`` must be a
            list with the same length.

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

        nthreads : int, default=None
            Number of threads to use in the convolution. If None, the function
            reads from the `OMP_NUM_THREADS` environment variable.

        integrate_in_band : bool, default=False
            Whether to integrate the signal over the detector's frequency band.
            Only implemented for the Jones formalism.

        include_beam_throughput: bool, default=False
            Whether to include beam throughput in the bandpass in the bandpass profile.

    Raises:
        NotImplementedError : If `integrate_in_band` is True and the HWP calculus
            is set to Mueller.

    """

    # use getattr to avoid ty errors for dynamically
    # generated Observation class attributes
    g_one_over_k = getattr(observation, "g_one_over_k")
    amplitude_2f_k = getattr(observation, "amplitude_2f_k")
    pol_angle_rad = getattr(observation, "pol_angle_rad")
    pointing_theta_phi_psi_deg = getattr(observation, "pointing_theta_phi_psi_deg")
    bandcenter_ghz = getattr(observation, "bandcenter_ghz")
    bandwidth_ghz = getattr(observation, "bandwidth_ghz")

    assert hasattr(observation, "tod"), "Observation must have 'tod' initialized"

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

    for idet in range(observation.n_detectors):
        if pointings is None or hwp_angle is None:
            pointings, hwp_angle = observation.get_pointings(
                detector_idx=idet, pointings_dtype=pointings_dtype
            )
            pointings = pointings.reshape(-1, 3)

        # ----------------------------------------------------------
        # Get pointings in the correct coordinate system
        # ----------------------------------------------------------

        curr_pointings_det, hwp_angle = _get_pointings_array(
            detector_idx=idet,
            pointings=pointings,
            hwp_angle=hwp_angle,
            output_coordinate_system=CoordinateSystem.Galactic,
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

        pixmap = maps_det.values
        scheme = "NESTED" if maps_det.nest else "RING"
        hpx = Healpix_Base(maps_det.nside, scheme)

        pixel_ind_det = hpx.ang2pix(curr_pointings_det[:, 0:2])

        if pixmap.ndim == 2:
            # Shape: (nstokes, Npix)
            if maps_det.nstokes == 1:
                input_T = pixmap[0, pixel_ind_det]
                input_Q = np.zeros_like(input_T)
                input_U = input_Q
            else:
                input_T = pixmap[0, pixel_ind_det]
                input_Q = pixmap[1, pixel_ind_det]
                input_U = pixmap[2, pixel_ind_det]
        elif pixmap.ndim == 3:
            frequencies = np.array(maps_det.frequencies_ghz)

            # Find indices for the frequency band range for this detector
            indices = np.where(
                (frequencies >= bandcenter_ghz[idet] - bandwidth_ghz[idet] / 2)
                & (frequencies <= bandcenter_ghz[idet] + bandwidth_ghz[idet] / 2)
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
                                * np.exp(1j * np.deg2rad(cur_det_params.Phxx_0f[nu]))
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
                                * np.exp(1j * np.deg2rad(cur_det_params.Phyy_0f[nu]))
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
                                * np.exp(1j * np.deg2rad(cur_det_params.Phxx_2f[nu]))
                            ),
                            cur_det_params.Jxy_2f[nu]
                            * np.exp(1j * np.deg2rad(cur_det_params.Phxy_2f[nu])),
                        ],
                        [
                            cur_det_params.Jyx_2f[nu]
                            * np.exp(1j * np.deg2rad(cur_det_params.Phyx_2f[nu])),
                            (
                                cur_det_params.Jyy_2f[nu]
                                * np.exp(1j * np.deg2rad(cur_det_params.Phyy_2f[nu]))
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
                    m0f=observation.mueller_hwp[idet]["0f"],
                    m2f=observation.mueller_hwp[idet]["2f"],
                    m4f=observation.mueller_hwp[idet]["4f"],
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
                jones_0f = observation.jones_hwp[idet]["0f"]
                jones_2f = observation.jones_hwp[idet]["2f"]
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

    del pixel_ind_det
    del input_T, input_Q, input_U
    if not save_tod:
        del tod
