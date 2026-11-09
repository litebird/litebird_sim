import logging

from collections.abc import Sequence

import numpy as np
from ducc0.healpix import Healpix_Base
from numba import njit, prange
from numpy.typing import DTypeLike

from .coordinates import CoordinateSystem
from .hwp import HWP, IdealHWP
from .hwp_harmonics.hwp_harmonics import _harmonics_fields, fill_tod_with_hwp_harmonics
from .hwp_non_ideal import NonIdealHWP
from .input_sky import SkyInput
from .maps_and_harmonics import (
    HealpixMap,
    SphericalHarmonics,
    SpinMap,
    _interpolation_epsilon_for_tod,
    interpolate_alm,
)
from .observation_utilities import (
    ObservationPointings,
    _ArrayPointings,
    observation_pointings,
    _check_detector_fields,
    _detector_keys,
    _get_pol_angle,
)
from .observations import Observation
from .utilities import cos_sin, resolve_nthreads

log = logging.getLogger(__name__)


@njit
def compute_signal_for_one_sample(T, Q, U, co, si, gamma):
    """Bolometric equation"""
    return T + gamma * (co * Q + si * U)


@njit(parallel=True)
def scan_map_for_one_detector(
    tod_det, input_T, input_Q, input_U, pol_angle_det, pol_eff_det
):
    for i in prange(len(tod_det)):  # type: ignore[not-iterable]
        co, si = cos_sin(2 * pol_angle_det[i])
        tod_det[i] += compute_signal_for_one_sample(
            T=input_T[i],
            Q=input_Q[i],
            U=input_U[i],
            co=co,
            si=si,
            gamma=pol_eff_det,
        )


@njit(parallel=True)
def scan_map_generic_hwp_for_one_detector(
    tod_det,
    input_T,
    input_Q,
    input_U,
    orientation_telescope,
    pol_angle_det,
    pol_eff_det,
    hwp_angle,
    mueller_hwp,
):
    """Bolometric equation for a generic HWP Mueller matrix

    (1,0,0,0) x Mpol x Rpol x Rhwp^T x Mhwp x Rhwp x Rtel x Stokes

    The products are expanded in scalars, so that the parallel loop
    allocates nothing and shares no mutable state. The input V is zero
    and the polarimeter has no V component, so the last row and column
    of Mhwp do not contribute.
    """
    co_pol, si_pol = cos_sin(2 * pol_angle_det)
    pol_q = pol_eff_det * co_pol
    pol_u = pol_eff_det * si_pol

    for i in prange(len(tod_det)):  # type: ignore[not-iterable]
        T = input_T[i]

        # Rtel x Stokes
        co_tel, si_tel = cos_sin(2 * orientation_telescope[i])
        q_tel = co_tel * input_Q[i] + si_tel * input_U[i]
        u_tel = -si_tel * input_Q[i] + co_tel * input_U[i]

        # Rhwp x (...)
        co_hwp, si_hwp = cos_sin(2 * hwp_angle[i])
        q_hwp = co_hwp * q_tel + si_hwp * u_tel
        u_hwp = -si_hwp * q_tel + co_hwp * u_tel

        # Mhwp x (...)
        t_m = (
            mueller_hwp[0, 0] * T
            + mueller_hwp[0, 1] * q_hwp
            + mueller_hwp[0, 2] * u_hwp
        )
        q_m = (
            mueller_hwp[1, 0] * T
            + mueller_hwp[1, 1] * q_hwp
            + mueller_hwp[1, 2] * u_hwp
        )
        u_m = (
            mueller_hwp[2, 0] * T
            + mueller_hwp[2, 1] * q_hwp
            + mueller_hwp[2, 2] * u_hwp
        )

        # (1,0,0,0) x Mpol x Rpol x Rhwp^T x (...)
        tod_det[i] += (
            t_m
            + pol_q * (co_hwp * q_m - si_hwp * u_m)
            + pol_u * (si_hwp * q_m + co_hwp * u_m)
        )


def scan_map(
    tod,
    pointings,
    maps: SkyInput,
    pol_angle_detectors: np.ndarray | None = None,
    pol_eff_detectors: np.ndarray | None = None,
    hwp: HWP | None = None,
    hwp_angle: np.ndarray | None = None,
    mueller_hwp: np.ndarray | None = None,
    input_names: list[str] | None = None,
    interpolation: str | None = "",
    pointings_dtype=np.float64,
    nthreads: int | None = None,
):
    """
    Scan a sky map and fill time-ordered data (TOD) based on detector observations.

    This function modifies the values in `tod` by adding the contribution of the
    bolometric equation given a T/Q/U description of the sky. The `pointings`
    argument must be a DxNx2 matrix containing the pointing information, where
    D is the number of detectors for the current observation and N is the size
    of the `tod` array.

    Supported sky descriptions
    --------------------------
    - `HealpixMap`
    - `SphericalHarmonics`
    - `dict[str, HealpixMap]`
    - `dict[str, SphericalHarmonics]`

    In the dictionary case, the key selected for each detector is taken from
    `input_names[detector_idx]`.

    The sky coordinate system is taken from the `coordinates` attribute of the
    selected object (either `HealpixMap` or `SphericalHarmonics`). If
    `coordinates` is ``None``, Galactic coordinates are assumed and a warning
    is issued.

    Parameters
    ----------
    tod : np.ndarray
        Time-ordered data (TOD) array of shape (n_detectors, n_samples) that will be
        filled with the simulated sky signal.

    pointings : np.ndarray, callable or ObservationPointings
        Pointing information for each detector. An :class:`.ObservationPointings`
        gives them in the coordinate system of each map, together with the HWP
        angle if `hwp_angle` is None. If an array, it should have shape
        (n_detectors, n_samples, 2 or 3), where the first two entries are (theta, phi)
        in radians, and the optional third entry is the telescope orientation.
        If a callable, it should return pointing data when passed a detector index.

    maps : HealpixMap | SphericalHarmonics | dict[str, HealpixMap] |
           dict[str, SphericalHarmonics]
        Sky model to be scanned. In dictionary form, the keys must match the
        entries of `input_names`.

    pol_angle_detectors : np.ndarray or None, default=None
        Polarization angles of detectors in radians. If None, all angles are set to zero.

    pol_eff_detectors : np.ndarray or None, default=None
        Polarization efficiency of detectors. If None, all detectors have unit efficiency.

    hwp : HWP, optional
        Half-wave plate (HWP) model. If None, HWP effects are ignored unless
        the `Observation` object contains HWP data.

    hwp_angle : np.ndarray or None, default=None
        Half-wave plate angles of an external HWP object.

    mueller_hwp : np.ndarray or None, default=None
        Mueller matrices for a non-ideal HWP.

    input_names : array-like of str or None, default=None
        Per-detector keys to select entries in `maps` when `maps` is a dictionary.

    interpolation : str or None, default=""
        Currently unused here for the harmonic case; real-space maps are sampled
        by nearest-neighbour (Healpix ang2pix).

    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly.

    nthreads : int or None, default=None
        Number of threads to use for convolution. If None, resolved via
        :func:`.resolve_nthreads`.

    Raises
    ------
    TypeError
        If `maps` is None at this level, or has an unsupported type.
    AssertionError
        If `tod` and `pointings` shapes are inconsistent.

    Notes
    -----
    - The function modifies `tod` in place by adding the scanned sky signal.
    - If `mueller_hwp` is provided, a full HWP Mueller matrix transformation is applied.
    - Polarization angles are corrected based on telescope orientation and HWP effects.
    - This function is crucial for simulating realistic observations in CMB and astrophysical
      experiments.
    """

    nthreads = resolve_nthreads(nthreads)

    n_detectors = tod.shape[0]

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]
    if not isinstance(pointings, ObservationPointings):
        pointings = _ArrayPointings(
            pointings,
            hwp_angle=hwp_angle,
            pointings_dtype=pointings_dtype,
            nthreads=nthreads,
        )

    if pol_angle_detectors is None:
        pol_angle_detectors = np.zeros(n_detectors)

    if pol_eff_detectors is None:
        pol_eff_detectors = np.ones(n_detectors)

    for detector_idx in range(n_detectors):
        # ----------------------------------------------------------
        # Select per-detector sky object
        # ----------------------------------------------------------
        if isinstance(maps, dict):
            if input_names is None:
                raise ValueError(
                    "scan_map: maps is a dict, but input_names is None. "
                    "You must provide per-detector keys."
                )
            key = input_names[detector_idx]
            try:
                maps_det = maps[key]
            except KeyError as exc:
                raise KeyError(
                    f"scan_map: maps does not contain an entry for '{key}' "
                    f"(detector index {detector_idx})"
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

        # ----------------------------------------------------------
        # Get pointings in the correct coordinate system
        # ----------------------------------------------------------
        curr_pointings_det = pointings.detector(
            detector_idx, output_coordinate_system=coordinates
        )
        if hwp_angle is None:
            hwp_angle = pointings.hwp_angle

        # ----------------------------------------------------------
        # REAL SPACE (HealpixMap)
        # ----------------------------------------------------------
        if isinstance(maps_det, HealpixMap):
            pixmap = maps_det.values  # (nstokes, Npix)
            scheme = "NESTED" if maps_det.nest else "RING"
            hpx = Healpix_Base(maps_det.nside, scheme)

            pixel_ind_det = hpx.ang2pix(curr_pointings_det[:, 0:2], nthreads=nthreads)

            if maps_det.nstokes == 1:
                input_T = pixmap[0, pixel_ind_det]
                input_Q = np.zeros_like(input_T)
                input_U = input_Q
            else:
                input_T = pixmap[0, pixel_ind_det]
                input_Q = pixmap[1, pixel_ind_det]
                input_U = pixmap[2, pixel_ind_det]

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

        # ----------------------------------------------------------
        # Apply detector response / HWP
        # ----------------------------------------------------------
        if hwp is None or isinstance(hwp, IdealHWP):
            # With HWP implements:
            # (T + Q ρ Cos[2 (2 α - θ + ψ])] + U ρ Sin[2 (2 α - θ + ψ)])
            # without
            # (T + Q ρ Cos[2 (θ + ψ])] + U ρ Sin[2 (θ + ψ)])
            # ρ: polarization efficiency
            # θ: polarization angle
            # ψ: angle of the telescope
            # α: HWP angle
            scan_map_for_one_detector(
                tod_det=tod[detector_idx],
                input_T=input_T,
                input_Q=input_Q,
                input_U=input_U,
                pol_angle_det=_get_pol_angle(
                    curr_pointings_det, hwp_angle, pol_angle_detectors[detector_idx]
                ),
                pol_eff_det=pol_eff_detectors[detector_idx],
            )
        elif isinstance(hwp, NonIdealHWP):
            # This implements:
            # (1,0,0,0) x Mpol(ρ) x Rpol(θ) x Rhwp(α)^T x Mhwp x Rhwp(α) x Rtel(ψ) x Stokes
            assert mueller_hwp is not None, (
                "Non ideal HWP selected but no mueller matrix provided."
            )
            assert all(m is not None for m in mueller_hwp), (
                "Non ideal hwp type was selected but not all detectors have a mueller matrix associated. Please set det.mueller_hwp attribute."
            )
            scan_map_generic_hwp_for_one_detector(
                tod_det=tod[detector_idx],
                input_T=input_T,
                input_Q=input_Q,
                input_U=input_U,
                orientation_telescope=curr_pointings_det[:, 2],
                pol_angle_det=pol_angle_detectors[detector_idx],
                pol_eff_det=pol_eff_detectors[detector_idx],
                hwp_angle=hwp_angle,
                mueller_hwp=mueller_hwp[detector_idx],
            )
        else:
            raise TypeError(
                "scan_map: hwp must be None, IdealHWP, or NonIdealHWP "
                f"(got {type(hwp)!r})"
            )


def scan_map_in_observations(
    observations: Observation | list[Observation],
    maps: SkyInput | None = None,
    pointings: np.ndarray | list[np.ndarray] | None = None,
    hwp: HWP | None = None,
    component: str = "tod",
    pointings_dtype: DTypeLike = np.float64,
    apply_non_linearity: bool = False,
    add_2f_hwpss: bool = False,
    mueller_phases: dict | None = None,
    integrate_in_band: bool = False,
    nthreads: int | None = None,
):
    """
    Scan a sky map and fill time-ordered data (TOD) for a set of observations.

    This is a wrapper around the :func:`.scan_map` function (and, for the
    harmonic-expansion case, :func:`.fill_tod`) that applies to the TOD
    stored in `observations` and the pointings stored in `pointings`. The two
    types can either be a single :class:`.Observation` instance and a NumPy
    matrix, or a list of observations and a list of NumPy matrices; in the
    latter case, they must have the same number of elements.

    The field `maps` can be:

    - a single :class:`.HealpixMap` instance, used for all detectors;
    - a single :class:`.SphericalHarmonics` instance, used for all detectors;
    - a dictionary mapping detector or channel names (strings) to
      :class:`.HealpixMap` or :class:`.SphericalHarmonics` objects.

    In the harmonic case (when the underlying routines see
    :class:`.SphericalHarmonics` objects), the code performs interpolation using
    the ducc0 spherical harmonics backend. For real-space inputs
    (:class:`.HealpixMap`), no interpolation in harmonic space is performed.

    By default, the signal is added to ``Observation.tod``. If you want to add
    it to some other field of the :class:`.Observation` class, use `component`::

        for cur_obs in sim.observations:
            # Allocate a new TOD for the sky signal alone
            cur_obs.sky_tod = np.zeros_like(cur_obs.tod)

        # Ask `scan_map_in_observations` to store the sky signal
        # in `observations.sky_tod`
        scan_map_in_observations(sim.observations, ..., component="sky_tod")

    Parameters
    ----------
    observations : Observation or list[Observation]
        One or more `Observation` objects containing detector names, pointings,
        and TOD data, to which the computed sky signal will be added.

    maps : HealpixMap, SphericalHarmonics, dict[str, HealpixMap] or dict[str, SphericalHarmonics], optional
        Sky description. If a dictionary, keys should match detector or channel
        names, and values should be :class:`.HealpixMap` or
        :class:`.SphericalHarmonics` instances. If a single object is provided,
        the same sky is used for all detectors. If ``None``, the function
        attempts to read `sky` from the first observation.

    pointings : np.ndarray or list[np.ndarray], optional
        Pointing matrices associated with the observations. If None, the function
        extracts pointing information from the `Observation` objects.

    hwp : HWP, optional
        Half-wave plate (HWP) model. If None, HWP effects are ignored unless the
        `Observation` object itself contains HWP data.

    component : str, default="tod"
        The TOD component in the `Observation` object where the computed signal
        will be stored.

    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly. If the pointing is passed or
        already precomputed this parameter is ineffective. Default is `np.float64`.

    apply_non_linearity : bool, optional
        (For the harmonics expansion case) applies the coupling of the
        non-linearity systematics with `hwp_sys`.

    add_2f_hwpss : bool, optional
        (For the harmonics expansion case) adds the 2f HWP synchronous signal
        to the TOD.

    mueller_phases : dict, optional
        (For the harmonics expansion case) the non-ideal phases of the 2f and
        4f Mueller matrices, as ``{"2f": ..., "4f": ...}``, the same for all
        the detectors. Required with the Mueller formalism.

    integrate_in_band : bool, default=False
        Whether to integrate the signal over the detector's frequency band.
        Only implemented for the Jones formalism and explicit harmonics expansion case.

    nthreads : int, default=None
        Number of threads to use in the convolution. If None, the function reads from the `OMP_NUM_THREADS`
        environment variable.

    Raises
    ------
    ValueError
        If a dictionary `maps` does not contain the required detector or channel keys.
    AssertionError
        If the number of observations and pointings do not match.
        If `tod` and `pointings` shapes are inconsistent.

    Notes
    -----
    - This function modifies `observations` in place by adding the computed sky
      signal to the specified `component` field.
    - If `pointings` is None, the function attempts to extract them from
      `Observation` objects. If the pointing is generated on the fly,
      `pointings_dtype` specifies its type.
    - If an HWP model is provided, the function computes HWP angles and applies
      the corresponding Mueller matrices.
    - This function supports both single observations and lists of observations,
      handling each one separately.
    """

    # Pointings in Ecliptic coordinates by default: scan_map asks for those
    # of the coordinate system of each map
    obs_pointings = observation_pointings(
        observations,
        pointings=pointings,
        hwp=hwp,
        pointings_dtype=pointings_dtype,
        nthreads=nthreads,
    )
    obs_list = [cur_obs for cur_obs, _ in obs_pointings]

    if maps is None:
        try:
            assert obs_list, "No observations available"
            maps = obs_list[0].sky
        except AttributeError:
            msg = (
                "'maps' is None and nothing is found in the observation. "
                "You should either pass the maps here, or store them in "
                "the observations."
            )
            raise AttributeError(msg)

    if not isinstance(maps, (dict, HealpixMap, SphericalHarmonics)):
        raise TypeError(
            "maps must be a HealpixMap, a SphericalHarmonics instance, "
            "or a dict mapping detector/channel names to such objects."
        )
    # The keys of the maps and the fields of all the observations, before
    # modifying any TOD
    input_names_list = [_detector_keys(cur_obs, maps, "sky") for cur_obs in obs_list]
    required_fields = set()
    for cur_obs in obs_list:
        cur_hwp = hwp if hwp is not None else cur_obs.hwp
        if isinstance(cur_hwp, NonIdealHWP) and cur_hwp.harmonic_expansion:
            required_fields.update(_harmonics_fields(cur_hwp))
        else:
            required_fields.update(["pol_angle_rad", "pol_efficiency", "mueller_hwp"])
    _check_detector_fields(obs_list, sorted(required_fields), component)

    for (cur_obs, cur_ptg), input_names in zip(obs_pointings, input_names_list):
        # HWP handling: external HWP overrides, otherwise use obs.hwp
        cur_hwp = hwp if hwp is not None else cur_obs.hwp
        hwp_angle = cur_ptg.hwp_angle

        # Set number of threads
        nthreads = resolve_nthreads(nthreads)

        if isinstance(cur_hwp, NonIdealHWP) and cur_hwp.harmonic_expansion:
            # Harmonic-expansion case: delegate to fill_tod
            fill_tod_with_hwp_harmonics(
                hwp=cur_hwp,
                observation=cur_obs,
                tod=getattr(cur_obs, component),
                pointings=cur_ptg,
                maps=maps,
                hwp_angle=hwp_angle,
                pointings_dtype=pointings_dtype,
                input_names=input_names,
                apply_non_linearity=apply_non_linearity,
                add_2f_hwpss=add_2f_hwpss,
                mueller_phases=mueller_phases,
                integrate_in_band=integrate_in_band,
                nthreads=nthreads,
            )
        else:
            # Standard scanning case
            scan_map(
                tod=getattr(cur_obs, component),
                pointings=cur_ptg,
                maps=maps,
                pol_angle_detectors=cur_obs.pol_angle_rad,
                pol_eff_detectors=cur_obs.pol_efficiency,
                hwp=cur_hwp,
                hwp_angle=hwp_angle,
                mueller_hwp=cur_obs.mueller_hwp,
                input_names=input_names,
                pointings_dtype=pointings_dtype,
                nthreads=nthreads,
            )


@njit(parallel=True)
def scan_spin_maps_for_one_detector(
    tod_det,
    spin_values,
    spins,
    pixel_ind,
    pol_angle_det,
):
    """Add the contribution of a set of spin maps to one detector."""
    n_spins = len(spins)
    for i in prange(len(tod_det)):
        pixel = pixel_ind[i]
        alpha = pol_angle_det[i]
        for ispin in range(n_spins):
            spin = spins[ispin]
            value = spin_values[ispin, pixel]
            if spin == 0:
                # no factor 2, since the convention is that we put T in the
                # spin 0 map, and not T/2
                tod_det[i] += value.real
            else:
                co, si = cos_sin(spin * alpha)
                # factor 2 for spin != 0, since the convention is that we put
                # (Q + iU)/2 in the spin 2 map
                tod_det[i] += 2.0 * (value.real * co + value.imag * si)


def scan_spin_maps(
    tod,
    pointings,
    spin_maps: SpinMap | dict[str, SpinMap],
    detector_names: Sequence[str] | None = None,
    pol_angle_rad: np.ndarray | None = None,
    pointings_dtype=np.float64,
    nthreads: int | None = None,
):
    r"""Scan a set of spin maps and add them to time-ordered data (TOD).

    This function adds to `tod` the contribution of a set of spin maps, such as those used by the the spin-mapmaking codes SMARTIES and SBM.

    .. math::

        d_t = \sum_s c_s \, \mathrm{Re}\!\left(S_s(p_t)\,e^{-i s \alpha_t}\right),

    where :math:`c_s = 1` for :math:`s = 0` and :math:`c_s = 2` otherwise,
    :math:`p_t` is the pixel observed at time :math:`t`, and :math:`\alpha_t =
    \psi_t + \gamma_d` is the parallactic angle of the telescope :math:`\psi_t` plus
    the polarization angle :math:`\gamma_d` of the detector.

    The spin maps are taken from :class:`.SpinMap` objects, which hold one
    complex map per spin order, in the detector frame, and are selected per
    detector. A single :class:`.SpinMap` is used for all the detectors when
    `spin_maps` is not a dictionary.

    Parameters
    ----------
    tod : np.ndarray
        Time-ordered data of shape ``(n_detectors, n_samples)``, modified in
        place.

    pointings : np.ndarray, callable or ObservationPointings
        Pointing information for each detector. An
        :class:`.ObservationPointings` gives them in the coordinate system of
        each map. If an array, it must have shape
        ``(n_detectors, n_samples, 3)``, holding the colatitude and longitude
        in radians and the telescope orientation. If a callable, it is called
        like :meth:`.Observation.get_pointings` and returns the pointings.

    spin_maps : SpinMap or dict[str, SpinMap]
        Spin maps to scan. In dictionary form, the key for each detector is
        taken from `detector_names`.

    detector_names : sequence of str, optional
        Names used as keys of `spin_maps`, one per detector. It is required
        when `spin_maps` is a dictionary.

    pol_angle_rad : np.ndarray, optional
        Polarization angle of each detector in radians, one entry per
        detector. If ``None``, all the angles are set to zero.

    pointings_dtype : dtype, default=np.float64
        Data type used when pointings are computed on the fly.

    nthreads : int, optional
        Number of threads used by ``ang2pix``. If ``None``, it is resolved
        through :func:`.resolve_nthreads`.

    Raises
    ------
    TypeError
        If `spin_maps` is neither a :class:`.SpinMap` nor a dictionary of
        :class:`.SpinMap` objects.

    ValueError
        If `spin_maps` is a dictionary but `detector_names` is missing or does
        not have one entry per detector.

    KeyError
        If `spin_maps` does not contain an entry for a detector.

    AssertionError
        If `tod` and `pointings` are both arrays with inconsistent shapes.

    Notes
    -----
    - The presence of an HWP is currently unsupported
    - The signal is added in place; the function returns ``None``.
    - Maps are sampled with nearest-neighbour lookup (``ang2pix``). Only
      real-space maps are supported.
    - The spin maps are evaluated at the observed pixel on the fly, so no
      array indexed by spin and by sample is ever allocated.
    """
    nthreads = resolve_nthreads(nthreads)

    n_detectors = tod.shape[0]

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

    if not isinstance(pointings, ObservationPointings):
        pointings = _ArrayPointings(
            pointings,
            pointings_dtype=pointings_dtype,
            nthreads=nthreads,
        )

    if isinstance(spin_maps, SpinMap):
        maps_per_detector = [spin_maps] * n_detectors
    elif isinstance(spin_maps, dict):
        if detector_names is None:
            raise ValueError(
                "scan_spin_maps: spin_maps is a dict, but detector_names is None. "
                "You must provide the names used as keys."
            )
        detector_names = list(detector_names)
        if len(detector_names) != n_detectors:
            raise ValueError(
                f"scan_spin_maps: detector_names has {len(detector_names)} "
                f"entries but tod has {n_detectors} detectors"
            )
        missing_names = [name for name in detector_names if name not in spin_maps]
        if missing_names:
            raise KeyError(
                "scan_spin_maps: spin_maps does not contain an entry for "
                + ", ".join(repr(name) for name in missing_names)
            )
        maps_per_detector = []
        for name in detector_names:
            cur_maps = spin_maps[name]
            if not isinstance(cur_maps, SpinMap):
                raise TypeError(
                    "scan_spin_maps: every entry of spin_maps must be a "
                    f"SpinMap, got {type(cur_maps)!r} for '{name}'"
                )
            maps_per_detector.append(cur_maps)
    else:
        raise TypeError(
            "scan_spin_maps: spin_maps must be a SpinMap or a dict of SpinMap "
            f"objects, got {type(spin_maps)!r}"
        )

    if not maps_per_detector:
        return

    if pol_angle_rad is None:
        pol_angle_rad = np.zeros(n_detectors)
    detector_pol_angles = np.asarray(pol_angle_rad, dtype=np.float64)
    if detector_pol_angles.shape != (n_detectors,):
        raise ValueError(
            "scan_spin_maps: pol_angle_rad must have one entry per "
            f"detector ({n_detectors}), got shape {detector_pol_angles.shape}"
        )

    if any(cur.coordinates is None for cur in maps_per_detector):
        log.warning(
            "scan_spin_maps: some spin maps have coordinates=None — assuming "
            "CoordinateSystem.Galactic for them"
        )

    for detector_idx in range(n_detectors):
        cur_maps = maps_per_detector[detector_idx]

        coordinates = cur_maps.coordinates
        if coordinates is None:
            coordinates = CoordinateSystem.Galactic

        curr_pointings_det = pointings.detector(
            detector_idx, output_coordinate_system=coordinates
        )

        scheme = "NESTED" if cur_maps.nest else "RING"
        hpx = Healpix_Base(cur_maps.nside, scheme)
        pixel_ind_det = hpx.ang2pix(curr_pointings_det[:, 0:2], nthreads=nthreads)

        pol_angle_det = _get_pol_angle(
            curr_pointings_det,
            None,
            detector_pol_angles[detector_idx],
        )

        scan_spin_maps_for_one_detector(
            tod_det=tod[detector_idx],
            spin_values=cur_maps.values,
            spins=cur_maps.spins,
            pixel_ind=pixel_ind_det,
            pol_angle_det=pol_angle_det,
        )


def scan_spin_maps_in_observations(
    observations: Observation | list[Observation],
    spin_maps: SpinMap | dict[str, SpinMap],
    pointings: np.ndarray | list[np.ndarray] | None = None,
    component: str = "tod",
    pointings_dtype: DTypeLike = np.float64,
    nthreads: int | None = None,
):
    r"""Scan spin maps and add them to the TODs of a set of observations.

    This is a wrapper around :func:`.scan_spin_maps` that reads the TODs, the
    pointings, the detector polarization angles and the detector names from
    the :class:`.Observation` objects. It accepts either a single
    :class:`.Observation` and a pointing matrix, or a list of observations
    and a list of pointing matrices with the same length.

    The maps are selected through the names of the detectors of the current
    observation, so a dictionary containing only the detectors owned by the
    current MPI process is enough.

    Parameters
    ----------
    observations : Observation or list[Observation]
        One or more observations whose TODs will be filled.

    spin_maps : SpinMap or dict[str, SpinMap]
        Spin maps to scan. In dictionary form, the key for each detector is
        taken from the detector names of the observation.

    pointings : np.ndarray or list[np.ndarray], optional
        Pointing matrices associated with the observations. If ``None``, the
        pointings are read from the observations.

    component : str, default="tod"
        The TOD component of the :class:`.Observation` objects to fill.

    pointings_dtype : dtype, default=np.float64
        Data type used when pointings are computed on the fly.

    nthreads : int, optional
        Number of threads used by ``ang2pix``. If ``None``, it is resolved
        through :func:`.resolve_nthreads`.

    Raises
    ------
    AssertionError
        If the number of observations and pointings do not match.

    KeyError
        If `spin_maps` does not contain an entry for a detector.

    Notes
    -----
    - The TODs are modified in place; the function returns ``None``.
    - No MPI communication is performed: each process fills the observations
      it owns.
    """
    for cur_obs, cur_pointings in observation_pointings(
        observations,
        pointings,
        pointings_dtype=pointings_dtype,
        nthreads=nthreads,
    ):
        detector_names = list(cur_obs.name) if isinstance(spin_maps, dict) else None

        scan_spin_maps(
            tod=getattr(cur_obs, component),
            pointings=cur_pointings,
            spin_maps=spin_maps,
            detector_names=detector_names,
            pol_angle_rad=cur_obs.pol_angle_rad,
            pointings_dtype=pointings_dtype,
            nthreads=nthreads,
        )
