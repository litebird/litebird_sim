import numpy as np
from numba import njit, prange
import os

from ducc0.healpix import Healpix_Base
from .observations import Observation
from .hwp_harmonics import fill_tod
from .hwp import HWP, IdealHWP, NonIdealHWP
from .pointings_in_obs import (
    _get_hwp_angle,
    _get_pointings_array,
    _get_pol_angle,
    _normalize_observations_and_pointings,
)
from .coordinates import CoordinateSystem

from .maps_and_harmonics import SphericalHarmonics, HealpixMap, interpolate_alm
from .constants import NUM_THREADS_ENVVAR

import logging


@njit
def vec_stokes(stokes, T, Q, U):
    stokes[0] = T
    stokes[1] = Q
    stokes[2] = U


@njit
def vec_polarimeter(angle, gamma):
    # (1,0,0,0) x Mpol x Rpol
    vec = np.empty(4, dtype=np.float64)
    vec[0] = 1
    vec[1] = gamma * np.cos(2 * angle)
    vec[2] = gamma * np.sin(2 * angle)
    vec[3] = 0
    return vec


@njit
def rot_matrix(mat, angle):
    ca = np.cos(2 * angle)
    sa = np.sin(2 * angle)
    mat[1, 1:3] = ca, sa
    mat[2, 1:3] = -sa, ca


@njit
def compute_signal_for_one_sample(T, Q, U, co, si, gamma):
    """Bolometric equation"""
    return T + gamma * (co * Q + si * U)


@njit(parallel=True)
def scan_map_for_one_detector(
    tod_det, input_T, input_Q, input_U, pol_angle_det, pol_eff_det
):
    for i in prange(len(tod_det)):
        tod_det[i] += compute_signal_for_one_sample(
            T=input_T[i],
            Q=input_Q[i],
            U=input_U[i],
            co=np.cos(2 * pol_angle_det[i]),
            si=np.sin(2 * pol_angle_det[i]),
            gamma=pol_eff_det,
        )


@njit
def compute_signal_generic_hwp_for_one_sample(Stokes, Vpol, Rhwp, Mhwp, Rtel):
    """Bolometric equation for generic HWP Mueller matrix
    (1,0,0,0) x Mpol x Rpol x Rhwp^T x Mhwp x Rhwp x Rtel x Stokes
    """
    return Vpol @ Rhwp.T @ Mhwp @ Rhwp @ Rtel @ Stokes


@njit
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
    polarimeter = vec_polarimeter(pol_angle_det, pol_eff_det)

    vec_S = np.zeros(4, dtype=np.float64)
    rot_hwp = np.eye(4, dtype=np.float64)
    rot_tel = np.eye(4, dtype=np.float64)

    for i in range(len(tod_det)):
        vec_stokes(vec_S, input_T[i], input_Q[i], input_U[i])
        rot_matrix(rot_hwp, hwp_angle[i])
        rot_matrix(rot_tel, orientation_telescope[i])

        tod_det[i] += compute_signal_generic_hwp_for_one_sample(
            Stokes=vec_S,
            Vpol=polarimeter,
            Rhwp=rot_hwp,
            Mhwp=mueller_hwp,
            Rtel=rot_tel,
        )


def scan_map(
    tod,
    pointings,
    maps: (
        HealpixMap
        | dict[str, HealpixMap]
        | SphericalHarmonics
        | dict[str, SphericalHarmonics]
    ),
    pol_angle_detectors: np.ndarray | None = None,
    pol_eff_detectors: np.ndarray | None = None,
    hwp: HWP | None = None,
    hwp_angle: np.ndarray | None = None,
    mueller_hwp: np.ndarray | None = None,
    input_names: str | None = None,
    interpolation: str | None = "",
    pointings_dtype=np.float64,
    nthreads: int = 0,
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

    pointings : np.ndarray or callable
        Pointing information for each detector. If an array, it should have shape
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

    nthreads : int, default=0
        Number of threads to use for convolution. If set to 0, all available CPU cores
        will be used.

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

    n_detectors = tod.shape[0]

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

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
            logging.warning(
                "scan_map: maps_det.coordinates is None — assuming "
                "CoordinateSystem.Galactic"
            )
            coordinates = CoordinateSystem.Galactic

        # ----------------------------------------------------------
        # Get pointings in the correct coordinate system
        # ----------------------------------------------------------
        curr_pointings_det, hwp_angle = _get_pointings_array(
            detector_idx=detector_idx,
            pointings=pointings,
            hwp_angle=hwp_angle,
            output_coordinate_system=coordinates,
            pointings_dtype=pointings_dtype,
        )

        # ----------------------------------------------------------
        # REAL SPACE (HealpixMap)
        # ----------------------------------------------------------
        if isinstance(maps_det, HealpixMap):
            pixmap = maps_det.values  # (nstokes, Npix)
            scheme = "NESTED" if maps_det.nest else "RING"
            hpx = Healpix_Base(maps_det.nside, scheme)

            pixel_ind_det = hpx.ang2pix(curr_pointings_det[:, 0:2])

            if maps_det.nstokes == 1:
                input_T = pixmap[0, pixel_ind_det]
                input_Q = np.zeros_like(input_T)
                input_U = np.zeros_like(input_T)
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


def scan_map_in_observations(
    observations: Observation | list[Observation],
    maps: (
        HealpixMap
        | dict[str, HealpixMap]
        | SphericalHarmonics
        | dict[str, SphericalHarmonics]
        | None
    ) = None,
    pointings: np.ndarray | list[np.ndarray] | None = None,
    hwp: HWP | None = None,
    component: str = "tod",
    pointings_dtype: np.dtype = np.float64,
    save_tod: bool = True,
    apply_non_linearity: bool = False,
    add_2f_hwpss: bool = False,
    mueller_phases: np.ndarray | None = None,
    comm: bool | None = None,
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

    mueller_phases : dict or np.ndarray, optional
        (For the harmonics expansion case) the non-ideal phases for the Mueller
        matrix elements. When None is given, temporary values from
        Patanchon et al. [2021] are used.

    comm : SerialMpiCommunicator, optional
        (For the harmonics expansion case) MPI communicator.

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

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )

    if maps is None:
        try:
            maps = observations[0].sky
        except AttributeError:
            msg = (
                "'maps' is None and nothing is found in the observation. "
                "You should either pass the maps here, or store them in "
                "the observations."
            )
            raise AttributeError(msg)
        # If you still have old dict-based `sky` objects with a "type" field,
        # you can re-enable the check below:
        # assert maps["type"] == "maps", (
        #     "'maps' should be of type 'maps'. Disable 'store_alms' in "
        #     "'MbsParameters' to make it so."
        # )

    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        # Decide how to select maps for this observation
        if isinstance(maps, dict):
            if all(name in maps for name in cur_obs.name):
                input_names = cur_obs.name
            elif all(ch in maps for ch in cur_obs.channel):
                input_names = cur_obs.channel
            else:
                raise ValueError(
                    "The dictionary 'maps' does not contain all the relevant "
                    "keys; please check the list of detectors and channels."
                )
        else:
            if not isinstance(maps, (HealpixMap, SphericalHarmonics)):
                raise TypeError(
                    "maps must be a HealpixMap, a SphericalHarmonics instance, "
                    "or a dict mapping detector/channel names to such objects."
                )
            input_names = None

        # HWP handling: external HWP overrides, otherwise use obs.hwp
        cur_hwp = hwp if hwp is not None else cur_obs.hwp

        # Could still be None here; _get_hwp_angle should handle that
        hwp_angle = _get_hwp_angle(
            obs=cur_obs, hwp=cur_hwp, pointing_dtype=pointings_dtype
        )

        # Set number of threads
        if nthreads is None:
            nthreads = int(os.environ.get(NUM_THREADS_ENVVAR, 0))

        if isinstance(cur_hwp, NonIdealHWP) and cur_hwp.harmonic_expansion:
            # Harmonic-expansion case: delegate to fill_tod
            fill_tod(
                observations=cur_obs,
                pointings=cur_ptg,
                maps=maps,
                hwp_angle=hwp_angle,
                pointings_dtype=pointings_dtype,
                save_tod=save_tod,
                input_names=input_names,
                apply_non_linearity=apply_non_linearity,
                add_2f_hwpss=add_2f_hwpss,
                mueller_phases=mueller_phases,
                comm=comm,
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
