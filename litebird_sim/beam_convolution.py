# -*- encoding: utf-8 -*-

import logging
import os
from dataclasses import dataclass
from typing import Union, List, Dict, Optional

import numpy as np
import numpy.typing as npt
from ducc0.totalconvolve import Interpolator, Interpolator_f

from .coordinates import CoordinateSystem
from .hwp import HWP
from .mueller_convolver import MuellerConvolver
from .observations import Observation
from .pointings_in_obs import (
    _get_hwp_angle,
    _normalize_observations_and_pointings,
    _get_centered_pointings,
    _get_pointings_array,
)
from .spherical_harmonics import SphericalHarmonics

# Name of the environment variable used in the convolution
NUM_THREADS_ENVVAR = "OMP_NUM_THREADS"


@dataclass
class BeamConvolutionParameters:
    """Parameters used by the 4π beam convolution code

    Fields:

    - ``lmax`` (int): Maximum value for ℓ for the sky and beam coefficients
    - ``mmax`` (int): Maximum value for m (azimuthal moment) for beam coefficients,
      must be ≦ lmax - 4
    - ``single_precision`` (bool): Set it to ``False`` to use 64-bit floating points
      in the calculation
    - ``epsilon`` (float): The desired relative accuracy of the interpolation
    - ``strict_typing`` (bool): If ``True`` (the default), a ``TypeError` exception
      will be raised if the type of the pointings does not match ``single_precision``.
      If ``False``, the type of the pointings will be converted silently to make the
      code works at the expense of additional memory consumption.
    """

    lmax: int
    mmax: int
    single_precision: bool = True
    epsilon: float = 1e-5
    strict_typing: bool = True


def add_convolved_sky_to_one_detector(
    tod_det,
    sky_alms_det: SphericalHarmonics,
    beam_alms_det: SphericalHarmonics,
    pointings_det,
    mueller_matrix,
    hwp_angle,
    convolution_params: Optional[BeamConvolutionParameters] = None,
    nside_centering: Union[int, None] = None,
    nthreads: int = 0,
):
    """
    Convolve given sky alms with a detector beam alms and add the result to the TOD of a single detector.

    Parameters
    ----------
    tod_det : np.ndarray
        Time-ordered data (TOD) for the given detector, to which the convolved signal is added.
    sky_alms_det : SphericalHarmonics
        Spherical harmonic coefficients representing the sky map for this detector.
    beam_alms_det : SphericalHarmonics
        Spherical harmonic coefficients representing the beam function for this detector.
    pointings_det : np.ndarray
        Pointing information for the given detector.
    mueller_matrix : np.ndarray or None
        Mueller matrix of the HWP for the given detector. If None, the classic 4π convolution is used.
    hwp_angle : np.ndarray or None
        Half-wave plate (HWP) angle values for the given detector. If None, the classic
        4π convolution is used.
    convolution_params : BeamConvolutionParameters, optional
        Parameters controlling the convolution, such as resolution and precision. If None,
        reasonable defaults are chosen based on the sky and beam properties.
    nside_centering : int, default=None
        If set, shifts the detector pointings to the centers of the corresponding HEALPix pixels
        at the given NSIDE resolution. If None, no centering is applied.
    nthreads : int, default=0
        Number of threads to use for convolution. If set to 0, all available CPU cores
        will be used.

    Raises
    ------
    TypeError
        If `strict_typing` is enabled and the data type of `pointings_det` does not match
        the expected precision.

    Notes
    -----
    - If no HWP is present, a standard 4π convolution is performed.
    - If HWP is present, the Mueller matrix is used to properly handle polarization.
    - The function modifies `tod_det` in place by adding the convolved signal.
    """

    if not convolution_params:
        sky_lmax = sky_alms_det.lmax

        beam_lmax = beam_alms_det.lmax
        beam_mmax = beam_alms_det.mmax

        default_lmax = min(sky_lmax, beam_lmax)
        default_mmax = min(default_lmax - 4, beam_mmax)

        logging.warning(
            (
                "No convolution parameters, I will use the defaults "
                "(ℓ_max={lmax}, m_max={mmax}), but this "
                "might lead to unexpected errors and "
                "gross misestimates"
            ).format(lmax=default_lmax, mmax=default_mmax)
        )
        convolution_params = BeamConvolutionParameters(
            lmax=default_lmax,
            mmax=default_mmax,
        )

    else:
        assert convolution_params.lmax - 4 >= convolution_params.mmax, (
            "Error in the convolution parameters m_max must be ≦ ℓ_max-4!"
            "Here ℓ_max={lmax} and m_max={mmax}!"
        ).format(lmax=convolution_params.lmax, mmax=convolution_params.mmax)

    if convolution_params.lmax != sky_alms_det.lmax:
        _slm = sky_alms_det.resize_alm(
            convolution_params.lmax, convolution_params.lmax, inplace=False
        ).values
    else:
        _slm = sky_alms_det.values

    if (convolution_params.lmax != beam_alms_det.lmax) or (
        convolution_params.mmax != beam_alms_det.mmax
    ):
        _blm = beam_alms_det.resize_alm(
            convolution_params.lmax, convolution_params.mmax, inplace=False
        ).values
    else:
        _blm = beam_alms_det.values

    if hwp_angle is None:
        # we cannot simulate HWP, so let's use classic 4pi convolution
        if convolution_params.single_precision:
            real_type = np.float32
            complex_type = np.complex64
            intertype = Interpolator_f
        else:
            real_type = np.float64
            complex_type = np.complex128
            intertype = Interpolator

        inter = intertype(
            sky=_slm.astype(complex_type),
            beam=_blm.astype(complex_type),
            separate=False,
            lmax=convolution_params.lmax,
            kmax=convolution_params.mmax,
            epsilon=convolution_params.epsilon,
            nthreads=nthreads,
        )

        if convolution_params.strict_typing and (pointings_det.dtype != real_type):
            raise TypeError(
                "pointings are {} but they should be {}; consider "
                "passing `strict_typing=False` to BeamConvolutionParameters.".format(
                    pointings_det.dtype, real_type
                )
            )
        tod_det += inter.interpol(
            pointings_det.astype(real_type, copy=False)
            if nside_centering is None
            else _get_centered_pointings(
                pointings_det.astype(real_type, copy=False), nside_centering
            )
        )[0]
    else:
        fullconv = MuellerConvolver(
            slm=_slm,
            blm=_blm,
            mueller=mueller_matrix,
            lmax=convolution_params.lmax,
            kmax=convolution_params.mmax,
            single_precision=convolution_params.single_precision,
            epsilon=convolution_params.epsilon,
            nthreads=nthreads,
        )
        tod_det += fullconv.signal(
            ptg=(
                pointings_det
                if nside_centering is None
                else _get_centered_pointings(pointings_det, nside_centering)
            ),
            alpha=hwp_angle,
            strict_typing=convolution_params.strict_typing,
        )


def add_convolved_sky(
    tod,
    pointings,
    sky_alms: Union[SphericalHarmonics, Dict[str, SphericalHarmonics]],
    beam_alms: Union[SphericalHarmonics, Dict[str, SphericalHarmonics]],
    hwp_angle: Union[np.ndarray, None] = None,
    mueller_hwp: Union[np.ndarray, None] = None,
    input_sky_names: Union[str, None] = None,
    input_beam_names: Union[str, None] = None,
    convolution_params: Optional[BeamConvolutionParameters] = None,
    input_sky_alms_in_galactic: bool = True,
    pointings_dtype=np.float64,
    nside_centering: Union[int, None] = None,
    nthreads: int = 0,
):
    """
    Convolve a set of sky maps with detector beams and add the resulting signals to the
    time-ordered data (TOD) for multiple detectors.

    Parameters
    ----------
    tod : np.ndarray
        Time-ordered data (TOD) for multiple detectors, to which the convolved signal is added.
    pointings : np.ndarray or callable
        Pointing information for each detector. If an array, it should have shape
        (n_detectors, n_samples, 3). If a callable, it should return pointing data when
        passed a detector index.
    sky_alms : Union[SphericalHarmonics, Dict[str, SphericalHarmonics]]
        Spherical harmonic coefficients representing the sky maps. If a dictionary, keys should
        correspond to detector or channel names.
    beam_alms : Union[SphericalHarmonics, Dict[str, SphericalHarmonics]]
        Spherical harmonic coefficients representing the beam functions. If a dictionary,
        keys should correspond to detector or channel names.
    hwp_angle : np.ndarray or None, default=None
        Half-wave plate (HWP) angle values for each detector. If None, the classic 4π
        convolution is used.
    mueller_hwp : np.ndarray or None, default=None
        Mueller matrices of the HWP. If None, the classic 4π convolution is used.
    input_sky_names : str or None, default=None
        Names of the sky maps to use for each detector. If None, all detectors use the same sky.
    input_beam_names : str or None, default=None
        Names of the beam maps to use for each detector. If None, all detectors use the same beam.
    convolution_params : BeamConvolutionParameters, optional
        Parameters controlling the convolution, such as resolution and precision. If None,
        reasonable defaults are chosen based on the sky and beam properties, but a warning
        will be raised.
    input_sky_alms_in_galactic : bool, default=True
        Whether the input sky maps are provided in Galactic coordinates. If False, they are
        assumed to be in equatorial coordinates.
    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly. If the pointing is passed or
        already precomputed this parameter is ineffective. Default is `np.float64`.
    nside_centering : int, default=None
        If set, shifts the detector pointings to the centers of the corresponding HEALPix pixels
        at the given NSIDE resolution. If None, no centering is applied.
    nthreads : int, default=0
        Number of threads to use for convolution. If set to 0, all available CPU cores
        will be used.

    Raises
    ------
    ValueError
        If dictionary keys in `sky_alms` or `beam_alms` do not match detector names.
    AssertionError
        If `tod` and `pointings` shapes are inconsistent.

    Notes
    -----
    - This function loops over all detectors and applies `add_convolved_sky_to_one_detector`
      to each one.
    - If `input_sky_names` and `input_beam_names` are provided, they must match the structure
      of the `tod` array.
    - The function modifies `tod` in place by adding the convolved signals for all detectors.
    """

    if mueller_hwp is not None:
        assert tod.shape[0] == mueller_hwp.shape[0]

    n_detectors = tod.shape[0]

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

    for detector_idx in range(n_detectors):
        if input_sky_alms_in_galactic:
            output_coordinate_system = CoordinateSystem.Galactic
        else:
            output_coordinate_system = CoordinateSystem.Ecliptic

        curr_pointings_det, hwp_angle = _get_pointings_array(
            detector_idx=detector_idx,
            pointings=pointings,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
            pointings_dtype=pointings_dtype,
        )

        # FIXME: Fix this at some point, ducc wants phi 0 -> 2pi
        curr_pointings_det[:, 1] = np.mod(curr_pointings_det[:, 1], 2 * np.pi)

        if input_sky_names is None:
            sky_alms_det = sky_alms
        else:
            sky_alms_det = sky_alms[input_sky_names[detector_idx]]

        if input_beam_names is None:
            beam_alms_det = beam_alms
        else:
            beam_alms_det = beam_alms[input_beam_names[detector_idx]]

        if mueller_hwp is None:
            mueller_matrix = None
        else:
            mueller_matrix = mueller_hwp[detector_idx]

        add_convolved_sky_to_one_detector(
            tod_det=tod[detector_idx],
            sky_alms_det=sky_alms_det,
            beam_alms_det=beam_alms_det,
            pointings_det=curr_pointings_det,
            mueller_matrix=mueller_matrix,
            hwp_angle=hwp_angle,
            convolution_params=convolution_params,
            nside_centering=nside_centering,
            nthreads=nthreads,
        )


def add_convolved_sky_to_observations(
    observations: Union[Observation, List[Observation]],
    sky_alms: Union[
        SphericalHarmonics, Dict[str, SphericalHarmonics]
    ],  # at some point optional, taken from the obs
    beam_alms: Union[
        SphericalHarmonics, Dict[str, SphericalHarmonics]
    ],  # at some point optional, taken from the obs
    pointings: Union[npt.ArrayLike, List[npt.ArrayLike], None] = None,
    hwp: Optional[HWP] = None,
    input_sky_alms_in_galactic: bool = True,
    convolution_params: Optional[BeamConvolutionParameters] = None,
    component: str = "tod",
    pointings_dtype=np.float64,
    nside_centering: Union[int, None] = None,
    nthreads: Union[int, None] = None,
):
    """
    Applies beam convolution to sky maps and adds the resulting signal to the TOD of one or more observations.

    This function processes one or multiple `Observation` objects, handling sky and beam convolution for
    all detectors within each observation. It supports Galactic coordinate transformations and HWP effects.

    Parameters
    ----------
    observations : Observation or list of Observation
        A single Observation object or a list of them, containing detector names, pointings, and TOD data.
    sky_alms : SphericalHarmonics or dict
        The spherical harmonics representation of the sky signal, either as a single object or a dictionary
        keyed by detector/channel names.
    beam_alms : SphericalHarmonics or dict
        The spherical harmonics representation of the detector beams, either as a single object or a dictionary
        keyed by detector/channel names.
    pointings : np.ndarray, list of np.ndarray, or None, default=None
        Detector pointing matrices. If None, the function extracts pointings from the `Observation` objects.
    hwp : Optional[HWP], default=None
        Half-Wave Plate (HWP) parameters. If None, the function either assumes the information stored in the
        `Observation` objects, or, if they are absent, assumes no HWP.
    input_sky_alms_in_galactic : bool, default=True
        Whether the input sky alms are in Galactic coordinates.
    convolution_params : Optional[BeamConvolutionParameters], default=None
        Parameters controlling the beam convolution, including resolution limits and numerical precision.
    component : str, default="tod"
        The name of the TOD component to which the computed data is added.
    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly. If the pointing is passed or
        already precomputed this parameter is ineffective. Default is `np.float64`.
    nside_centering : int, default=None
        If set, shifts the detector pointings to the centers of the corresponding HEALPix pixels
        at the given NSIDE resolution. If None, no centering is applied.
    nthreads : int, default=None
        Number of threads to use in the convolution. If None, the function reads from the `OMP_NUM_THREADS`
        environment variable.

    Notes
    -----
    - If `pointings` is not provided, it is inferred from the `Observation` objects. If the pointing is passed
        or already precomputed this parameter is ineffective. Default is `np.float32`.
    - The function determines the correct sky and beam harmonics from the detector names or channels.
    - Calls `add_convolved_sky` to process the TOD for all detectors.
    """

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )

    if sky_alms is None:
        try:
            sky_alms = observations[0].sky
        except AttributeError:
            msg = "'sky_alms' is None and nothing is found in the observation. You should either pass the spherical harmonics, or store them in the observations if 'mbs' is used."
            raise AttributeError(msg)
        assert sky_alms["type"] == "alms", (
            "'sky_alms' should be of type 'alms'. Use 'store_alms' of 'MbsParameters' to make it so."
        )

    if beam_alms is None:
        try:
            beam_alms = observations[0].blms
        except AttributeError:
            msg = "'beam_alms' is None and nothing is found in the observation. You should either pass the spherical harmonics of the beam, or store them in the observations if 'get_gauss_beam_alms' is used."
            raise AttributeError(msg)

    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        # Determine input sky names
        if isinstance(sky_alms, dict):
            input_sky_names = (
                cur_obs.name
                if all(k in sky_alms for k in cur_obs.name)
                else cur_obs.channel
                if all(k in sky_alms for k in cur_obs.channel)
                else None
            )
            if input_sky_names is None:
                raise ValueError(
                    "Sky a_lm dictionary keys do not match detector/channel names."
                )

            if "Coordinates" in sky_alms:
                dict_input_sky_alms_in_galactic = (
                    sky_alms["Coordinates"] is CoordinateSystem.Galactic
                )
                if dict_input_sky_alms_in_galactic != input_sky_alms_in_galactic:
                    logging.warning(
                        "Overriding `input_sky_alms_in_galactic` from sky_alms dictionary."
                    )
                input_sky_alms_in_galactic = dict_input_sky_alms_in_galactic
        else:
            assert isinstance(sky_alms, SphericalHarmonics), "Invalid sky_alms format."
            input_sky_names = None

        # Determine input beam names
        if isinstance(beam_alms, dict):
            input_beam_names = (
                cur_obs.name
                if all(k in beam_alms for k in cur_obs.name)
                else cur_obs.channel
                if all(k in beam_alms for k in cur_obs.channel)
                else None
            )
            if input_beam_names is None:
                raise ValueError(
                    "Beam a_lm dictionary keys do not match detector/channel names."
                )
        else:
            assert isinstance(beam_alms, SphericalHarmonics), (
                "Invalid beam_alms format."
            )
            input_beam_names = None

        # If you pass an external HWP, get hwp_angle here, otherwise this is handled in add_convolved_sky
        if hwp is None:
            hwp_angle = None
        else:
            hwp_angle = _get_hwp_angle(
                obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype
            )

        # Set number of threads
        if nthreads is None:
            nthreads = int(os.environ.get(NUM_THREADS_ENVVAR, 0))

        # Perform convolution
        add_convolved_sky(
            tod=getattr(cur_obs, component),
            pointings=cur_ptg,
            sky_alms=sky_alms,
            beam_alms=beam_alms,
            hwp_angle=hwp_angle,
            mueller_hwp=cur_obs.mueller_hwp,
            input_sky_names=input_sky_names,
            input_beam_names=input_beam_names,
            convolution_params=convolution_params,
            input_sky_alms_in_galactic=input_sky_alms_in_galactic,
            pointings_dtype=pointings_dtype,
            nside_centering=nside_centering,
            nthreads=nthreads,
        )
