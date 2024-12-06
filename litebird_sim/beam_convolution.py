# -*- encoding: utf-8 -*-

import logging
from dataclasses import dataclass
from typing import Union, List, Dict, Optional

from ducc0.totalconvolve import Interpolator, Interpolator_f
import numpy as np

from .coordinates import rotate_coordinates_e2g, CoordinateSystem
from .hwp import HWP
from .mueller_convolver import MuellerConvolver
from .observations import Observation
from .pointings import get_hwp_angle


@dataclass
class BeamConvolutionParameters:
    """Parameters used by the 4π beam convolution code

    Fields:

    - ``lmax`` (int): Maximum value for ℓ for the sky and beam coefficients
    - ``kmax`` (int): Maximum value for m (azimuthal moment) for beam coefficients
    - ``single_precision`` (bool): Set it to ``False`` to use 64-bit floating points
      in the calculation
    - ``epsilon`` (float): The desired relative accuracy of the interpolation
    """

    lmax: int
    kmax: int
    single_precision: bool = True
    epsilon: float = 1e-7


def add_convolved_sky_to_one_detector(
    tod_det,
    sky_alms_det,
    beam_alms_det,
    mueller_matrix,
    pointings_det,
    hwp_angle,
    convolution_params: Optional[BeamConvolutionParameters] = None,
    nthreads: int = 0,
):
    """ """

    if not convolution_params:
        if sky_alms_det.ndim == 2:
            sky_lamx = len(sky_alms_det[0])
        else:
            sky_lamx = len(sky_alms_det)

        if beam_alms_det.ndim == 2:
            beam_lamx = len(beam_alms_det[0])
        else:
            beam_lamx = len(beam_alms_det)

        default_lmax = min(sky_lamx, beam_lamx)
        default_kmax = default_lmax - 4
        logging.warning(
            (
                "No convolution parameters, I will use the defaults "
                "(ℓ_max={lmax}, m_max={kmax}), but this "
                "might lead to unexpected errors and "
                "gross misestimates"
            ).format(lmax=default_lmax, kmax=default_kmax)
        )
        convolution_params = BeamConvolutionParameters(
            lmax=default_lmax,
            kmax=default_kmax,
        )

    if (
        hwp_angle is None
    ):  # we cannot simulate HWP, so let's use classic 4pi convolution
        if convolution_params.single_precision:
            _ftype = np.float32
            _ctype = np.complex64
            intertype = Interpolator_f
        else:
            _ftype = np.float64
            _ctype = np.complex128
            intertype = Interpolator

        _slm = sky_alms_det.astype(_ctype)

        inter = intertype(
            sky=_slm,
            beam=beam_alms_det,
            separate=False,
            lmax=convolution_params.lmax,
            kmax=convolution_params.kmax,
            epsilon=convolution_params.epsilon,
            nthreads=nthreads,
        )
        tod_det += inter.interpol(pointings_det.astype(_ftype))
    else:
        fullconv = MuellerConvolver(
            slm=sky_alms_det,
            blm=beam_alms_det,
            mueller=mueller_matrix,
            lmax=convolution_params.lmax,
            kmax=convolution_params.kmax,
            single_precision=convolution_params.single_precision,
            epsilon=convolution_params.epsilon,
            nthreads=nthreads,
        )
        tod_det += fullconv.signal(ptg=pointings_det, alpha=hwp_angle)


def add_convolved_sky(
    tod,
    pointings,
    hwp_angle,
    sky_alms: Dict[str, np.ndarray],
    input_sky_names,
    beam_alms: Dict[str, np.ndarray],
    input_beam_names,
    convolution_params: Optional[BeamConvolutionParameters] = None,
    input_sky_alms_in_galactic: bool = True,
    nthreads: int = 0,
):
    """ """

    # just filled
    mueller = np.identity(4)

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

    for detector_idx in range(tod.shape[0]):
        if type(pointings) is np.ndarray:
            curr_pointings_det = pointings[detector_idx, :, :]
        else:
            curr_pointings_det, hwp_angle = pointings(detector_idx)

        if input_sky_alms_in_galactic:
            curr_pointings_det = rotate_coordinates_e2g(curr_pointings_det)

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

        add_convolved_sky_to_one_detector(
            tod_det=tod[detector_idx],
            sky_alms_det=sky_alms_det,
            beam_alms_det=beam_alms_det,
            mueller_matrix=mueller,
            pointings_det=curr_pointings_det,
            hwp_angle=hwp_angle,
            convolution_params=convolution_params,
            nthreads=nthreads,
        )


def add_convolved_sky_to_observations(
    observations: Union[Observation, List[Observation]],
    sky_alms: Dict[str, np.ndarray],  # at some point optional, taken from the obs
    beam_alms: Dict[str, np.ndarray],  # at some point optional, taken from the obs
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[HWP] = None,
    input_sky_alms_in_galactic: bool = True,
    convolution_params: Optional[BeamConvolutionParameters] = None,
    component: str = "tod",
    nthreads: int = 0,
):
    """Convolve sky maps with generic detector beams and add the resulting
    signal to TOD.

    Arguments
    ---------
    observations: Union[Observation, List[Observation]],
        List of Observation objects, containing detector names, pointings,
        and TOD data, to which the computed TOD are added.
    sky_alms: Dict[str, np.ndarray]
        sky a_lm. Typically only one set of sky a_lm is needed per detector frequency
    beam_alms: Dict[str, np.ndarray]
        beam a_lm. Usually one set of a_lm is needed for every detector.
    pointings: Union[np.ndarray, List[np.ndarray], None] = None
        detector pointing information
    hwp: Optional[HWP] = None
        the HWP information. If `None`, we assume traditional 4pi convolution.
    input_sky_alms_in_galactic: bool = True
        whether the input sky alms are in galactic coordinates.
    convolution_params: Optional[BeamConvolutionParameters]
        Parameters to tune the beam convolution. If the default is used, the
        code will try to find sensible numbers for these parameters. However,
        this should not be used in production code!
    component: str
        name of the TOD component to which the computed data shall be added
    nthreads: int
        number of threads to use in the convolution. The default (0) will use
        all the available CPUs. If you have a :class:`Simulation`
        object, a wise choice is to pass the field ``numba_num_of_threads``
        if it is nonzero, because the caller might have specified to use
        fewer cores than what is available.
    """

    if pointings is None:
        if isinstance(observations, Observation):
            obs_list = [observations]
            if hasattr(observations, "pointing_matrix"):
                ptg_list = [observations.pointing_matrix]
            else:
                ptg_list = [observations.get_pointings]
        else:
            obs_list = observations
            ptg_list = []
            for ob in observations:
                if hasattr(ob, "pointing_matrix"):
                    ptg_list.append(ob.pointing_matrix)
                else:
                    ptg_list.append(ob.get_pointings)
    else:
        if isinstance(observations, Observation):
            assert isinstance(pointings, np.ndarray), (
                "You must pass a list of observations *and* a list "
                "of pointing matrices to add_convolved_sky_to_observations"
            )
            obs_list = [observations]
            ptg_list = [pointings]
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to add_convolved_sky_to_observations, "
                "you must do the same for `pointings`"
            )
            assert len(observations) == len(pointings), (
                f"The list of observations has {len(observations)} elements, but "
                + f"the list of pointings has {len(pointings)} elements"
            )
            obs_list = observations
            ptg_list = pointings

    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        if isinstance(sky_alms, dict):
            if all(item in sky_alms.keys() for item in cur_obs.name):
                input_sky_names = cur_obs.name
            elif all(item in sky_alms.keys() for item in cur_obs.channel):
                input_sky_names = cur_obs.channel
            else:
                raise ValueError(
                    "The dictionary maps does not contain all the relevant "
                    "keys, please check the list of detectors and channels"
                )
            if "Coordinates" in sky_alms.keys():
                dict_input_sky_alms_in_galactic = (
                    sky_alms["Coordinates"] is CoordinateSystem.Galactic
                )
                if dict_input_sky_alms_in_galactic != input_sky_alms_in_galactic:
                    logging.warning(
                        "input_sky_alms_in_galactic variable in add_convolved_sky_to_observations"
                        " overwritten!"
                    )
                input_sky_alms_in_galactic = dict_input_sky_alms_in_galactic
        else:
            assert isinstance(sky_alms, np.ndarray), (
                "sky_alms must be either a dictionary contaning the keys for all the"
                + "channels/detectors or a 3×N_ℓm NumPy array"
            )
            input_sky_names = None

        if isinstance(beam_alms, dict):
            if all(item in beam_alms.keys() for item in cur_obs.name):
                input_beam_names = cur_obs.name
            elif all(item in beam_alms.keys() for item in cur_obs.channel):
                input_beam_names = cur_obs.channel
            else:
                raise ValueError(
                    "The dictionary beams does not contain all the relevant "
                    "keys, please check the list of detectors and channels"
                )
        else:
            assert isinstance(beam_alms, np.ndarray), (
                "beam_alms must be either a dictionary containing keys for all the "
                "channels/detectors or a 3×N_ℓm NumPy array"
            )
            input_beam_names = None

        if hwp is None:
            if hasattr(cur_obs, "hwp_angle"):
                hwp_angle = cur_obs.hwp_angle
            else:
                hwp_angle = None
        else:
            if type(cur_ptg) is np.ndarray:
                hwp_angle = get_hwp_angle(cur_obs, hwp)
            else:
                logging.warning(
                    "To use an external HWP object, you must pass a pre-calculated pointing, too"
                )

        add_convolved_sky(
            tod=getattr(cur_obs, component),
            pointings=cur_ptg,
            hwp_angle=hwp_angle,
            sky_alms=sky_alms,
            input_sky_names=input_sky_names,
            beam_alms=beam_alms,
            input_beam_names=input_beam_names,
            convolution_params=convolution_params,
            input_sky_alms_in_galactic=input_sky_alms_in_galactic,
            nthreads=nthreads,
        )
