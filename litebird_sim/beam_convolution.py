# -*- encoding: utf-8 -*-

import numpy as np
import ducc0
from typing import Union, List, Dict, Optional
from .observations import Observation
from .hwp import HWP
from .pointings import get_hwp_angle
from .coordinates import rotate_coordinates_e2g, CoordinateSystem
from .mueller_convolver import MuellerConvolver
import logging


def add_convolved_sky_to_one_detector(
    tod_det,
    sky_alms_det,
    beam_alms_det,
    mueller_matrix,
    pointings_det,
    hwp_angle,
    convolution_params,  # convolution_params: XXX = YYY,
):
    """ """

    # global variable?
    nthreads = 0

    if (
        hwp_angle is None
    ):  # we cannot simulate HWP, so let's use classic 4pi convolution
        inter = ducc0.totalconvolve.Interpolator(
            sky_alms_det,
            beam_alms_det,
            separate=False,
            lmax=convolution_params.lmax,
            kmax=convolution_params.kmax,
            epsilon=convolution_params.epsilon,
            nthreads=nthreads,
        )
        tod_det += inter.interpol(pointings_det)
    else:
        fullconv = MuellerConvolver(
            convolution_params.lmax,
            convolution_params.kmax,
            sky_alms_det,
            beam_alms_det,
            mueller_matrix,
            single_precision=convolution_params.single_precision,
            epsilon=convolution_params.epsilon,
            nthreads=nthreads,
        )
        tod_det += fullconv.signal(pointings_det, hwp_angle)


def add_convolved_sky(
    tod,
    pointings,
    hwp_angle,
    sky_alms: Dict[str, np.ndarray],
    input_sky_names,
    beam_alms: Dict[str, np.ndarray],
    input_beam_names,
    convolution_params,  # convolution_params: XXX = YYY,
    input_sky_alms_in_galactic: bool = True,
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

        if hwp_angle is None:
            hwp_angle = 0

        if input_sky_alms_in_galactic:
            curr_pointings_det = rotate_coordinates_e2g(curr_pointings_det)

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
        )


def add_convolved_sky_to_observations(
    observations: Union[Observation, List[Observation]],
    sky_alms: Dict[str, np.ndarray],  # at some point optional
    beam_alms: Dict[str, np.ndarray],  # at some point optional
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[HWP] = None,
    input_sky_alms_in_galactic: bool = True,
    convolution_params=None,  # convolution_params: XXX = YYY,
    component: str = "tod",
):
    """Convolve sky maps with generic detector beams and add the resulting
    signal to TOD.

    Arguments
    ---------
    obs_list: list[Observation],
        List of Observation objects, containing detector names, pointings,
        and TOD data, to which the computed TOD are added.
    slm_dictionary:  dict[str, any]
        sky a_lm. Typically only one set of sky a_lm is needed per detector frequency
    blm_dictionary: dict[str, any]
        beam a_lm. Usually one set of a_lm is needed for every detector.
    det2slm: dict[str, str]
        converts detector name to a key for `slm_dictionary`
    det2slm: dict[str, str]
        converts detector name to a key for `blm_dictionary`
    component: str
        name of the TOD component to which the computed data shall be added
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
                + "of pointing matrices to add_convolved_sky_to_observations"
            )
            obs_list = [observations]
            ptg_list = [pointings]
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to add_convolved_sky_to_observations, "
                + "you must do the same for `pointings`"
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
                    "The dictionary maps does not contain all the relevant"
                    + "keys, please check the list of detectors and channels"
                )
            if "Coordinates" in sky_alms.keys():
                dict_input_sky_alms_in_galactic = (
                    sky_alms["Coordinates"] is CoordinateSystem.Galactic
                )
                if dict_input_sky_alms_in_galactic != input_sky_alms_in_galactic:
                    logging.warning(
                        "input_sky_alms_in_galactic variable in add_convolved_sky_to_observations"
                        + " overwritten!"
                    )
                input_sky_alms_in_galactic = dict_input_sky_alms_in_galactic
        else:
            assert isinstance(sky_alms, np.ndarray), (
                "sky_alms must either a dictionary contaning keys for all the"
                + "channels/detectors, or be a numpy array of dim (3 x Nlm)"
            )
            input_sky_names = None

        if isinstance(beam_alms, dict):
            if all(item in beam_alms.keys() for item in cur_obs.name):
                input_beam_names = cur_obs.name
            elif all(item in beam_alms.keys() for item in cur_obs.channel):
                input_beam_names = cur_obs.channel
            else:
                raise ValueError(
                    "The dictionary beams does not contain all the relevant"
                    + "keys, please check the list of detectors and channels"
                )
        else:
            assert isinstance(beam_alms, np.ndarray), (
                "beam_alms must either a dictionary contaning keys for all the"
                + "channels/detectors, or be a numpy array of dim (3 x Nlm)"
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
                    "For using an external HWP object also pass a pre-calculated pointing"
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
        )

    # These need to be provided by the user. They could be simply scalars, or
    # they could vary from detector to detector, not sure yet which is most useful.
    #    lmax, kmax = 1000, 5

    # find all involved detector names


#    detnames = set()
#    for obs in obs_list:
#        for det in obs.name:
#            detnames.add(det)


#    for cur_det in detnames:
# Set up the interpolator for this particular detector
#        slm = slm_dictionary[det2slm[cur_det]]
#        blm = blm_dictionary[det2blm[cur_det]]
#        interp = ducc0.totalconvolve.Interpolator(sky=slm, beam=blm,
#            separate=False, lmax=lmax, kmax=kmax, epsilon=1e-5, nthreads=1)

# Now go through all the pointings for this detector
# It might be advantageous to concatenate several chunks of observations
# together - this can make interpolation a bit more efficient.
# For now, let's just go through the chunks individually ...

#        for cur_obs in obs_list:
#            det_idx = list(cur_obs.name).index(cur_det)
#            ptg = cur_obs.pointings[det_idx]
#            psi = cur_obs.psi[det_idx]
#            # Ducc requires pointing information as a single array with shape
#            # (nptg, 3)
#            ptgnew = np.empty((ptg.shape[0], 3), dtype=ptg.dtype)
#            ptgnew[:, 0:2] = ptg
#            ptgnew[:, 2] = psi

# Get a reference to the TOD to which we should add our signal
#    cur_tod = getattr(cur_obs, component)[det_idx]

# Compute our signal and add it
#    cur_tod += interp.interpol(ptgnew)
