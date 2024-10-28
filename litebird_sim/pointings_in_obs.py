# -*- encoding: utf-8 -*-

from typing import List, Optional, Union

import numpy as np

from .detectors import InstrumentInfo
from .hwp import HWP
from .observations import Observation
from .pointings import PointingProvider
from .scanning import RotQuaternion


def prepare_pointings(
    observations: Union[Observation, List[Observation]],
    instrument: InstrumentInfo,
    spin2ecliptic_quats: RotQuaternion,
    hwp: Optional[HWP] = None,
) -> None:
    """Store the quaternions needed to compute pointings into a list of :class:`.Observation` objects

    This function computes the quaternions that convert the boresight direction
    of `instrument` into the Ecliptic reference frame. The `spin2ecliptic_quats`
    object must be an instance of the :class:`.RotQuaternion` class and can
    be created using the method :meth:`.ScanningStrategy.generate_spin2ecl_quaternions`.
    """

    bore2ecliptic_quats = spin2ecliptic_quats * instrument.bore2spin_quat
    pointing_provider = PointingProvider(
        bore2ecliptic_quats=bore2ecliptic_quats,
        hwp=hwp,
    )

    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    for cur_obs in obs_list:
        cur_obs.pointing_provider = pointing_provider


def precompute_pointings(
    observations: Union[Observation, List[Observation]],
    pointings_dtype=np.float32,
) -> None:
    """Precompute all the pointings for a set of observations

    Compute the full pointing matrix and the HWP angle for each :class:`.Observation`
    object in `obs_list` and store them in the fields ``pointing_matrix`` and ``hwp_angle``.
    The datatype for the pointings is specified by `pointings_dtype`.
    """

    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    for cur_obs in obs_list:
        assert "pointing_provider" in dir(cur_obs), (
            "you must call prepare_pointings() on a set of observations "
            "before calling precompute_pointings()"
        )

        pointing_matrix, hwp_angle = cur_obs.get_pointings(
            detector_idx="all", pointings_dtype=pointings_dtype
        )
        cur_obs.pointing_matrix = pointing_matrix
        cur_obs.hwp_angle = hwp_angle
