# -*- encoding: utf-8 -*-

from typing import List, Optional

from .detectors import InstrumentInfo
from .hwp import HWP
from .observations import Observation
from .pointings import PointingProvider
from .scanning import RotQuaternion


def prepare_pointings(
    observations: List[Observation],
    instrument: InstrumentInfo,
    spin2ecliptic_quats: RotQuaternion,
    hwp: Optional[HWP],
    store_full_pointings: bool = False,
) -> None:
    """Store the quaternions needed to compute pointings into a list of :class:`.Observation` objects

    This function computes the quaternions that convert the boresight direction
    of `instrument` into the Ecliptic reference frame. The `spin2ecliptic_quats`
    object must be an instance of the :class:`.RotQuaternion` class and can
    be created using the method :meth:`.ScanningStrategy.generate_spin2ecl_quaternions`.

    If `store_full_pointings` is ``True``, each :class:`.Observation` object in
    `observations` will have the full pointing matrix and the HWP angle stored
    as members of the object in the fields ``pointing_matrix`` and ``hwp_angle``.
    """

    bore2ecliptic_quats = spin2ecliptic_quats * instrument.bore2spin_quat
    pointing_provider = PointingProvider(
        bore2ecliptic_quats=bore2ecliptic_quats,
        hwp=hwp,
    )

    for cur_obs in observations:
        cur_obs.pointing_provider = pointing_provider

        if store_full_pointings:
            pointing_matrix, hwp_angle = cur_obs.get_pointings(detector_idx="all")
            cur_obs.pointing_matrix = pointing_matrix
            cur_obs.hwp_angle = hwp_angle
