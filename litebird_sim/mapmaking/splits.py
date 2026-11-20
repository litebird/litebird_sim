"""Detector and time splits of the map-makers

Each split name is bound to the function that builds its mask, so that the
names accepted by :func:`.check_valid_splits` and those that the map-makers
can build are the same by construction. The map-makers validate the splits
every time they build a mask, also when they are called without a
:class:`.Simulation`.
"""

from collections.abc import Callable

import astropy.time
import numpy as np
import numpy.typing as npt

from litebird_sim.observations import Observation

#: Length of a year and of a survey (half a year) in the time splits, in seconds
T_YEAR_S = 365 * 24 * 3600
T_SURVEY_S = T_YEAR_S / 2

MAX_YEARS = 3
MAX_SURVEYS = 6

#: Wafers of the LiteBIRD focal plane, accepted as ``"wafer<name>"``
LITEBIRD_WAFERS = (
    [f"L{i:02d}" for i in range(8)]
    + [f"M{i:02d}" for i in range(7)]
    + [f"H{i:02d}" for i in range(3)]
)


def _get_initial_time(obs: Observation) -> float:
    if isinstance(obs.start_time_observation, astropy.time.Time):
        return obs.start_time_observation.cxcsec
    return obs.start_time_observation


def _get_end_time(obs: Observation) -> float:
    if isinstance(obs.end_time_observation, astropy.time.Time):
        return obs.end_time_observation.cxcsec
    return obs.end_time_observation


def _full(obs: Observation) -> npt.NDArray:
    # A read-only view needs no memory: a real array would take one byte per
    # sample for *all* the observations at once (~7 GB per process for 3 years
    # of one 75 Hz detector)
    return np.broadcast_to(np.True_, obs.n_samples_process)


def _samples(selection: Callable[[int], slice]) -> Callable:
    def build(obs: Observation) -> npt.NDArray:
        mask = np.zeros(obs.n_samples_process, dtype=bool)
        mask[selection(obs.n_samples_process)] = True
        return mask

    return build


def _period(index: int, length_s: float) -> Callable:
    def build(obs: Observation) -> npt.NDArray:
        times = obs.get_times() - _get_initial_time(obs)
        return (times >= (index - 1) * length_s) & (times < index * length_s)

    return build


#: Time split name → function returning the mask of one observation
TIME_SPLITS: dict[str, Callable[[Observation], npt.NDArray]] = {
    "full": _full,
    "odd": _samples(lambda n: slice(0, None, 2)),
    "even": _samples(lambda n: slice(1, None, 2)),
    "first_half": _samples(lambda n: slice(0, n // 2)),
    "second_half": _samples(lambda n: slice(n // 2, None)),
    **{f"year{i}": _period(i, T_YEAR_S) for i in range(1, MAX_YEARS + 1)},
    **{f"survey{i}": _period(i, T_SURVEY_S) for i in range(1, MAX_SURVEYS + 1)},
}

#: Valid detector split names
DETECTOR_SPLITS = ["full"] + [f"wafer{w}" for w in LITEBIRD_WAFERS]


def _validate_time_split(time_split: str, obs_list: list[Observation]) -> None:
    if time_split not in TIME_SPLITS:
        raise ValueError(
            f"Unknown time split '{time_split}', valid ones are {list(TIME_SPLITS)}"
        )
    for prefix, length_s in (("year", T_YEAR_S), ("survey", T_SURVEY_S)):
        if not time_split.startswith(prefix):
            continue
        index = int(time_split.removeprefix(prefix))
        for cur_obs in obs_list:
            duration = round(_get_end_time(cur_obs) - _get_initial_time(cur_obs), 0)
            if index > duration // length_s:
                raise AssertionError(
                    f"Time split '{time_split}' not possible for an observation "
                    f"lasting {round(duration / T_YEAR_S, 2)} years"
                )


def _validate_detector_split(detector_split: str, obs_list: list[Observation]) -> None:
    if detector_split not in DETECTOR_SPLITS:
        raise ValueError(
            f"Unknown detector split '{detector_split}', valid ones are "
            f"{DETECTOR_SPLITS}"
        )
    if detector_split == "full":
        return
    wafer = detector_split.removeprefix("wafer")
    for cur_obs in obs_list:
        if cur_obs.wafer is None or wafer not in cur_obs.wafer:
            raise AssertionError(
                f"The requested wafer '{wafer}' is not in the observation, "
                f"which has the wafers {cur_obs.wafer}"
            )


def _build_mask_time_split(
    time_split: str, obs_list: list[Observation]
) -> list[npt.NDArray]:
    """One Boolean mask ``(n_samples,)`` per observation, True for the samples
    in `time_split`. Raise if the split is unknown or not possible."""
    _validate_time_split(time_split, obs_list)
    build = TIME_SPLITS[time_split]
    return [build(cur_obs) for cur_obs in obs_list]


def _build_mask_detector_split(
    detector_split: str, obs_list: list[Observation]
) -> list[npt.NDArray]:
    """One Boolean mask ``(n_detectors,)`` per observation, True for the
    detectors in `detector_split`. Raise if the split is unknown or not
    possible."""
    _validate_detector_split(detector_split, obs_list)
    if detector_split == "full":
        return [np.ones(cur_obs.n_detectors, dtype=bool) for cur_obs in obs_list]
    wafer = detector_split.removeprefix("wafer")
    return [np.asarray(cur_obs.wafer) == wafer for cur_obs in obs_list]


def check_valid_splits(
    observations: Observation | list[Observation],
    detector_splits: str | list[str] = "full",
    time_splits: str | list[str] = "full",
):
    """Check if the splits are valid

    For each observation in the list, check if the detector and time splits
    are valid.
    In particular, the compatibility between the detectors in each observation
    and the desired split in detector domain is checked. On the other hand, this
    assess whether the desired time split fits inside the duration of the
    observation (when this applies).
    If the splits are not compatible with the input data, an error is raised:
    ``ValueError`` for an unknown split, ``AssertionError`` for a split that
    the observations cannot satisfy.

    Args:
        observations (list of :class:`Observations`): observations to be mapped.
            They are required to have the following attributes as arrays

            * `pointings`: the pointing information (in radians) for each tod
               sample. It must be a tensor with shape ``(N_d, N_t, 3)``,
               with ``N_d`` number of detectors and ``N_t`` number of
               samples in the TOD.
            * any attribute listed in `components` (by default, `tod`) and
              containing the TOD(s) to be binned together.

            If the observations are distributed over some communicator(s), they
            must share the same group processes.
            If pointings and psi are not included in the observations, they can
            be provided through an array (or a list of arrays) of dimension
            (Ndetectors x Nsamples x 3), containing theta, phi and psi
        detector_splits (str | list[str], optional): detector-domain splits
            used to produce maps.

            * "full": every detector in the observation will be used;
            * "waferXXX": the mapmaking will be performed on the intersection
                of the detectors specified in the input and the detectors specified
                in the detector_split.
                The wafer must be specified in the format "waferXXX". The valid values
                for "XXX" are all the 3-digits strings corresponding to the wafers
                in the LITEBIRD focal plane (e.g. L00, M01, H02).

        time_splits (str | list[str], optional): time-domain splits
            used to produce maps. This defaults to "full" indicating that every
            sample in the observation will be used. In addition, the user can specify
            a string, or a list of strings, to indicate a subsample of the observation
            to be used:

            * "full": every sample in the observation will be used;
            * "first_half" and/or "second_half": the first and/or second half of the
                observation will be used;
            * "odd" and/or "even": the odd and/or even samples in the observation
                will be used;
            * "yearX": the samples in the observation will be
                used according to the year they belong to (relative to the
                starting time). The valid values for "X" are ["1", "2", "3"].
            * "surveyX": the samples in the observation will be used according
                to the requested survey. In this context, a survey is taken to
                be complete in 6 months, thus the valid values for "X" are
                ["1", "2", "3", "4", "5", "6"].

    """
    if isinstance(observations, Observation):
        observations = [observations]
    if isinstance(detector_splits, str):
        detector_splits = [detector_splits]
    if isinstance(time_splits, str):
        time_splits = [time_splits]

    for detector_split in detector_splits:
        _validate_detector_split(detector_split, observations)
    for time_split in time_splits:
        _validate_time_split(time_split, observations)
