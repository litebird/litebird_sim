# -*- encoding: utf-8 -*-

from .version import __version__
from time import perf_counter
from typing import Any


class TimeProfiler:
    """A context manager to profile the time spent by the code

    This class implements a context manager that uses the performance
    counter provided by the system to compute the time spent within
    a ``with`` block.
    """

    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.parameters = dict(kwargs)
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, typ, value, traceback):
        self.end = perf_counter()

    def valid(self):
        return self.start and self.end

    def elapsed_time_s(self):
        return self.end - self.start


def profile_list_to_speedscope(profile_list: list[TimeProfiler]) -> dict[Any, Any]:
    """
    Convert a list of :class:`.TimeProfiler` objects into a Speedscope file

    This function takes a list of :class:`.TimeProfiler` objects and creates a
    dictionary that conforms to the specifications of the `Speedscope webapp
    <https://www.speedscope.app/>`_. The dictionary should be saved to a JSON
    file and then opened within Speedscope.
    """

    # Generate a set of all the unique names in the list of profiles
    frame_names = sorted(set([x.name for x in profile_list]))

    # This dictionary associates a frame name to a unique index starting from zero
    frame_name_to_index = dict(
        (name, index) for (index, name) in enumerate(frame_names)
    )

    try:
        start_time = min([prof.start for prof in profile_list if prof.valid()])
    except ValueError:
        # This happens if no profile data was available
        start_time = 0.0

    try:
        end_time = min([prof.end for prof in profile_list if prof.valid()])
    except ValueError:
        # This happens if no profile data was available
        end_time = 0.0

    events = []  # type: list[dict[Any, Any]]

    for prof in profile_list:
        if not prof.valid():
            continue

        cur_index = frame_name_to_index[prof.name]
        events.append({"type": "O", "frame": cur_index, "at": prof.start})
        events.append({"type": "C", "frame": cur_index, "at": prof.end})

    result = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "exporter": f"litebird_sim@{__version__}",
        "shared": {
            "frames": [{"name": name} for name in frame_names],
        },
        "profiles": [
            {
                "type": "evented",
                "unit": "seconds",
                "startValue": start_time,
                "endValue": end_time,
                "events": sorted(events, key=lambda e: e["at"]),
            },
        ],
    }

    return result
