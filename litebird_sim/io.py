# -*- encoding: utf-8 -*-

from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import astropy.time
import h5py
import numpy as np

from .detectors import DetectorInfo
from .mpi import MPI_COMM_WORLD
from .observations import Observation
from .simulations import Simulation


def write_one_observation(
    output_file: h5py.File, obs: Observation, tod_dtype, pointings_dtype
):
    tod_dataset = output_file.create_dataset("tod", data=obs.tod, dtype=tod_dtype)

    # Save pointing information only if it is available
    try:
        output_file.create_dataset(
            "pointings", data=obs.__getattribute__("pointings"), dtype=pointings_dtype
        )
        output_file.create_dataset("pixel_index", data=obs.__getattribute__("pixel"))
        output_file.create_dataset(
            "psi", data=obs.__getattribute__("psi"), dtype=pointings_dtype
        )
    except AttributeError:
        pass

    if isinstance(obs.start_time, astropy.time.Time):
        tod_dataset.attrs["start_time"] = obs.start_time.to_value(
            format="mjd", subfmt="bytes"
        )
        tod_dataset.attrs["mjd_time"] = True
    else:
        tod_dataset.attrs["start_time"] = obs.start_time
        tod_dataset.attrs["mjd_time"] = False

    tod_dataset.attrs["sampling_rate_hz"] = obs.sampling_rate_hz
    tod_dataset.attrs["mpi_rank"] = MPI_COMM_WORLD.rank
    tod_dataset.attrs["mpi_size"] = MPI_COMM_WORLD.size

    # This code assumes that the parameter `detectors` passed to Observation
    # was either an integer or a list of `DetectorInfo` objects, i.e., we
    # neglect the possibility that it was a list of dictionaries passing
    # custom keys that have no parallel in DetectorInfo, as it would
    # be too bothersome to understand which keys should be saved here.

    det_info = []
    detector_info_fields = [x.name for x in fields(DetectorInfo())]
    for det_idx in range(obs.n_detectors):
        new_detector = {}

        for attribute in detector_info_fields:
            try:
                if isinstance(obs.__getattribute__(attribute),float):
                    new_detector[attribute] = obs.__getattribute__(attribute)
                else:
                    if type(obs.__getattribute__(attribute)[det_idx]) is np.int64:
                        new_detector[attribute] = int(obs.__getattribute__(attribute)[det_idx])
                    if type(obs.__getattribute__(attribute)[det_idx]) is np.ndarray:
                        new_detector[attribute] = obs.__getattribute__(attribute)[det_idx].tolist()
                    else: 
                        new_detector[attribute] = obs.__getattribute__(attribute)[det_idx]
            except AttributeError:
                pass

        det_info.append(new_detector)

    # Now encode `det_info` in a JSON string and save it as an attribute
    tod_dataset.attrs["detectors"] = json.dumps(det_info)


def write_list_of_observations(
    obs: Union[Observation, List[Observation]],
    path: Union[str, Path],
    tod_dtype=np.float32,
    pointings_dtype=np.float32,
    file_name_mask: str = "tod_mpi{mpi_rank:04d}_{index:04d}.h5",
    custom_placeholders: Union[None, List[Dict[str, Any]]] = None,
    start_index: int = 0,
):
    """
    Save a list of observations in a set of HDF5 files

    This function takes one or more observations and saves the TODs in several
    HDF5 (each observation leads to *one* file), using `tod_dtype` and
    `pointings_dtype` as the default datatypes for the samples and the pointing
    angles.

    The name of the HDF5 files is built using the variable `file_name_mask`,
    which can contain placeholders in the form ``{name}``, where ``name`` can
    be one of the following keys:

    - ``mpi_rank``: the rank number of the MPI process that owns this
      observation (starting from zero)
    - ``mpi_size``: the number of running MPI processes
    - ``num_of_obs``: the number of observations
    - ``index``: the number of the current observation (see below)

    The ``index`` placeholder starts from zero, but this can be changed using
    the parameter `start_index`.

    You can provide other placeholders through `custom_dicts`, which must be
    a list of dictionaries. The number of elements in the list must be the
    same as the number of observations, and each dictionary will be used to
    determine the placeholders for the file name related to the corresponding
    observation. Here is an example::

        custom_dicts = [
            { "myvalue": "A" },
            { "myvalue", "B" },
        ]

        write_list_of_observations(
            obs=[obs1, obs2],  # Write two observations
            path=".",
            file_name_mask="tod_{myvalue}.h5",
            custom_dicts=custom_dicts,
        )

        # The two observations will be saved in
        # - tod_A.h5
        # - tod_B.h5
    """
    try:
        obs[0]
    except TypeError:
        obs = [obs]

    if not isinstance(path, Path):
        path = Path(path)

    # Iterate over all the observations and create one HDF5 file for each of them
    for obs_idx, cur_obs in enumerate(obs):
        params = {
            "mpi_rank": MPI_COMM_WORLD.rank,
            "mpi_size": MPI_COMM_WORLD.size,
            "num_of_obs": len(obs),
            "index": start_index + obs_idx,
        }

        # Merge the standard dictionary used to build the file name with any other
        # key in `custom_placeholders`. If `custom_placeholders` contains some
        # duplicated key, it will take the precedence over our default here.
        if custom_placeholders is not None:
            cur_placeholders = custom_placeholders[obs_idx]
            params = dict(params, **cur_placeholders)

        # Build the file name out of the template
        file_name = path / file_name_mask.format(**params)

        # Write the HDF5 file
        with h5py.File(file_name, "w") as output_file:
            write_one_observation(
                output_file=output_file,
                obs=cur_obs,
                tod_dtype=tod_dtype,
                pointings_dtype=pointings_dtype,
            )


def write_observations(
    sim: Simulation, subdir_name: Union[None, str] = "tod", *args, **kwargs
):
    """Write a set of observations as HDF5

    This function is a wrapper to :func:`.write_list_of_observations` that saves
    the observations associated with the simulation to a subdirectory within the
    output path of the simulation. The subdirectory is named `subdir_name`; if
    you want to avoid creating a subdirectory, just pass an empty string or None.

    This function only writes HDF5 for the observations that belong to the current
    MPI process. If you have distributed the observations over several processes,
    you must call this function on each MPI process.
    """

    if subdir_name:
        tod_path = sim.base_path / subdir_name
        # Ensure that the subdirectory exists
        tod_path.mkdir(exist_ok=True)
    else:
        tod_path = sim.base_path

    write_list_of_observations(obs=sim.observations, path=tod_path, *args, **kwargs)
