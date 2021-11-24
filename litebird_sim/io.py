# -*- encoding: utf-8 -*-

from dataclasses import fields, asdict
import json
import logging as log
from pathlib import Path
import re
from typing import Any, Dict, List, Union, Optional

import astropy.time
import h5py
import numpy as np

from .compress import rle_compress, rle_decompress
from .detectors import DetectorInfo
from .mpi import MPI_COMM_WORLD
from .observations import Observation
from .simulations import Simulation

__NUMPY_INT_TYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
]

__NUMPY_FLOAT_TYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.float128,
]

__NUMPY_SCALAR_TYPES = __NUMPY_INT_TYPES + __NUMPY_FLOAT_TYPES


__OBSERVATION_FILE_NAME_MASK = "litebird_tod_mpi{mpi_rank:04d}_{index:04d}.h5"


__FLAGS_GROUP_NAME_REGEXP = re.compile("flags_([0-9]+)")


def write_one_observation(
    output_file: h5py.File, obs: Observation, tod_dtype, pointings_dtype
):
    tod_dataset = output_file.create_dataset("tod", data=obs.tod, dtype=tod_dtype)

    # Save pointing information only if it is available
    try:
        output_file.create_dataset(
            "pointings", data=obs.__getattribute__("pointings"), dtype=pointings_dtype
        )
    except (AttributeError, TypeError):
        pass

    try:
        output_file.create_dataset("pixel_index", data=obs.__getattribute__("pixel"))
    except (AttributeError, TypeError):
        pass

    try:
        output_file.create_dataset(
            "psi", data=obs.__getattribute__("psi"), dtype=pointings_dtype
        )
    except (AttributeError, TypeError):
        pass

    try:
        output_file.create_dataset(
            "global_flags",
            data=rle_compress(obs.__getattribute__("global_flags")),
        )
    except (AttributeError, TypeError):
        pass

    try:
        # We must separate the flags belonging to different detectors because they
        # might have different shapes
        for det_idx in range(obs.local_flags.shape[0]):
            flags = obs.__getattribute__("local_flags")
            compressed_flags = rle_compress(flags[det_idx, :])
            output_file.create_dataset(
                f"flags_{det_idx:04d}", data=compressed_flags, dtype=flags.dtype
            )
    except (AttributeError, TypeError):
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
    # We must use this ugly hack because Observation does not store DetectorInfo
    # classes but «spreads» their fields in the namespace of the class Observation.
    detector_info_fields = [x.name for x in fields(DetectorInfo())]
    for det_idx in range(obs.n_detectors):
        new_detector = {}

        for attribute in detector_info_fields:
            try:
                attr_value = obs.__getattribute__(attribute)
            except AttributeError:
                continue

            if type(attr_value) in (str, bool, int, float, dict):
                # Plain Python type
                new_detector[attribute] = attr_value
            elif type(attr_value) in __NUMPY_SCALAR_TYPES:
                # Convert a NumPy type into a Python type
                new_detector[attribute] = attr_value.item()
            else:
                # From now on, assume that this attribute is an array whose length
                # is the same as `obs.n_detectors`
                attr_value = attr_value[det_idx]
                if type(attr_value) in __NUMPY_SCALAR_TYPES:
                    new_detector[attribute] = attr_value.item()
                elif type(attr_value) is np.ndarray:
                    new_detector[attribute] = [x.item() for x in attr_value]
                else:
                    # Fallback: we assume this is a plain Python type
                    new_detector[attribute] = attr_value

        det_info.append(new_detector)

    # Now encode `det_info` in a JSON string and save it as an attribute
    tod_dataset.attrs["detectors"] = json.dumps(det_info)


def write_list_of_observations(
    obs: Union[Observation, List[Observation]],
    path: Union[str, Path],
    tod_dtype=np.float32,
    pointings_dtype=np.float32,
    file_name_mask: str = __OBSERVATION_FILE_NAME_MASK,
    custom_placeholders: Optional[List[Dict[str, Any]]] = None,
    start_index: int = 0,
) -> List[Path]:
    """
    Save a list of observations in a set of HDF5 files

    This function takes one or more observations and saves the TODs in several
    HDF5 (each observation leads to *one* file), using `tod_dtype` and
    `pointings_dtype` as the default datatypes for the samples and the pointing
    angles. The function returns a list of the file written (``pathlib.Path``
    objects).

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
    file_list = []
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

        file_list.append(file_name)

    return file_list


def write_observations(
    sim: Simulation, subdir_name: Union[None, str] = "tod", *args, **kwargs
) -> List[Path]:
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

    file_list = write_list_of_observations(
        obs=sim.observations, path=tod_path, *args, **kwargs
    )

    sim.append_to_report(
        """
## TOD files

{% if file_list %}
The following files containing Time-Ordered Data (TOD) have been written:

{% for file in file_list %}
- {{ file }}
{% endfor %}
{% else %}
No TOD files have been written to disk.
{% endif %}
""",
        file_list=file_list,
    )

    return file_list


def __find_flags(inpf, expected_num_of_dets: int, expected_num_of_samples: int):
    flags_matches = [__FLAGS_GROUP_NAME_REGEXP.fullmatch(x) for x in inpf]
    flags_matches = [x for x in flags_matches if x]
    if not flags_matches:
        return None

    if len(flags_matches) < expected_num_of_dets:
        log.warning(
            f"only {len(flags_matches)} flag datasets have been found in "
            + f"file {inpf.name}, but {expected_num_of_dets} were expected"
        )

    # We use np.zeros instead of np.empty because in this way we set to zero flags
    # for those detectors that do not have a "flags_NNNN" dataset in the HDF5 file
    flags = np.zeros(
        (expected_num_of_dets, expected_num_of_samples),
        dtype=inpf[flags_matches[0].string].dtype,
    )

    for match in flags_matches:
        det_idx = int(match.groups()[0])
        flags[det_idx, :] = rle_decompress(inpf[match.string][:])

    return flags


def read_one_observation(
    path: Union[str, Path],
    limit_mpi_rank=True,
    tod_dtype=np.float32,
    pointings_dtype=np.float32,
    read_pointings_if_present=True,
    read_pixidx_if_present=True,
    read_psi_if_present=True,
    read_global_flags_if_present=True,
    read_flags_if_present=True,
) -> Optional[Observation]:
    with h5py.File(str(path), "r") as inpf:
        assert "tod" in inpf
        hdf5_tod = inpf["tod"]

        if limit_mpi_rank:
            mpi_size = hdf5_tod.attrs["mpi_size"]
            assert mpi_size == MPI_COMM_WORLD.size, (
                '"{name}" was created using {orig_size} MPI processes, '
                + "but now {actual_size} are available"
            ).format(
                name=str(path), orig_size=mpi_size, actual_size=MPI_COMM_WORLD.size
            )
            if hdf5_tod.attrs["mpi_rank"] != MPI_COMM_WORLD.rank:
                # We are not supposed to load this observation in this MPI process
                return None

        if hdf5_tod.attrs["mjd_time"]:
            start_time = astropy.time.Time(hdf5_tod.attrs["start_time"], format="mjd")
        else:
            start_time = hdf5_tod.attrs["start_time"]

        detectors = [
            DetectorInfo.from_dict(x) for x in json.loads(hdf5_tod.attrs["detectors"])
        ]
        result = Observation(
            detectors=[asdict(d) for d in detectors],
            n_samples_global=hdf5_tod.shape[1],
            start_time_global=start_time,
            n_blocks_det=1,
            n_blocks_time=1,
            allocate_tod=True,
            sampling_rate_hz=hdf5_tod.attrs["sampling_rate_hz"],
            comm=None if limit_mpi_rank else MPI_COMM_WORLD,
            dtype_tod=hdf5_tod.dtype,
        )

        # Copy the TOD in the newly created observation
        assert result.tod.shape == hdf5_tod.shape
        result.tod[:] = hdf5_tod.astype(tod_dtype)[:]

        # Do the same for other optional datasets
        for attr, attr_type, should_read in [
            ("pointings", pointings_dtype, read_pointings_if_present),
            ("pixidx", np.int32, read_pixidx_if_present),
            ("psi", pointings_dtype, read_psi_if_present),
            ("global_flags", None, read_global_flags_if_present),
        ]:
            if (attr in inpf) and should_read:
                result.__setattr__(attr, inpf[attr].astype(attr_type)[:])

        # Checking if flags are present is trickier because there should be N
        # datasets, where N is the number of detectors
        if read_flags_if_present:
            flags = __find_flags(
                inpf,
                expected_num_of_dets=result.tod.shape[0],
                expected_num_of_samples=result.tod.shape[1],
            )
            if flags is not None:
                result.__setattr__("local_flags", flags)

    return result


def read_list_of_observations(
    path: Union[str, Path],
    tod_dtype=np.float32,
    pointings_dtype=np.float32,
    file_name_mask: str = __OBSERVATION_FILE_NAME_MASK,
    custom_placeholders: Optional[List[Dict[str, Any]]] = None,
    start_index: int = 0,
    limit_mpi_rank: bool = True,
) -> List[Observation]:

    path = Path(path)

    cur_file_index = start_index
    observations = []
    while True:
        params = {
            "mpi_rank": MPI_COMM_WORLD.rank,
            "mpi_size": MPI_COMM_WORLD.size,
            "index": cur_file_index,
        }

        # Merge the standard dictionary used to build the file name with any other
        # key in `custom_placeholders`. If `custom_placeholders` contains some
        # duplicated key, it will take the precedence over our default here.
        if custom_placeholders is not None:
            cur_placeholders = custom_placeholders[cur_file_index]
            params = dict(params, **cur_placeholders)

        cur_file_name = path / file_name_mask.format(**params)

        try:
            observations.append(
                read_one_observation(
                    cur_file_name,
                    limit_mpi_rank=limit_mpi_rank,
                    tod_dtype=tod_dtype,
                    pointings_dtype=pointings_dtype,
                )
            )
        except FileNotFoundError:
            print(f"{cur_file_name} with index {cur_file_index} not found")
            break

        cur_file_index += 1

    return observations


def read_observations(
    sim: Simulation,
    path: Union[str, Path],
    subdir_name: Union[None, str] = "tod",
    *args,
    **kwargs,
):
    obs = read_list_of_observations(path=path / subdir_name, *args, **kwargs)
    sim.observations = obs
