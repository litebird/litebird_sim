# -*- encoding: utf-8 -*-

from collections import namedtuple
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
from .mpi import MPI_ENABLED, MPI_COMM_WORLD
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


__OBSERVATION_FILE_NAME_MASK = "litebird_tod{global_index:04d}.h5"


__FLAGS_GROUP_NAME_REGEXP = re.compile("flags_([0-9]+)")


def write_one_observation(
    output_file: h5py.File,
    obs: Observation,
    tod_dtype,
    pointings_dtype,
    global_index: int,
    local_index: int,
    tod_fields: List[str] = ["tod"],
):
    """Write one :class:`Observation` object in a HDF5 file.

    This is a low-level function that stores a TOD in a HDF5 file. You should usually
    use more high-level functions that are able to write several observations at once,
    like :func:`.write_list_of_observations` and :func:`.write_observations`.

    The output file is specified by `output_file` and should be opened for writing; the
    observation to be written is passed through the `obs` parameter. The data type to
    use for writing TODs and pointings is specified in the `tod_dtype` and
    `pointings_dtype` (it can either be a NumPy type like ``numpy.float64` or a
    string). The `global_index` and `local_index` parameters are two integers that are
    used by high-level functions like :func:`.write_observations` to understand how to
    read several HDF5 files at once; if you do not need them, you can pass 0 to both.
    """

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

    # Now encode `det_info` in a JSON string that will be saved later as an attribute
    detectors_json = json.dumps(det_info)

    # Write all the TOD timelines in the HDF5 file, in separate datasets
    for field_name in tod_fields:
        cur_dataset = output_file.create_dataset(
            field_name, data=obs.__getattribute__(field_name), dtype=tod_dtype
        )
        if isinstance(obs.start_time, astropy.time.Time):
            cur_dataset.attrs["start_time"] = obs.start_time.to_value(
                format="mjd", subfmt="bytes"
            )
            cur_dataset.attrs["mjd_time"] = True
        else:
            cur_dataset.attrs["start_time"] = obs.start_time
            cur_dataset.attrs["mjd_time"] = False

        cur_dataset.attrs["sampling_rate_hz"] = obs.sampling_rate_hz
        cur_dataset.attrs["detectors"] = detectors_json

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

    output_file.attrs["mpi_rank"] = MPI_COMM_WORLD.rank
    output_file.attrs["mpi_size"] = MPI_COMM_WORLD.size
    output_file.attrs["global_index"] = global_index
    output_file.attrs["local_index"] = local_index


def _compute_global_start_index(
    num_of_obs: int, start_index: int, collective_mpi_call: bool
) -> int:
    global_start_index = start_index
    if MPI_ENABLED and collective_mpi_call:
        # Count how many observations are kept in the MPI processes with lower rank
        # than this one.
        num_of_obs = np.asarray(MPI_COMM_WORLD.allgather(num_of_obs))
        global_start_index += np.sum(num_of_obs[0 : MPI_COMM_WORLD.rank])

    return global_start_index


def write_list_of_observations(
    obs: Union[Observation, List[Observation]],
    path: Union[str, Path],
    tod_dtype=np.float32,
    pointings_dtype=np.float32,
    file_name_mask: str = __OBSERVATION_FILE_NAME_MASK,
    custom_placeholders: Optional[List[Dict[str, Any]]] = None,
    start_index: int = 0,
    collective_mpi_call: bool = True,
    tod_fields: List[str] = ["tod"],
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
    - ``global_index``: an unique increasing index identifying this observation
      among all the observations used by MPI processes (see below)
    - ``local_index``: the number of the current observation within the current
      MPI process (see below)

    You can provide other placeholders through `custom_placeholders`, which must be
    a list of dictionaries. The number of elements in the list must be the
    same as the number of observations, and each dictionary will be used to
    determine the placeholders for the file name related to the corresponding
    observation. Here is an example::

        custom_dicts = [
            { "myvalue": "A" },
            { "myvalue": "B" },
        ]

        write_list_of_observations(
            obs=[obs1, obs2],  # Write two observations
            path=".",
            file_name_mask="tod_{myvalue}.h5",
            custom_placeholders=custom_dicts,
        )

        # The two observations will be saved in
        # - tod_A.h5
        # - tod_B.h5

    If the parameter `collective_mpi_call` is ``True`` and MPI is enabled
    (see ``litebird_sim.MPI_ENABLED``), the code assumes that this function
    was called at the same time by *all* the MPI processes that are currently
    running. This is the most typical case, i.e., you have several MPI
    processes and want that each of them save their observations in HDF5 files.
    Pass ``collective_mpi_call=False`` only if you are calling this function
    on some of the MPI processes. Here is an example::

        if lbs.MPI_COMM_WORLD.rank == 0:
            # Do this only for the first MPI process
            write_list_of_observations(
                ...,
                collective_mpi_call=False,
            )

    In the example, ``collective_mpi_call=False`` signals that not every MPI
    process is writing their observations to disk.

    The ``local_index`` and ``global_index`` placeholders used in the template
    file name start from zero, but this can be changed using the parameter
    `start_index`.

    If observations contain more than one timeline in separate fields
    (e.g., foregrounds, dipole, noise…), you can specify the names of the
    fields using the parameter ``tod_fields`` (list of strings), which by
    default will only save `Observation.tod`.

    """
    try:
        obs[0]
    except TypeError:
        obs = [obs]
    except IndexError:
        # Empty list
        # We do not want to return here, as we still need to participate to
        # the call to _compute_global_start_index below
        obs = []  # type: List[Observation]

    if not isinstance(path, Path):
        path = Path(path)

    global_start_index = _compute_global_start_index(
        num_of_obs=len(obs),
        start_index=start_index,
        collective_mpi_call=collective_mpi_call,
    )

    # Iterate over all the observations and create one HDF5 file for each of them
    file_list = []
    for obs_idx, cur_obs in enumerate(obs):
        params = {
            "mpi_rank": MPI_COMM_WORLD.rank,
            "mpi_size": MPI_COMM_WORLD.size,
            "num_of_obs": len(obs),
            "global_index": global_start_index + obs_idx,
            "local_index": start_index + obs_idx,
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
                global_index=params["global_index"],
                local_index=params["local_index"],
                tod_fields=tod_fields,
            )

        file_list.append(file_name)

    return file_list


def write_observations(
    sim: Simulation,
    subdir_name: Union[None, str] = "tod",
    include_in_report: bool = True,
    *args,
    **kwargs,
) -> List[Path]:
    """Write a set of observations as HDF5

    This function is a wrapper to :func:`.write_list_of_observations` that saves
    the observations associated with the simulation to a subdirectory within the
    output path of the simulation. The subdirectory is named `subdir_name`; if
    you want to avoid creating a subdirectory, just pass an empty string or None.

    This function only writes HDF5 for the observations that belong to the current
    MPI process. If you have distributed the observations over several processes,
    you must call this function on each MPI process.

    For a full explanation of the available parameters, see the documentation for
    :func:`.write_list_of_observations`.
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

    if include_in_report:
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
    read_local_flags_if_present=True,
    tod_fields: List[str] = ["tod"],
) -> Optional[Observation]:
    """Read one :class:`.Observation` object from a HDF5 file.

    This is a low-level function that is wrapped by :func:`.read_list_of_observations`
    and :func:`.read_observations`. It returns a :class:`.Observation` object filled
    with the data read from the HDF5 file pointed by `path`.

    If `limit_mpi_rank` is ``True`` (the default), the function makes sure that the
    rank of the MPI process reading this file is the same as the rank of the process
    that originally wrote it.

    The flags `tod_dtype` and `pointings_dtype` permit to override the data type of
    TOD samples and pointing angles used in the HDF5 file.

    The parameters `read_pointings_if_present`, `read_pixidx_if_present`,
    `read_psi_if_present`, `read_global_flags_if_present`, and
    `read_local_flags_if_present` permit to avoid loading some parts of the HDF5 if
    they are not needed.

    The function returns a :class:`.Observation`, or ``Nothing`` if the HDF5 file
    was ill-formed.
    """

    assert len(tod_fields) > 0

    with h5py.File(str(path), "r") as inpf:
        if limit_mpi_rank:
            mpi_size = inpf.attrs["mpi_size"]
            assert mpi_size == MPI_COMM_WORLD.size, (
                '"{name}" was created using {orig_size} MPI processes, '
                + "but now {actual_size} are available"
            ).format(
                name=str(path), orig_size=mpi_size, actual_size=MPI_COMM_WORLD.size
            )
            if inpf.attrs["mpi_rank"] != MPI_COMM_WORLD.rank:
                # We are not supposed to load this observation in this MPI process
                return None

        # Load the TOD(s)
        result = None  # Optional[Observation]

        for cur_field in tod_fields:
            assert cur_field in inpf
            hdf5_tod = inpf[cur_field]

            if hdf5_tod.attrs["mjd_time"]:
                start_time = astropy.time.Time(
                    hdf5_tod.attrs["start_time"], format="mjd"
                )
            else:
                start_time = hdf5_tod.attrs["start_time"]

            if result is None:
                detectors = [
                    DetectorInfo.from_dict(x)
                    for x in json.loads(hdf5_tod.attrs["detectors"])
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
                result.__setattr__(cur_field, hdf5_tod.astype(tod_dtype)[:])
            else:
                # All the fields must conform to the same shape as `Observation.tod`
                assert result.tod.shape == hdf5_tod.shape
                result.__setattr__(cur_field, hdf5_tod.astype(tod_dtype)[:])

        # If we arrive here, we must have read at least one TOD
        assert result is not None

        # If it is required, read other optional datasets
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
        if read_local_flags_if_present:
            flags = __find_flags(
                inpf,
                expected_num_of_dets=result.tod.shape[0],
                expected_num_of_samples=result.tod.shape[1],
            )
            if flags is not None:
                result.__setattr__("local_flags", flags)

    return result


_FileEntry = namedtuple(
    "_FileEntry", ["path", "mpi_size", "mpi_rank", "global_index", "local_index"]
)


def _build_file_entry_table(file_name_list: List[Union[str, Path]]) -> List[_FileEntry]:
    file_entries = []  # type: List[_FileEntry]
    for cur_file_name in file_name_list:
        with h5py.File(cur_file_name, "r") as inpf:
            try:
                file_entries.append(
                    _FileEntry(
                        path=Path(cur_file_name),
                        mpi_size=inpf.attrs.get("mpi_size", 1),
                        mpi_rank=inpf.attrs.get("mpi_rank", 0),
                        global_index=inpf.attrs["global_index"],
                        local_index=inpf.attrs["local_index"],
                    )
                )
            except KeyError as e:
                print(f"List of keys in {cur_file_name}:", list(inpf.attrs.keys()))
                raise RuntimeError(f"malformed TOD file {cur_file_name}: {e}")
    return file_entries


def read_list_of_observations(
    file_name_list: List[Union[str, Path]],
    tod_dtype=np.float32,
    pointings_dtype=np.float32,
    limit_mpi_rank: bool = True,
    tod_fields: List[str] = ["tod"],
) -> List[Observation]:
    """Read a list of HDF5 files containing TODs and return a list of observations

    The function reads all the HDF5 files listed in `file_name_list` (either a list of
    strings or ``pathlib.Path`` objects) and assigns them to the various MPI processes
    that are currently running, provided that `limit_mpi_rank` is ``True``; otherwise,
    all the files are read by the current process and returned in a list.

    When using MPI, the observations are distributed among the MPI processes using the
    same layout that was used to save them; this means that you are forced to use the
    same number of processes you used when saving the files. This number is saved in
    the attribute ``mpi_size`` in each of the HDF5 files.

    If the HDF5 file contains more than one TOD, e.g., foregrounds, dipole, noise…,
    you can specify which datasets to load with ``tod_fields``, that by default is
    equal to ``["tod"]``. Each dataset will be initialized as a member field of
    the :class:`.Observation` class, like ``Observation.tod``.
    """

    observations = []

    # When running several MPI processes, make just one of them read the HDF5 metadata,
    # otherwise we put too much burden on the storage filesystem
    if MPI_ENABLED:
        file_entries = (
            _build_file_entry_table(file_name_list)
            if (MPI_COMM_WORLD.rank == 0)
            else None
        )
        file_entries = MPI_COMM_WORLD.bcast(file_entries, root=0)
    else:
        file_entries = _build_file_entry_table(file_name_list)

    # Decide which files should be read by this process
    if limit_mpi_rank and MPI_ENABLED:
        file_entries = [x for x in file_entries if x.mpi_rank == MPI_COMM_WORLD.rank]

    for cur_file_entry in file_entries:
        observations.append(
            read_one_observation(
                cur_file_entry.path,
                limit_mpi_rank=limit_mpi_rank,
                tod_dtype=tod_dtype,
                pointings_dtype=pointings_dtype,
                tod_fields=tod_fields,
            )
        )

    return observations


def read_observations(
    sim: Simulation,
    path: Union[str, Path] = None,
    subdir_name: Union[None, str] = "tod",
    *args,
    **kwargs,
):
    """Read a list of observations from a set of files in a simulation

    This function is a wrapper around the function :func:`.read_list_of_observations`.
    It reads all the HDF5 files that are present in the directory whose name is
    `subdir_name` and is a child of `path`, and it stores them in the
    :class:`.Simulation` object `sim`.

    If `path` is not specified, the default is to use the value of ``sim.base_path``;
    this is meaningful if you are trying to read back HDF5 files that have been saved
    earlier in the same session.
    """
    if path is None:
        path = sim.base_path

    obs = read_list_of_observations(
        file_name_list=list((path / subdir_name).glob("**/*.h5")), *args, **kwargs
    )
    sim.observations = obs
