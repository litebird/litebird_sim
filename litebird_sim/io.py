# -*- encoding: utf-8 -*-

import json
import logging as log
import re
from collections import namedtuple
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import astropy.time
import h5py
import numpy as np

from .compress import rle_compress, rle_decompress
from .detectors import DetectorInfo
from .hwp import read_hwp_from_hdf5
from .mpi import MPI_ENABLED, MPI_COMM_WORLD
from .observations import Observation, TodDescription
from .pointings import PointingProvider
from .scanning import RotQuaternion

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
]

if "float128" in dir(np):
    __NUMPY_FLOAT_TYPES.append(np.float128)

__NUMPY_SCALAR_TYPES = __NUMPY_INT_TYPES + __NUMPY_FLOAT_TYPES


__OBSERVATION_FILE_NAME_MASK = "litebird_tod{global_index:04d}.h5"


__FLAGS_GROUP_NAME_REGEXP = re.compile("flags_([0-9]+)")


class DetectorJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, RotQuaternion):
            result = {
                "start_time": o.start_time
                if isinstance(o.start_time, float) or o.start_time is None
                else str(o.start_time),
                "sampling_rate_hz": o.sampling_rate_hz,
            }

            if o.quats.nbytes < 1024:
                # Only save the quaternions if they do not occupy too much memory
                # (in this case, the information is duplicated because these
                # quaternions appear both in the metadata and in a dataset within
                # the HDF5 file, but the waste of space is minimal)
                result["quats"] = o.quats.tolist()
            else:
                # Just save the shape of the matrix; for HDF5 files, the full
                # matrix will be saved nevertheless
                result["quaternion_matrix_shape"] = list(o.quats.shape)

            return result

        return super().default(o)


def write_rot_quaternion_to_hdf5(
    output_file: h5py.File,
    rot_matrix: RotQuaternion,
    field_name: str,
    compression: Optional[str],
) -> h5py.Dataset:
    new_dataset = output_file.create_dataset(
        field_name,
        data=rot_matrix.quats,
        dtype=np.float64,
        compression=compression,
    )

    if rot_matrix.start_time is not None:
        if isinstance(rot_matrix.start_time, astropy.time.Time):
            start_time = str(rot_matrix.start_time)
        else:
            start_time = float(rot_matrix.start_time)

        new_dataset.attrs["start_time"] = start_time

    if rot_matrix.sampling_rate_hz is not None:
        new_dataset.attrs["sampling_rate_hz"] = float(rot_matrix.sampling_rate_hz)

    return new_dataset


def read_rot_quaternion_from_hdf5(
    input_file: h5py.File,
    field_name: str,
) -> RotQuaternion:
    dataset = input_file[field_name]

    start_time = dataset.attrs.get("start_time", None)
    if isinstance(start_time, str):
        start_time = astropy.time.Time(start_time)

    sampling_rate_hz = dataset.attrs.get("sampling_rate_hz", None)

    return RotQuaternion(
        quats=np.array(dataset),
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
    )


def write_pointing_provider_to_hdf5(
    output_file: h5py.File,
    field_name: str,
    pointing_provider: PointingProvider,
    compression: Optional[str],
):
    rot_quaternion_field_name = f"{field_name}_rot_quaternion"
    write_rot_quaternion_to_hdf5(
        output_file=output_file,
        rot_matrix=pointing_provider.bore2ecliptic_quats,
        field_name=rot_quaternion_field_name,
        compression=compression,
    )

    if pointing_provider.has_hwp():
        hwp_field_name = f"{field_name}_hwp"
        pointing_provider.hwp.write_to_hdf5(
            output_file=output_file,
            field_name=hwp_field_name,
            compression=compression,
        )


def read_pointing_provider_from_hdf5(
    input_file: h5py.File,
    field_name: str,
) -> PointingProvider:
    rot_quaternion_field_name = f"{field_name}_rot_quaternion"
    rot_quaternion = read_rot_quaternion_from_hdf5(
        input_file=input_file,
        field_name=rot_quaternion_field_name,
    )

    hwp = None
    hwp_field_name = f"{field_name}_hwp"
    if hwp_field_name in input_file:
        hwp = read_hwp_from_hdf5(
            input_file=input_file,
            field_name=hwp_field_name,
        )

    return PointingProvider(
        bore2ecliptic_quats=rot_quaternion,
        hwp=hwp,
    )


def write_one_observation(
    output_file: h5py.File,
    obs: Observation,
    tod_dtype,
    pointings_dtype,
    global_index: int,
    local_index: int,
    tod_fields: List[Union[str, TodDescription]] = None,
    gzip_compression: bool = False,
    write_full_pointings: bool = False,
):
    """Write one :class:`Observation` object in a HDF5 file.

    This is a low-level function that stores a TOD in a HDF5 file. You should usually
    use more high-level functions that are able to write several observations at once,
    like :func:`.write_list_of_observations` and :func:`.write_observations`.

    By default, this function only saves the TODs and the quaternions necessary to
    compute the pointings; if you want the full pointing information, i.e., the
    angles θ (colatitude), φ (longitude), ψ (orientation) and α (HWP angle), you
    must set `write_full_pointings` to ``True``.

    The output file is specified by `output_file` and should be opened for writing; the
    observation to be written is passed through the `observations` parameter. The data
    type to use for writing TODs and pointings is specified in the `tod_dtype` and
    `pointings_dtype` (it can either be a NumPy type like ``numpy.float64`` or a
    string, pass ``None`` to use the same type as the one used in the observation).
    Note that quaternions are always saved using 64-bit floating point numbers.

    The `global_index` and `local_index` parameters are two integers that are
    used by high-level functions like :func:`.write_observations` to understand how to
    read several HDF5 files at once; if you do not need them, you can pass 0 to both.
    """

    compression = "gzip" if gzip_compression else None

    if obs.pointing_provider is not None:
        write_pointing_provider_to_hdf5(
            output_file=output_file,
            field_name="pointing_provider",
            pointing_provider=obs.pointing_provider,
            compression=compression,
        )

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
            elif isinstance(attr_value, RotQuaternion):
                # Do not save any information, as the rotation quaternion will be
                # saved in a dedicated dataset (see below)
                pass
            else:
                # From now on, assume that this attribute is an array whose length
                # is the same as `observations.n_detectors`
                attr_value = attr_value[det_idx]
                if type(attr_value) in __NUMPY_SCALAR_TYPES:
                    new_detector[attribute] = attr_value.item()
                elif isinstance(attr_value, np.ndarray):
                    new_detector[attribute] = attr_value.tolist()
                else:
                    # Fallback: we assume this is a plain Python type
                    new_detector[attribute] = attr_value

        det_info.append(new_detector)

        # Save the rotation quaternion of this detector in a dedicated dataset
        write_rot_quaternion_to_hdf5(
            output_file=output_file,
            field_name=f"rot_quaternion_{det_idx:04d}",
            rot_matrix=obs.quat[det_idx],
            compression=compression,
        )

    # Now encode `det_info` in a JSON string that will be saved later as an attribute
    detectors_json = json.dumps(det_info, cls=DetectorJSONEncoder)

    if not tod_fields:
        tod_fields = obs.tod_list

    # Write all the TOD timelines in the HDF5 file, in separate datasets
    for cur_field in tod_fields:
        if not isinstance(cur_field, TodDescription):
            try:
                cur_field = [x for x in obs.tod_list if x.name == cur_field][0]
            except IndexError:
                raise KeyError(f'TOD with name "{cur_field}" not found in observation')

        cur_dataset = output_file.create_dataset(
            cur_field.name,
            data=obs.__getattribute__(cur_field.name),
            dtype=tod_dtype if tod_dtype else cur_field.dtype,
            compression=compression,
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
        cur_dataset.attrs["description"] = cur_field.description

    # Save pointing information only if it is available
    if obs.pointing_provider and write_full_pointings:
        n_detectors = obs.n_detectors
        n_samples = obs.n_samples

        pointing_matrix = np.empty(shape=(n_detectors, n_samples, 3))

        hwp_angle = None
        if obs.pointing_provider.has_hwp():
            hwp_angle = np.empty(shape=(n_samples,))

        for det_idx in range(n_detectors):
            obs.get_pointings(
                det_idx,
                pointing_buffer=pointing_matrix[det_idx, :, :],
                hwp_buffer=hwp_angle,
            )

        output_file.create_dataset(
            "pointings",
            data=pointing_matrix,
            dtype=pointings_dtype,
            compression=compression,
        )

        if hwp_angle is not None:
            output_file.create_dataset(
                "hwp_angle",
                data=hwp_angle,
                dtype=pointings_dtype,
                compression=compression,
            )

    try:
        output_file.create_dataset(
            "global_flags",
            data=rle_compress(obs.__getattribute__("global_flags")),
            compression=compression,
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
                f"flags_{det_idx:04d}",
                data=compressed_flags,
                dtype=flags.dtype,
                compression=compression,
            )
    except (AttributeError, TypeError):
        pass

    output_file.attrs["mpi_rank"] = MPI_COMM_WORLD.rank
    output_file.attrs["mpi_size"] = MPI_COMM_WORLD.size
    output_file.attrs["global_index"] = global_index
    output_file.attrs["local_index"] = local_index
    output_file.attrs["det_idx"] = json.dumps([int(x) for x in obs.det_idx])


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
    observations: Union[Observation, List[Observation]],
    path: Union[str, Path],
    tod_dtype=np.float32,
    pointings_dtype=np.float64,
    file_name_mask: str = __OBSERVATION_FILE_NAME_MASK,
    custom_placeholders: Optional[List[Dict[str, Any]]] = None,
    start_index: int = 0,
    collective_mpi_call: bool = True,
    tod_fields: List[Union[str, TodDescription]] = [],
    gzip_compression: bool = False,
    write_full_pointings: bool = False,
) -> List[Path]:
    """
    Save a list of observations in a set of HDF5 files

    This function takes one or more observations and saves the TODs in several
    HDF5 (each observation leads to *one* file), using `tod_dtype` and
    `pointings_dtype` as the default datatypes for the samples and the pointing
    angles. The function returns a list of the file written (``pathlib.Path``
    objects).

    By default, this function only saves the TODs and the quaternions necessary to
    compute the pointings; if you want the full pointing information, i.e., the
    angles θ (colatitude), φ (longitude), ψ (orientation) and α (HWP angle), you
    must set `write_full_pointings` to ``True``.

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
            observations=[obs1, obs2],  # Write two observations
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
    fields using the parameter ``tod_fields`` (list of strings or
    :class:`.TodDescription` objects), which by default will only save
    `Observation.tod`.

    To save disk space, you can choose to apply GZip compression to the
    data frames in each HDF5 file (the file will still be a valid .h5
    file).

    """
    try:
        observations[0]
    except TypeError:
        observations = [observations]
    except IndexError:
        # Empty list
        # We do not want to return here, as we still need to participate to
        # the call to _compute_global_start_index below
        observations = []  # type: List[Observation]

    if not isinstance(path, Path):
        path = Path(path)

    global_start_index = _compute_global_start_index(
        num_of_obs=len(observations),
        start_index=start_index,
        collective_mpi_call=collective_mpi_call,
    )

    # Iterate over all the observations and create one HDF5 file for each of them
    file_list = []
    for obs_idx, cur_obs in enumerate(observations):
        params = {
            "mpi_rank": MPI_COMM_WORLD.rank,
            "mpi_size": MPI_COMM_WORLD.size,
            "num_of_obs": len(observations),
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
                gzip_compression=gzip_compression,
                write_full_pointings=write_full_pointings,
            )

        file_list.append(file_name)

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
    read_quaternions_if_present=True,
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

    The flags `tod_dtype` permits to override the data type of TOD samples used in
    the HDF5 file.

    The parameters `read_global_flags_if_present`, and `read_local_flags_if_present`
    permit to avoid loading some parts of the HDF5 if they are not needed.

    The function returns a :class:`.Observation`, or ``Nothing`` if the HDF5 file
    was ill-formed.
    """

    assert len(tod_fields) > 0

    # We'll fill the description later
    tod_full_fields = [
        TodDescription(name=x, dtype=tod_dtype, description="") for x in tod_fields
    ]

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
        # We'll slowly build "result" while iterating over the TODs, one by one
        result = None  # Optional[Observation]

        for cur_field_idx, cur_field in enumerate(tod_fields):
            if isinstance(cur_field, TodDescription):
                cur_field_name = cur_field.name
            else:
                cur_field_name = cur_field

            assert cur_field_name in inpf, (
                f"Field {cur_field_name} not found in HDF5 file {path}"
            )
            hdf5_tod = inpf[cur_field_name]

            if hdf5_tod.attrs["mjd_time"]:
                start_time = astropy.time.Time(
                    hdf5_tod.attrs["start_time"], format="mjd"
                )
            else:
                start_time = hdf5_tod.attrs["start_time"]

            tod_full_fields[cur_field_idx].description = hdf5_tod.attrs.get(
                "description", ""
            )

            if result is None:
                detectors = [
                    DetectorInfo.from_dict(x)
                    for x in json.loads(hdf5_tod.attrs["detectors"])
                ]

                # Read the rotation quaternion for this detector, as
                # it might not have been saved in the JSON record
                for det_idx, cur_det in enumerate(detectors):
                    cur_det.quat = read_rot_quaternion_from_hdf5(
                        input_file=inpf, field_name=f"rot_quaternion_{det_idx:04d}"
                    )

                result = Observation(
                    detectors=[asdict(d) for d in detectors],
                    n_samples_global=hdf5_tod.shape[1],
                    start_time_global=start_time,
                    n_blocks_det=1,
                    n_blocks_time=1,
                    allocate_tod=True,
                    sampling_rate_hz=hdf5_tod.attrs["sampling_rate_hz"],
                    comm=None if limit_mpi_rank else MPI_COMM_WORLD,
                    tods=tod_full_fields,
                )
                result.det_idx = np.array(json.loads(inpf.attrs["det_idx"]))
                # Copy the TOD in the newly created observation
                result.__setattr__(cur_field_name, hdf5_tod.astype(tod_dtype)[:])
            else:
                # All the fields must conform to the same shape as `Observation.tod`
                assert result.__getattribute__(tod_fields[0]).shape == hdf5_tod.shape
                result.__setattr__(cur_field_name, hdf5_tod.astype(tod_dtype)[:])

        # If we arrive here, we must have read at least one TOD
        assert result is not None

        # Let's fix the description of the TODs, now that we have read them all
        for i in range(len(tod_full_fields)):
            result.tod_list[i].description = tod_full_fields[i].description

        # If it is required, read other optional datasets
        for attr, attr_type, should_read in [
            ("global_flags", None, read_global_flags_if_present),
        ]:
            if (attr in inpf) and should_read:
                result.__setattr__(attr, inpf[attr].astype(attr_type)[:])

        if read_quaternions_if_present:
            pointing_provider = read_pointing_provider_from_hdf5(
                input_file=inpf,
                field_name="pointing_provider",
            )
            result.pointing_provider = pointing_provider

        # Checking if local flags are present is trickier because there should be N
        # datasets, where N is the number of detectors
        if read_local_flags_if_present:
            flags = __find_flags(
                inpf,
                expected_num_of_dets=result.__getattribute__(tod_fields[0]).shape[0],
                expected_num_of_samples=result.__getattribute__(tod_fields[0]).shape[1],
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
    limit_mpi_rank: bool = True,
    tod_fields: List[Union[str, TodDescription]] = ["tod"],
) -> List[Observation]:
    """Read a list of HDF5 files containing TODs and return a list of observations

    The function reads all the HDF5 files listed in `file_name_list` (either a list of
    strings or ``pathlib.Path`` objects) and assigns them to the various MPI processes
    that are currently running, provided that `limit_mpi_rank` is ``True``; otherwise,
    all the files are read by the current process and returned in a list. By default,
    only the ``tod`` field is loaded; if the HDF5 file contains multiple TODs, you
    must load each of them.

    When using MPI, the observations are distributed among the MPI processes using the
    same layout that was used to save them; this means that you are forced to use the
    same number of processes you used when saving the files. This number is saved in
    the attribute ``mpi_size`` in each of the HDF5 files.

    If the HDF5 file contains more than one TOD, e.g., foregrounds, dipole, noise…,
    you can specify which datasets to load with ``tod_fields`` (a list of strings
    or :class:`.TodDescription` objects), which defaults to ``["tod"]``. Each
    dataset will be initialized as a member field of the :class:`.Observation`
    class, like ``Observation.tod``.
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
                tod_fields=tod_fields,
            )
        )

    return observations
