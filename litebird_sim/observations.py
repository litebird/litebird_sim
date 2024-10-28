# -*- encoding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, List, Any, Optional

import astropy.time
import numpy as np
import numpy.typing as npt

from .coordinates import DEFAULT_TIME_SCALE
from .distribute import distribute_evenly


@dataclass
class TodDescription:
    """A brief description of a TOD held in a :class:`.Observation` object

    This field is used to pass information about one TOD in a :class:`.Observation`
    object. It is mainly used by the method :meth:`.Simulation.create_observation`
    to figure out how much memory allocate and how to organize it.

    The class contains three fields:

    - `name` (a ``str``): the name of the field to be created within each
      :class:`.Observation` object.

    - `dtype` (the NumPy type to use, e.g., ``numpy.float32``)

    - `description` (a ``str``): human-readable description
    """

    name: str
    dtype: Any
    description: str


class Observation:
    """An observation made by one or multiple detectors over some time window

    After construction at least the following attributes are available

    - :py:meth:`.start_time`

    - :py:meth:`.n_detectors`

    - :py:meth:`.n_samples`

    - :py:meth:`.tod` 2D array (`n_detectors` by `n_samples)` stacking
      the times streams of the detectors.

    A note for MPI-parallel application: unless specified, all the
      variables are *local*. Should you need the global counterparts,
      1) think twice, 2) append `_global` to the attribute name, like
      in the following:

    - :py:meth:`.start_time_global`

    - :py:meth:`.end_time_global`

    - :py:meth:`.n_detectors_global` `~ n_detectors * n_blocks_det`

    - :py:meth:`.n_samples_global` `~ n_samples * n_blocks_time`

    Following the same philosophy, :py:meth:`.get_times` returns the
    time stamps of the local time interval

    Args:
        detectors (int/list of dict): Either the number of detectors or
            a list of dictionaries with one entry for each detector. The keys of
            the dictionaries will become attributes of the observation. If a
            detector is missing a key it will be set to ``nan``.
            If an MPI communicator is passed to ``comm``, ``detectors`` is
            relevant only for the ``root`` process

        n_samples_global (int): The number of samples in this observation.

        start_time_global: Start time of the observation. It can either be a
            `astropy.time.Time` type or a floating-point number. In
            the latter case, it must be expressed in seconds.

        sampling_rate_hz (float): The sampling frequency, in Hertz.

        n_blocks_det (int): divide the detector axis of the tod (and all the
            arrays of detector attributes) in `n_blocks_det` blocks

        n_blocks_time (int): divide the time axis of the tod in
            `n_blocks_time` blocks

        comm: either `None` (do not use MPI) or a MPI communicator
            object, like `mpi4py.MPI.COMM_WORLD`. Its size is required to be at
            least `n_blocks_det` times `n_blocks_time`

        root (int): rank of the process receiving the detector list, if
            ``detectors`` is a list of dictionaries, otherwise it is ignored.

    """

    def __init__(
        self,
        detectors: Union[int, List[dict]],
        n_samples_global: int,
        start_time_global: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        allocate_tod=True,
        tods=None,
        n_blocks_det=1,
        n_blocks_time=1,
        comm=None,
        root=0,
    ):
        if tods is None:
            tods = [TodDescription(name="tod", dtype=np.float32, description="Signal")]

        self.comm = comm
        self.start_time_global = start_time_global
        self._n_samples_global = n_samples_global

        if isinstance(start_time_global, astropy.time.Time):
            delta = astropy.time.TimeDelta(
                1.0 / sampling_rate_hz, format="sec", scale=DEFAULT_TIME_SCALE
            )
        else:
            delta = 1.0 / sampling_rate_hz
        self.end_time_global = start_time_global + n_samples_global * delta

        if isinstance(detectors, int):
            self._n_detectors_global = detectors
        else:
            if comm and comm.size > 1:
                self._n_detectors_global = comm.bcast(len(detectors), root)
            else:
                self._n_detectors_global = len(detectors)

        self._sampling_rate_hz = sampling_rate_hz

        # Neme of the attributes that store an array with the value of a
        # property for each of the (local) detectors
        self._attr_det_names = []
        self._check_blocks(n_blocks_det, n_blocks_time)
        if comm and comm.size > 1:
            self._n_blocks_det = n_blocks_det
            self._n_blocks_time = n_blocks_time
        else:
            self._n_blocks_det = 1
            self._n_blocks_time = 1

        self.tod_list = tods
        for cur_tod in self.tod_list:
            if allocate_tod:
                setattr(
                    self,
                    cur_tod.name,
                    np.zeros(
                        self._get_tod_shape(n_blocks_det, n_blocks_time),
                        dtype=cur_tod.dtype,
                    ),
                )
            else:
                setattr(self, cur_tod.name, None)

        self.setattr_det_global("det_idx", np.arange(self._n_detectors_global), root)
        if not isinstance(detectors, int):
            self._set_attributes_from_list_of_dict(detectors, root)

        (
            self.start_time,
            self.start_sample,
            self.n_samples,
        ) = self._get_local_start_time_start_and_n_samples()

        self.pointing_provider = None  # type: Optional["PointingProvider"]

    @property
    def sampling_rate_hz(self):
        return self._sampling_rate_hz

    @property
    def n_detectors(self):
        return len(self.det_idx)

    def _get_local_start_time_start_and_n_samples(self):
        _, _, start, num = self._get_start_and_num(
            self._n_blocks_det, self._n_blocks_time
        )
        if self.comm:
            if self.comm.rank < self._n_blocks_time * self._n_blocks_det:
                start = start[self.comm.rank % self._n_blocks_time]
                num = num[self.comm.rank % self._n_blocks_time]
            else:
                start = 0
                num = 0
        else:
            start = start[0]
            num = num[0]

        if isinstance(self.start_time_global, astropy.time.Time):
            delta = astropy.time.TimeDelta(
                1.0 / self.sampling_rate_hz, format="sec", scale=DEFAULT_TIME_SCALE
            )
        else:
            delta = 1.0 / self.sampling_rate_hz

        return self.start_time_global + start * delta, start, num

    def _set_attributes_from_list_of_dict(self, list_of_dict, root):
        assert len(list_of_dict) == self.n_detectors_global

        # Turn list of dict into dict of arrays
        if not self.comm or self.comm.rank == root:
            # Build a list of all the keys in the dictionaries contained within
            # `list_of_dict` (which is a *list* of dictionaries)
            keys = list(set().union(*list_of_dict) - set(dir(self)))

            # This will be the dictionary associating each key with the
            # *array* of value for that dictionary
            dict_of_array = {k: [] for k in keys}

            # This array associates either np.nan or None to each type;
            # the former indicates that the value is a NumPy array, while
            # None is used for everything else
            nan_or_none = {}
            for k in keys:
                for d in list_of_dict:
                    if k in d:
                        try:
                            nan_or_none[k] = np.nan * d[k]
                        except TypeError:
                            nan_or_none[k] = None
                    break

            # Finally, build `dict_of_array`
            for d in list_of_dict:
                for k in keys:
                    dict_of_array[k].append(d.get(k, nan_or_none[k]))

            # Why should this code iterate over `keys`?!?
            for k in keys:
                dict_of_array = {k: np.array(dict_of_array[k]) for k in keys}
        else:
            keys = None
            dict_of_array = {}

        # Distribute the arrays
        if self.comm and self.comm.size > 1:
            keys = self.comm.bcast(keys)

        for k in keys:
            self.setattr_det_global(k, dict_of_array.get(k), root)

    @property
    def n_samples_global(self):
        """Samples in the whole observation

        If you need the time-lenght of the local TOD block  ``self.tod``, use
        either ``n_samples`` or ``self.tod.shape[1]``.
        """
        return self._n_samples_global

    @property
    def n_detectors_global(self):
        """Total number of detectors in the observation

        If you need the number of detectors in the local TOD block ``self.tod``,
        use either ``n_detectors`` or ``self.tod.shape[0]``.
        """
        return self._n_detectors_global

    @property
    def n_blocks_time(self):
        return self._n_blocks_time

    @property
    def n_blocks_det(self):
        return self._n_blocks_det

    def _check_blocks(self, n_blocks_det, n_blocks_time):
        if self.comm is None:
            if n_blocks_det != 1 or n_blocks_time != 1:
                raise ValueError("Only one block allowed without an MPI comm")
        elif n_blocks_det > self.n_detectors_global:
            raise ValueError(
                "You can not have more detector blocks than detectors "
                f"({n_blocks_det} > {self.n_detectors_global})"
            )
        elif n_blocks_time > self.n_samples_global:
            raise ValueError(
                "You can not have more time blocks than time samples "
                f"({n_blocks_time} > {self.n_blocks_time})"
            )
        elif self.comm.size < n_blocks_det * n_blocks_time:
            raise ValueError(
                "Too many blocks: n_blocks_det x n_blocks_time = "
                f"{n_blocks_det * n_blocks_time} but the number "
                f"processes is {self.comm.size}"
            )

    def _get_start_and_num(self, n_blocks_det, n_blocks_time):
        """For both detectors and time, returns the starting (global)
        index and lenght of each block if the number of blocks is changed to the
        values passed as arguments
        """
        det_start, det_n = np.array(
            [
                [span.start_idx, span.num_of_elements]
                for span in distribute_evenly(self._n_detectors_global, n_blocks_det)
            ]
        ).T
        time_start, time_n = np.array(
            [
                [span.start_idx, span.num_of_elements]
                for span in distribute_evenly(self._n_samples_global, n_blocks_time)
            ]
        ).T
        return (
            np.array(det_start),
            np.array(det_n),
            np.array(time_start),
            np.array(time_n),
        )

    def _get_tod_shape(self, n_blocks_det, n_blocks_time):
        """Return what the shape of ``self.tod`` will be if the blocks are set
        or changed to the values passed as arguments
        """
        if self.comm is None:
            # Observation not spread across MPI processes -> only one block
            return (self._n_detectors_global, self._n_samples_global)

        _, det_n, _, time_n = self._get_start_and_num(n_blocks_det, n_blocks_time)
        try:
            return (
                det_n[self.comm.rank // n_blocks_time],
                time_n[self.comm.rank % n_blocks_time],
            )
        except IndexError:
            return (0, 0)

    def set_n_blocks(self, n_blocks_det=1, n_blocks_time=1):
        """Change the number of blocks

        Args:
            n_blocks_det (int): new number of blocks in the detector direction
            n_blocks_time (int): new number of blocks in the time direction
        """
        # Checks and preliminaries
        self._check_blocks(n_blocks_det, n_blocks_time)
        if self.comm is None or self.comm.size == 1:
            return

        for cur_tod_def in self.tod_list:
            cur_tod = getattr(self, cur_tod_def.name)
            if cur_tod.dtype == np.float32:
                from mpi4py.MPI import REAL as mpi_dtype
            elif cur_tod.dtype == np.float64:
                from mpi4py.MPI import DOUBLE as mpi_dtype
            else:
                raise ValueError("Unsupported MPI dtype")

            new_tod = np.zeros(
                self._get_tod_shape(n_blocks_det, n_blocks_time), dtype=cur_tod.dtype
            )

            # Global start indices and number of elements of both the old and new
            # blocking scheme
            (
                det_start_sends,
                _,
                time_start_sends,
                time_num_sends,
            ) = self._get_start_and_num(self.n_blocks_det, self.n_blocks_time)
            (
                det_start_recvs,
                _,
                time_start_recvs,
                time_num_recvs,
            ) = self._get_start_and_num(n_blocks_det, n_blocks_time)

            # Prepare a matrix with the message sizes to be sent/received
            # For a given row, ith old block sends counts[i, j] elements to
            # the jth new block
            counts = (
                np.append(time_start_recvs, self._n_samples_global)[1:]
                - time_start_sends[:, None]
            )
            counts = np.where(counts < 0, 0, counts)
            counts = np.where(
                counts > time_num_sends[:, np.newaxis],
                time_num_sends[:, np.newaxis],
                counts,
            )
            counts = np.where(counts > time_num_recvs, time_num_recvs, counts)
            cum_counts = np.cumsum(counts, 1)
            excess = cum_counts - time_num_sends[:, np.newaxis]
            for i in range(self.n_blocks_time):
                for j in range(n_blocks_time):
                    if excess[i, j] > 0:
                        counts[i, j] -= excess[i, j]
                        counts[i, j + 1 :] = 0
                        break

            assert np.all(counts.sum(0) == time_num_recvs)
            assert np.all(counts.sum(1) == time_num_sends)

            # Move the tod to the new blocks (new_tod) row by row.
            for d in range(self.n_detectors_global):
                # Get the ranks of the processes involved in the send and in the
                # receive. and create a communicator for them
                first_sender = (
                    np.where(d >= det_start_sends)[0][-1] * self.n_blocks_time
                )
                first_recver = np.where(d >= det_start_recvs)[0][-1] * n_blocks_time
                is_rank_in_row = (
                    first_sender <= self.comm.rank < first_sender + self.n_blocks_time
                    or first_recver <= self.comm.rank < first_recver + n_blocks_time
                )

                comm_row = self.comm.Split(int(is_rank_in_row))
                if not is_rank_in_row:
                    continue  # Process not involved, move to the next row

                # first_sender and first_recver to the row ranks
                rank_gap = max(
                    first_recver - first_sender - self.n_blocks_time - 1,
                    first_sender - first_recver - n_blocks_time - 1,
                    0,
                )
                if first_sender < first_recver:
                    first_recver -= first_sender + rank_gap
                    first_sender = 0
                else:
                    first_sender -= first_recver + rank_gap
                    first_recver = 0

                # Prepare send data
                send_counts = np.zeros(comm_row.size, dtype=np.int32)
                i_block = comm_row.rank - first_sender
                if 0 <= i_block < self.n_blocks_time:
                    send_counts[first_recver : first_recver + n_blocks_time] = counts[
                        comm_row.rank - first_sender
                    ]
                send_disp = np.zeros_like(send_counts)
                np.cumsum(send_counts[:-1], out=send_disp[1:])
                try:
                    send_buff = cur_tod[
                        d - det_start_sends[self.comm.rank // self.n_blocks_time]
                    ]
                except IndexError:
                    send_buff = None
                send_data = [send_buff, send_counts, send_disp, mpi_dtype]

                # recv
                recv_counts = np.zeros(comm_row.size, dtype=np.int32)
                i_block = comm_row.rank - first_recver
                if 0 <= i_block < n_blocks_time:
                    recv_counts[
                        first_sender : first_sender + self.n_blocks_time
                    ] = counts[:, i_block]

                recv_disp = np.zeros_like(recv_counts)
                np.cumsum(recv_counts[:-1], out=recv_disp[1:])

                try:
                    recv_buff = new_tod[
                        d - det_start_recvs[self.comm.rank // n_blocks_time]
                    ]
                except IndexError:
                    recv_buff = None
                recv_data = [recv_buff, recv_counts, recv_disp, mpi_dtype]

                comm_row.Alltoallv(send_data, recv_data)

            setattr(self, cur_tod_def.name, new_tod)

        is_in_old_fist_col = self.comm.rank % self._n_blocks_time == 0
        is_in_old_fist_col &= self.comm.rank // self._n_blocks_time < self._n_blocks_det
        comm_first_col = self.comm.Split(int(is_in_old_fist_col))

        self._n_blocks_det = n_blocks_det
        self._n_blocks_time = n_blocks_time

        for name in self._attr_det_names:
            info = None
            if is_in_old_fist_col:
                info = comm_first_col.gather(getattr(self, name))
                if self.comm.rank == 0:
                    info = np.concatenate(info)
            self.setattr_det_global(name, info)

        (
            self.start_time,
            self.start_sample,
            self.n_samples,
        ) = self._get_local_start_time_start_and_n_samples()

    def setattr_det(self, name, info):
        """Add a piece of information about the detectors

        Store ``info`` as the attribute ``name`` of the observation.
        The difference with respect to ``self.name = info``, relevant only
        in MPI programs, are

         * ``info`` is assumed to contain a number of elements equal to
           ``self.tod.shape[0]`` and to have the same order (``info[i]`` is a
           property of ``self.tod[i]``)
         * When changing ``n_blocks_det``, :py:meth:`.set_n_blocks` is aware of
           ``name`` and will redistribute ``info`` in such a way that
           ``self.name[i]`` is a property of ``self.tod[i]`` in the new block
           distribution

        Args:
            name (str): Name of the detector information
            info (array): Information to be stored in the attribute ``name``.
                The array must contain one entry for each *local* detector.

        """
        self._attr_det_names.append(name)
        assert len(info) == self.n_detectors
        setattr(self, name, info)

    def setattr_det_global(self, name, info, root=0):
        """Add a piece of information on the detectors

        Variant of :py:meth:`.setattr_det` to be used when the information
        comes from a single MPI rank (``root``). In particular,

         * In the ``root`` process, ``info`` is required to have
           ``n_detectors_global`` elements (not ``self.tod.shape[1]``).
           For other processes info is irrelevant
         * ``info`` is scattered from the ``root`` rank to the relevant
           processes in ``self.comm`` so that ``self.name`` will have
           ``self.tod.shape[0]`` elements on all the processes  and
           ``self.name[i]`` is a property of ``self.tod[i]``
         * When changing ``n_blocks_det``, :py:meth:`.set_n_blocks` is aware of
           ``name`` and will redistribute ``info`` in such a way that
           ``self.name[i]`` is a property of ``self.tod[i]`` in the new block
           distribution

        Args:
            name (str): Name of the information
            info (array): Array containing ``n_detectors_global`` entries.
                Relevant only for thr ``root`` process
            root (int): Rank of the root process

        """
        if name not in self._attr_det_names:
            self._attr_det_names.append(name)

        if not self.comm or self.comm.size == 1:
            assert len(info) == self.n_detectors_global
            setattr(self, name, info)
            return

        is_in_grid = self.comm.rank < self._n_blocks_det * self._n_blocks_time
        comm_grid = self.comm.Split(int(is_in_grid))
        if not is_in_grid:  # The process does not own any detector (and TOD)
            setattr(self, name, None)
            return

        my_col = comm_grid.rank % self._n_blocks_time
        comm_col = comm_grid.Split(my_col)
        root_col = root // self._n_blocks_det
        if my_col == root_col:
            if comm_grid.rank == root:
                starts, nums, _, _ = self._get_start_and_num(
                    self._n_blocks_det, self._n_blocks_time
                )
                info = [info[s : s + n] for s, n in zip(starts, nums)]

            info = comm_col.scatter(info, root)

        comm_row = comm_grid.Split(comm_grid.rank // self._n_blocks_time)
        info = comm_row.bcast(info, root_col)
        assert (not self.tod_list) or len(info) == len(
            getattr(self, self.tod_list[0].name)
        )
        setattr(self, name, info)

    def get_delta_time(self) -> Union[float, astropy.time.TimeDelta]:
        """Return the time interval between two consecutive samples in this observation

        Depending whether the field ``start_time`` of the :class:`.Observation` object
        is a ``float`` or a ``astropy.time.Time`` object, the return value is either a
        ``float`` (in seconds) or an instance of ``astropy.time.TimeDelta``. See also
        :meth:`.get_time_span`."""

        delta = 1.0 / self.sampling_rate_hz
        if isinstance(self.start_time, astropy.time.Time):
            delta = astropy.time.TimeDelta(delta, format="sec", scale="tdb")

        return delta

    def get_time_span(self) -> Union[float, astropy.time.TimeDelta]:
        """Return the temporal length of the current observation

        This method can either return a ``float`` (in seconds) or a
        ``astropy.time.TimeDelta`` object, depending whether the field ``start_time``
        of the :class:`.Observation` object is a ``float`` or a
        ``astropy.time.Time`` instance. See also :meth:`.get_delta_time`."""
        return self.get_delta_time() * self.n_samples

    def get_times(self, normalize=False, astropy_times=False):
        """Return a vector containing the time of each sample in the observation

        The measure unit of the result depends on the value of
        `astropy_times`: if it's true, times are returned as
        `astropy.time.Time` objects, which can be converted to several
        units (MJD, seconds, etc.); if `astropy_times` is false (the
        default), times are expressed in seconds. In the latter case,
        you should interpret these times as sidereal.

        If `normalize=True`, then the first time is zero. Setting
        this flag requires that `astropy_times=False`.

        This can be a costly operation; you should cache this result
        if you plan to use it in your code, instead of calling this
        method over and over again.

        Note for MPI-parallel codes: the times returned are only those of the
        local portion of the data. This means that

         * the size of the returned array is ``n_samples``, smaller than
           ``n_samples_global`` whenever there is more than one time-block
         * ``self.tod * self.get_times()`` is a meaningless but always
           allowed operation
        """
        if normalize:
            assert (
                not astropy_times
            ), "you cannot pass astropy_times=True *and* normalize=True"

            return (
                np.arange(self.n_samples) + self.start_sample
            ) / self.sampling_rate_hz

        if astropy_times:
            assert isinstance(self.start_time, astropy.time.Time), (
                "to use astropy_times=True you must specify an astropy.time.Time "
                "object in Observation.__init__"
            )
            delta = self.get_delta_time()
            return self.start_time + np.arange(self.n_samples) * delta

        if isinstance(self.start_time, astropy.time.Time):
            # We use "cxcsec" because of the following features:
            #
            # 1. It's one of the astropy.time.Time formats that
            #    measures time in seconds (alongside with "unix"
            #    and "gps")
            #
            # 2. Of the three choices, "cxcsec" uses the most
            #    recent date as reference (1998-01-01, vs.
            #    1990-01-01 for "gps" and 1980-01-06 for "unix").
            t0 = self.start_time.cxcsec
        else:
            t0 = self.start_time

        return t0 + np.arange(self.n_samples) / self.sampling_rate_hz

    def get_pointings(
        self,
        detector_idx: Union[int, List[int], str] = "all",
        pointing_buffer: Optional[npt.NDArray] = None,
        hwp_buffer: Optional[npt.NDArray] = None,
        pointings_dtype=np.float32,
    ) -> (npt.NDArray, Optional[npt.NDArray]):
        """
        Compute the pointings for one or more detectors in this observation

        This method triggers the computation of the matrix of pointings that indicate
        the direction of the line of sight for each sample in the TOD of the current
        :class:`.Observation` instance. You must call either
        :func:`.prepare_pointings` or :meth:`.Simulation.prepare_pointings`
        *before* invoking this method.

        The parameter `detector_idx` specifies which detectors should be included in
        the computation. Use ``"all"`` to ask for the pointings of *all* the detectors
        in this Observation; if you just want a subset of them, pass a list with their
        zero-based index; if you just want the pointings for one detector, you can
        pass an integer. The following calls are all legitimate::

            # All the detectors are included
            pointings, hwp_angle = cur_obs.get_pointings("all")

            # Only the first two detectors are included
            pointings, hwp_angle = cur_obs.get_pointings([0, 1])

            # Only the first detector is used
            pointings, hwp_angle = cur_obs.get_pointings(0)

        The return value is a pair containing (1) the pointing matrix and (2) the
        HWP angle. The pointing matrix is a NumPy array with shape ``(N_det, N_samples, 3)``,
        where ``N_det` is the number of detectors and ``N_samples`` is the number of
        samples in the TOD (the field ``Observation.n_samples``). The last dimension
        spans the three angles θ (colatitude, in radians), φ (longitude, in radians),
        and ψ (orientation angle, in radians). *Important*: if you ask for just *one*
        detector, the shape of the pointing matrix will always be ``(N_samples, 3)``.
        The HWP angle is always a vector with shape ``(N_samples,)``, as it does
        not depend on the list of detectors.

        The return value is allocated internally by the method. If you instead want to
        pass a pre-allocated structure, you can use the `pointing_buffer` and
        `hwp_buffer` parameters. In this case, the return value will be *always*
        equal to ``(pointing_buffer, hwp_buffer)``.
        """
        assert (
            self.pointing_provider is not None
        ), "You must initialize pointing_provider; use Simulation.prepare_pointings()"

        # Simplest case: we need just one detector
        if isinstance(detector_idx, int):
            assert (
                (detector_idx >= 0) and (detector_idx < self.n_detectors)
            ), f"Invalid detector index {detector_idx}, it must be a number between 0 and {self.n_detectors - 1}"

            pointing, hwp = self.pointing_provider.get_pointings(
                detector_quat=self.quat[detector_idx],
                start_time=self.start_time,
                start_time_global=self.start_time_global,
                sampling_rate_hz=self.sampling_rate_hz,
                nsamples=self.n_samples,
                pointing_buffer=pointing_buffer,
                hwp_buffer=hwp_buffer,
                pointings_dtype=pointings_dtype,
            )
            return pointing.reshape(1, -1, 3), hwp

        # More complex case: we need all the detectors
        if isinstance(detector_idx, str):
            assert detector_idx == "all", f"Unknown set of detectors: '{detector_idx}'"

            # Recursive call
            return self.get_pointings(
                list(range(self.n_detectors)),
                pointing_buffer=pointing_buffer,
                hwp_buffer=hwp_buffer,
                pointings_dtype=pointings_dtype,
            )

        # Most complex case: an explicit list (or NumPy array) of detectors
        assert np.ndim(detector_idx) != 0

        expected_shape = (len(detector_idx), self.n_samples, 3)
        if pointing_buffer is not None:
            assert (
                pointing_buffer.shape == expected_shape
            ), "pointing_buffer has a wrong shape, it is {actual} but should be {expected}".format(
                actual=pointing_buffer.shape, expected=expected_shape
            )
        else:
            pointing_buffer = np.empty(expected_shape, dtype=pointings_dtype)

        expected_shape = (self.n_samples,)
        if self.pointing_provider.has_hwp():
            if hwp_buffer is not None:
                assert (
                    hwp_buffer.shape == expected_shape
                ), "hwp_buffer has a wrong shape, it is {actual} but should be {expected}".format(
                    actual=hwp_buffer.shape, expected=expected_shape
                )
            else:
                hwp_buffer = np.empty(expected_shape, dtype=pointings_dtype)
        else:
            hwp_buffer = None

        # For a generic case with *four* detectors, we need *two* iterators.
        # Consider the detectors
        #
        #      det#0    det#1    det#2    det#3
        #
        # Suppose that we're asking for the pointings of
        # detectors #1 and #3. In this case, the pointing matrix will be such that
        #
        #    pointings[0, :, :] ← pointings of det#1
        #    pointings[1, :, :] ← pointings of det#3
        #
        # Thus, index 0 in the `pointings` matrix maps to index 1 in the list of
        # detectors, and index 1 maps to index 3. The `rel_det_idx` index
        # is used with the `pointings` matrix, while `abs_det_idx` is used
        # with the list of detectors.
        for rel_det_idx, abs_det_idx in enumerate(detector_idx):
            _ = self.get_pointings(
                abs_det_idx,
                pointing_buffer=pointing_buffer[rel_det_idx, :, :],
                hwp_buffer=hwp_buffer,
            )

        return pointing_buffer, hwp_buffer
