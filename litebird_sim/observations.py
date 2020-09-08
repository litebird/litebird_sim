# -*- encoding: utf-8 -*-

from .distribute import distribute_evenly, distribute_optimally
import astropy.time as astrotime
import numpy as np


class Observation:
    """An observation made by one or multiple detectors over some time window

    This class encodes the data acquired by one or multipel detectors over a
    finite amount of time, and it is the fundamental block of a
    simulation. The characteristics of the detector are assumed to be
    stationary over the time span covered by a simulation; these include:

    - Noise parameters
    - Gain

    To access the TOD, use one of the following methods:

    - :py:meth:`.get_times` returns the array of time values (one per
      each sample in the TOD)
    - :py:meth:`.tod` returns the array of samples

    Args:
        detectors (int/list of dict): Either the number of detectors or
            a list of dictionaries with one entry for each detector. The keys of
            the dictionaries will become attributes of the observation. If a
            detector is missing a key it will be set to ``nan``. 
            If an MPI communicator is passed to ``comm``, ``detectors`` is
            relevant only for the ``root`` process

        n_samples (int): The number of samples in this observation.

        start_time: Start time of the observation. It can either be a
            `astropy.time.Time` type or a floating-point number. In
            the latter case, if `use_mjd` is ``False``, the number
            must be expressed in seconds; otherwise, it must be a MJD.

        sampling_rate_hz (float): The sampling frequency. Regardless of the
            measurement unit used for `start_time`, this *must* be
            expressed in Hertz.

        use_mjd (bool): If ``True``, the value of `start_time` is
            expressed in MJD, otherwise it's in seconds.

        dtype_tod (dtype): Data type of the TOD array

        n_blocks_det (int): divide the detector axis of the tod in
            `n_blocks_det` blocks

        n_blocks_time (int): divide the time axis of the tod in
            `n_blocks_time` blocks

        comm: either `None` (do not use MPI) or a MPI communicator
            object, like `mpi4py.MPI.COMM_WORLD`. Its size is required to be at
            least `n_blocks_det` times `n_blocks_time`

        root (int): rank of the process receiving the detector list, if
            ``detectors`` is a list of dictionaries, otherwise it is ignored.
    """

    def __init__(
        self, detectors, n_samples,
        start_time, sampling_rate_hz, use_mjd=False, dtype_tod=np.float32,
        n_blocks_det=1, n_blocks_time=1, comm=None, root=0
    ):
        self.comm = comm
        self._n_samples = n_samples
        if isinstance(detectors, int):
            self._n_detectors = detectors
        else:
            if comm:
                self._n_detectors = comm.bcast(len(detectors), root)
            else:
                self._n_detectors = len(detectors)

        self.use_mjd = use_mjd
        self.sampling_rate_hz = sampling_rate_hz

        # Neme of the attributes that store an array with the value of a
        # property for each of the (local) detectors
        self._detector_info_names = []

        if isinstance(start_time, astrotime.Time):
            if self.use_mjd:
                self.start_time = start_time.mjd
            else:
                # We use "cxcsec" because of the following features:
                #
                # 1. It's one of the astropy.time.Time formats that
                #    measures time in seconds (alongside with "unix"
                #    and "gps")
                #
                # 2. Of the three choices, "cxcsec" uses the most
                #    recent date as reference (1998-01-01, vs.
                #    1990-01-01 for "gps" and 1980-01-06 for "unix").
                self.start_time = start_time.cxcsec
        else:
            self.start_time = start_time

        self._check_blocks(n_blocks_det, n_blocks_time)
        if comm:
            self._n_blocks_det = n_blocks_det
            self._n_blocks_time = n_blocks_time
        else:
            self._n_blocks_det = 1
            self._n_blocks_time = 1

        self.tod = np.empty(
            self._get_tod_shape(n_blocks_det, n_blocks_time),
            dtype=dtype_tod)

        if not isinstance(detectors, int):
            self._set_attributes_from_list_of_dict(detectors, root)

    def _set_attributes_from_list_of_dict(self, list_of_dict, root):
        assert len(list_of_dict) == self.n_detectors

        # Turn list of dict into dict of arrays
        if not self.comm or self.comm.rank == root:
            keys = list(set().union(*list_of_dict))
            dict_of_array = {k: [] for k in keys}
            nan_or_none = {}
            for k in keys:
                for d in list_of_dict:
                    if k in d:
                        try:
                            nan_or_none[k] = np.nan * d[k]
                        except TypeError:
                            nan_or_none[k] = None
                    break

            for d in list_of_dict:
                for k in keys:
                    dict_of_array[k].append(d.get(k, nan_or_none[k]))

            for k in keys:
                dict_of_array = {k: np.array(dict_of_array[k]) for k in keys}
        else:
            keys = None
            dict_of_array = {}

        # Distribute the arrays
        if self.comm:
            keys = self.comm.bcast(keys)

        for k in keys:
            self.detector_global_info(k, dict_of_array.get(k), root)

    @property
    def n_samples(self):
        """ Samples in the whole observation

        Note
        ----
        If you need the time-lenght of the ``self.tod`` array, use directly
        ``self.tod.shape[1]``: ``n_samples`` allways refers to the full
        observation, even when it is distributed over multiple processes and
        ``self.tod`` is just the local block.
        """
        return self._n_samples

    @property
    def n_detectors(self):
        return self._n_detectors

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
        elif n_blocks_det > self.n_detectors:
            raise ValueError(
                "You can not have more detector blocks than detectors "
                f"({n_blocks_det} > {self.n_detectors})")
        elif n_blocks_time > self.n_samples:
            raise ValueError(
                "You can not have more time blocks than time samples "
                f"({n_blocks_time} > {self.n_blocks_time})")
        elif self.comm.size < n_blocks_det * n_blocks_time:
            raise ValueError(
                "Too many blocks: n_blocks_det x n_blocks_time = "
                f"{n_blocks_det * n_blocks_time} but the number "
                f"processes is {self.comm.size}")

    def _get_start_and_num(self, n_blocks_det, n_blocks_time):
        """ For both detectors and time, returns the starting (global)
        index and lenght of each block if the number of blocks is changed to the
        values passed as arguments
        """ 
        det_start, det_n = np.array(
            [[span.start_idx, span.num_of_elements]
             for span in distribute_evenly(self._n_detectors, n_blocks_det)]).T
        time_start, time_n = np.array(
            [[span.start_idx, span.num_of_elements]
             for span in distribute_evenly(self._n_samples, n_blocks_time)]).T
        return (np.array(det_start), np.array(det_n),
                np.array(time_start), np.array(time_n))

    def _get_tod_shape(self, n_blocks_det, n_blocks_time):
        """ Return what the shape of ``self.tod`` will be if the blocks are set
        or changed to the values passed as arguments
        """ 
        if self.comm is None:
            # Observation not spread across MPI processes -> only one block
            return (self._n_detectors, self._n_samples)

        _, det_n, _, time_n = self._get_start_and_num(n_blocks_det,
                                                      n_blocks_time)
        try:
            return (det_n[self.comm.rank // n_blocks_time],
                    time_n[self.comm.rank % n_blocks_time])
        except IndexError:
            return (0, 0)

    def set_n_blocks(self, n_blocks_det=1, n_blocks_time=1):
        """ Change the number of blocks

        Args:
            n_blocks_det (int): new number of blocks in the detector direction
            n_blocks_time (int): new number of blocks in the time direction
            
        """ 
        # Checks and preliminaries
        self._check_blocks(n_blocks_det, n_blocks_time)
        if self.comm is None:
            return
        if self.tod.dtype == np.float32:
            from mpi4py.MPI import REAL as mpi_dtype
        elif self.tod.dtype == np.float64:
            from mpi4py.MPI import DOUBLE as mpi_dtype
        else:
            raise ValueError("Unsupported MPI dtype")

        new_tod = np.zeros(
            self._get_tod_shape(n_blocks_det, n_blocks_time),
            dtype=self.tod.dtype)

        # Global start indices and number of elements of both the old and new
        # blocking scheme
        det_start_sends, _, time_start_sends, time_num_sends =\
            self._get_start_and_num(self.n_blocks_det, self.n_blocks_time)
        det_start_recvs, _, time_start_recvs, time_num_recvs =\
            self._get_start_and_num(n_blocks_det, n_blocks_time)

        # Prepare a matrix with the message sizes to be sent/received
        # For a given row, ith old block sends counts[i, j] elements to
        # the jth new block
        counts = (np.append(time_start_recvs, self._n_samples)[1:]
                  - time_start_sends[:, None])
        counts = np.where(counts < 0, 0, counts)
        counts = np.where(counts > time_num_sends[:, np.newaxis],
                          time_num_sends[:, np.newaxis], counts)
        counts = np.where(counts > time_num_recvs, time_num_recvs, counts)
        cum_counts = np.cumsum(counts, 1)
        excess = cum_counts - time_num_sends[:, np.newaxis]
        for i in range(self.n_blocks_time):
            for j in range(n_blocks_time):
                if excess[i, j] > 0:
                    counts[i, j] -= excess[i, j]
                    counts[i, j + 1:] = 0
                    break

        assert np.all(counts.sum(0) == time_num_recvs)
        assert np.all(counts.sum(1) == time_num_sends)

        # Move the tod to the new blocks (new_tod) row by row.
        for d in range(self.n_detectors):
            # Get the ranks of the processes involved in the send and in the
            # receive. and create a communicator for them
            first_sender = (np.where(d >= det_start_sends)[0][-1]
                            * self.n_blocks_time)
            first_recver = (np.where(d >= det_start_recvs)[0][-1]
                            * n_blocks_time)
            is_rank_in_row = (
                first_sender <= self.comm.rank < first_sender + self.n_blocks_time
                or first_recver <= self.comm.rank < first_recver + n_blocks_time)

            comm_row = self.comm.Split(int(is_rank_in_row))
            if not is_rank_in_row:
                continue  # Process not involved, move to the next row

            # first_sender and first_recver to the row ranks
            rank_gap = max(first_recver - first_sender - self.n_blocks_time - 1,
                           first_sender - first_recver - n_blocks_time - 1,
                           0)
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
                send_counts[first_recver: first_recver + n_blocks_time] =\
                    counts[comm_row.rank - first_sender]
            send_disp = np.zeros_like(send_counts)
            np.cumsum(send_counts[:-1], out=send_disp[1:])
            try:
                send_buff = self.tod[
                    d - det_start_sends[self.comm.rank // self.n_blocks_time]]
            except IndexError:
                send_buff = None
            send_data = [send_buff, send_counts, send_disp, mpi_dtype]

            # recv
            recv_counts = np.zeros(comm_row.size, dtype=np.int32)
            i_block = comm_row.rank - first_recver
            if 0 <= i_block < n_blocks_time:
                recv_counts[first_sender: first_sender + self.n_blocks_time] =\
                    counts[:, i_block]

            recv_disp = np.zeros_like(recv_counts)
            np.cumsum(recv_counts[:-1], out=recv_disp[1:])

            try:
                recv_buff = new_tod[
                    d - det_start_recvs[self.comm.rank // n_blocks_time]]
            except IndexError:
                recv_buff = None
            recv_data = [recv_buff, recv_counts, recv_disp, mpi_dtype]

            comm_row.Alltoallv(send_data, recv_data)

        self.tod = new_tod

        is_in_old_fist_col = (self.comm.rank % self._n_blocks_time == 0)
        is_in_old_fist_col &= (self.comm.rank // self._n_blocks_time
                               < self._n_blocks_det)
        comm_first_col = self.comm.Split(int(is_in_old_fist_col))

        self._n_blocks_det = n_blocks_det
        self._n_blocks_time = n_blocks_time

        for name in self._detector_info_names:
            info = None
            if is_in_old_fist_col:
                info = comm_first_col.gather(getattr(self, name))
                if self.comm.rank == 0:
                    info = np.concatenate(info)
            self.detector_global_info(name, info)

    def detector_info(self, name, info):
        """ Piece of information on the detectors

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
            name (str): Name of the information
            info (array): Information to be stored in the attribute ``name``.
                The array must contain one entry for each *local* detector.

        """
        self._detector_info_names.append(name)
        assert len(info) == len(self.tod)
        setattr(self, name, info)

    def detector_global_info(self, name, info, root=0):
        """ Piece of information on the detectors

        Variant of :py:meth:`.detector_info` to be used when the information
        comes from a single MPI rank (``root``). In particular,

         * In the ``root`` process, ``info`` is required to have
           ``n_detectors`` elements (not ``self.tod.shape[1]``).
           For other processes info is irrelevant
         * ``info`` is scattered from the ``root`` rank to all the processes in
           ``self.comm`` so that ``self.name`` will have ``self.tod.shape[0]``
           elements and ``self.name[i]`` is a property of ``self.tod[i]``
         * When changing ``n_blocks_det``, :py:meth:`.set_n_blocks` is aware of
           ``name`` and will redistribute ``info`` in such a way that
           ``self.name[i]`` is a property of ``self.tod[i]`` in the new block
           distribution

        Args:
            name (str): Name of the information
            info (array): Array containing ``n_detectors`` entries.
                Relevant only for thr ``root`` process
            root (int): Rank of the root process

        """
        if name not in self._detector_info_names:
            self._detector_info_names.append(name)

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
                    self._n_blocks_det, self._n_blocks_time)
                info = [info[s:s + n] for s, n, in zip(starts, nums)]

            info = comm_col.scatter(info, root)

        comm_row = comm_grid.Split(comm_grid.rank // self._n_blocks_time)
        info = comm_row.bcast(info, root_col)
        assert len(info) == len(self.tod)
        setattr(self, name, info)

    def get_times(self):
        """Return a vector containing the time of each sample in the tod

        The measure unit of the result depends whether
        ``self.use_mjd`` is true (return MJD) or false (return
        seconds). The number of elements in the vector is equal to
        ``self.tod.shape[1]``.

        Note for MPI-parallel codes: the times returned are only those of the
        local portion of the data. This means that

         * the size of the returned array is smaller than ``n_samples`` whenever
           there is more than one time-block
         * ``self.tod * self.get_times()`` is a meaningless but always
           allowed operation

        This can be a costly operation; you should cache this result
        if you plan to use it in your code, instead of calling this
        method over and over again.
        """
        start = self._get_start_and_num(self._n_blocks_det,
                                        self._n_blocks_time)[2]

        if self.comm:
            start = start[self.comm.rank % self._n_blocks_time]
        else:
            start = start[0]

        local_time_idx = start + np.arange(self.tod.shape[1])
        if self.use_mjd:
            delta = astrotime.TimeDelta(1.0 / self.sampling_rate_hz, format="sec")
            vec = (
                astrotime.Time(self.start_time, format="mjd")
                + local_time_idx * delta
            )
            return vec.mjd
        else:
            return (self.start_time + local_time_idx / self.sampling_rate_hz)
