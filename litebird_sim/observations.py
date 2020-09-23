# -*- encoding: utf-8 -*-

from typing import Union, List
import astropy.time
import numpy as np

from .distribute import distribute_evenly
from .scanning import (
    Spin2EclipticQuaternions,
    all_compute_pointing_and_polangle,
)


class Observation:
    """An observation made by one or multiple detectors over some time window

    This class encodes the data acquired by one or multiple detectors over a
    finite amount of time, and it is the fundamental block of a
    simulation. The characteristics of the detector are assumed to be
    stationary over the time span covered by a simulation; these can include:

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
            the latter case, it must be expressed in seconds.

        sampling_rate_hz (float): The sampling frequency, in Hertz.


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
        self,
        detectors: Union[int, List[dict]],
        n_samples: int,
        start_time: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        allocate_tod=True, dtype_tod=np.float32,
        n_blocks_det=1, n_blocks_time=1, comm=None, root=0
    ):
        self.comm = comm
        self.start_time = start_time
        self._n_samples = n_samples
        if isinstance(detectors, int):
            self._n_detectors = detectors
        else:
            if comm and comm.size > 1:
                self._n_detectors = comm.bcast(len(detectors), root)
            else:
                self._n_detectors = len(detectors)

        self._sampling_rate_hz = sampling_rate_hz

        # Neme of the attributes that store an array with the value of a
        # property for each of the (local) detectors
        self._detector_info_names = []
        self._check_blocks(n_blocks_det, n_blocks_time)
        if comm and comm.size > 1:
            self._n_blocks_det = n_blocks_det
            self._n_blocks_time = n_blocks_time
        else:
            self._n_blocks_det = 1
            self._n_blocks_time = 1

        if allocate_tod:
            self.tod = np.empty(
                self._get_tod_shape(n_blocks_det, n_blocks_time),
                dtype=dtype_tod)

        self.detector_global_info('det_idx', np.arange(self._n_detectors), root)
        if not isinstance(detectors, int):
            self._set_attributes_from_list_of_dict(detectors, root)

        self.local_start_time, self.local_start, self.local_n_samples = \
            self._get_local_start_time_start_and_n_samples()

    @property
    def sampling_rate_hz(self):
        return self._sampling_rate_hz

    def _get_local_start_time_start_and_n_samples(self):
        _, _, start, num = self._get_start_and_num(self._n_blocks_det,
                                                   self._n_blocks_time)
        if self.comm:
            if (self.comm.rank < self._n_blocks_time * self._n_blocks_det):
                start = start[self.comm.rank % self._n_blocks_time]
                num = num[self.comm.rank % self._n_blocks_time]
            else:
                start = 0
                num = 0
        else:
            start = start[0]
            num = num[0]

        if isinstance(self.start_time, astropy.time.Time):
            delta = astropy.time.TimeDelta(1.0 / self.sampling_rate_hz,
                                           format="sec", scale="tdb")
        else:
            delta = 1.0 / self.sampling_rate_hz

        return self.start_time + start * delta, start, num

    def _set_attributes_from_list_of_dict(self, list_of_dict, root):
        assert len(list_of_dict) == self.n_detectors

        # Turn list of dict into dict of arrays
        if not self.comm or self.comm.rank == root:
            keys = list(set().union(*list_of_dict) - set(dir(self)))
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
        if self.comm and self.comm.size > 1:
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
        if self.comm is None or self.comm.size == 1:
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

        self.local_start_time, self.local_start, self.local_n_samples = \
            self._get_local_start_time_start_and_n_samples()

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

        if not self.comm or self.comm.size == 1:
            assert len(info) == len(self.tod)
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
                    self._n_blocks_det, self._n_blocks_time)
                info = [info[s:s + n] for s, n, in zip(starts, nums)]

            info = comm_col.scatter(info, root)

        comm_row = comm_grid.Split(comm_grid.rank // self._n_blocks_time)
        info = comm_row.bcast(info, root_col)
        assert len(info) == len(self.tod)
        setattr(self, name, info)

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

        In any case the size of the returned array is ``self.tod.shape[1]``.

        This can be a costly operation; you should cache this result
        if you plan to use it in your code, instead of calling this
        method over and over again.

        Note for MPI-parallel codes: the times returned are only those of the
        local portion of the data. This means that

         * the size of the returned array is smaller than ``n_samples`` whenever
           there is more than one time-block
         * ``self.tod * self.get_times()`` is a meaningless but always
           allowed operation
        """
        if normalize:
            assert (
                not astropy_times
            ), "you cannot pass astropy_times=True *and* normalize=True"

            return ((np.arange(self.local_n_samples) + self.local_start)
                    / self.sampling_rate_hz)

        if astropy_times:
            assert isinstance(self.start_time, astropy.time.Time), (
                "to use astropy_times=True you must specify an astropy.time.Time "
                "object in Observation.__init__"
            )
            delta = astropy.time.TimeDelta(
                1.0 / self.sampling_rate_hz, format="sec", scale="tdb"
            )
            return (self.local_start_time
                    + np.arange(self.local_n_samples) * delta)

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
            t0 = self.local_start_time.cxcsec
        else:
            t0 = self.local_start_time

        return t0 + np.arange(self.local_n_samples) / self.sampling_rate_hz

    def get_det2ecl_quaternions(
        self,
        spin2ecliptic_quats: Spin2EclipticQuaternions,
        detector_quats,
        bore2spin_quat,
    ):
        """Return the detector-to-Ecliptic quaternions

        This function returns a ``(D, N, 4)`` tensor containing the
        quaternions that convert a vector in detector's coordinates
        into the frame of reference of the Ecliptic. The number of
        quaternions is equal to the number of samples hold in this
        observation.

        Given that the z axis in the frame of reference of a detector
        points along the main beam axis, this means that if you use
        these quaternions to rotate the vector `z = [0, 0, 1]`, you
        will end up with the sequence of vectors pointing towards the
        points in the sky (in Ecliptic coordinates) that are observed
        by the detector.

        This is a low-level method; you should usually call the method
        :meth:`.get_pointings`, which wraps this function to compute
        both the pointing direction and the polarization angle.

        See also the method :meth:`.get_ecl2det_quaternions`, which
        mirrors this one.

        """
        return np.array([spin2ecliptic_quats.get_detector_quats(
                detector_quat=detector_quat,
                bore2spin_quat=bore2spin_quat,
                time0=self.local_start_time,
                sampling_rate_hz=self.sampling_rate_hz,
                nsamples=self.local_n_samples,
            )
            for detector_quat in detector_quats])

    def get_ecl2det_quaternions(
        self,
        spin2ecliptic_quats: Spin2EclipticQuaternions,
        detector_quats,
        bore2spin_quat,
    ):
        """Return the Ecliptic-to-detector quaternions

        This function returns a ``(D, N, 4)`` matrix containing the ``N``
        quaternions of alla the ``D`` detectors
        that convert a vector in Ecliptic coordinates into
        the frame of reference of the detector itself. The number of
        quaternions is equal to the number of samples hold in this
        observation.

        This method is useful when you want to simulate how a point
        source is observed by the detector's beam: if you know the
        Ecliptic coordinates of the point sources, you can easily
        derive the location of the source with respect to the
        reference frame of the detector's beam.
        """

        quats = self.get_det2ecl_quaternions(
            spin2ecliptic_quats, detector_quats, bore2spin_quat
        )
        quats[..., 0:3] *= -1  # Apply the quaternion conjugate
        return quats

    def get_pointings(
        self,
        spin2ecliptic_quats: Spin2EclipticQuaternions,
        detector_quats,
        bore2spin_quat,
    ):
        """Return the time stream of pointings for the detector

        Given a :class:`Spin2EclipticQuaternions` and a quaternion
        representing the transformation from the reference frame of a
        detector to the boresight reference frame, compute a set of
        pointings for the detector that encompass the time span
        covered by this observation (i.e., starting from
        `self.start_time` and including `self.n_samples` pointings).

        The parameter `spin2ecliptic_quats` can be easily retrieved by
        the field `spin2ecliptic_quats` in a object of
        :class:`.Simulation` object, once the method
        :meth:`.Simulation.generate_spin2ecl_quaternions` is called.

        The parameter `detector_quats` is a stack of detector quaternions. For
        example, it can be

        - The stack of the field `quat` of an instance of the class
           :class:`.Detector`

        - If all you want to do is a simulation using a boresight
           direction, you can pass the value ``np.array([[0., 0., 0.,
           1.]])``, which represents the null rotation.

        The parameter `bore2spin_quat` is calculated through the class
        :class:`.Instrument`, which has the field ``bore2spin_quat``.
        If all you have is the angle β between the boresight and the
        spin axis, just pass ``quat_rotation_y(β)`` here.

        The return value is a ``(D x N × 3)`` tensor: the colatitude (in
        radians) is stored in column 0 (e.g., ``result[:, :, 0]``), the
        longitude (ditto) in column 1, and the polarization angle
        (ditto) in column 2. You can extract the three vectors using
        the following idiom::

            pointings = obs.get_pointings(...)
            # Extract the colatitude (theta), longitude (psi), and
            # polarization angle (psi) from pointings
            theta, phi, psi = [pointings[:, :, i] for i in (0, 1, 2)]

        """
        det2ecliptic_quats = self.get_det2ecl_quaternions(
            spin2ecliptic_quats, detector_quats, bore2spin_quat,
        )

        # Compute the pointing direction for each sample
        pointings_and_polangle = np.empty((self.n_detectors, self.n_samples, 3))
        all_compute_pointing_and_polangle(
            result_matrix=pointings_and_polangle, quat_matrix=det2ecliptic_quats
        )

        return pointings_and_polangle
