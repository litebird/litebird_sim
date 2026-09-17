from typing import cast
import numpy as np
import numpy.typing as npt

try:
    from mpi4py import MPI
    from mpi4py.MPI import Intracomm
except ImportError:
    pass


class SharedMemoryManager:
    """Manages MPI shared-memory communicators, window allocations, and tree
    group reductions.

    This manager splits a base MPI communicator into a node-level
    shared-memory communicator and allocates MPI window-backed shared NumPy
    arrays. It also splits the node-level communicator into a tree group
    sub-communicators to orchestrate sequential accumulations and group-wise
    reductions within each node.

    Parameters
    ----------
    base_comm : Intracomm
        The base MPI communicator (typically `MPI.COMM_WORLD`)
    node_root : int, optional
        The designated root rank within the node-level shared memory
        communicator. By default `0`

    Attributes
    ----------
    base_comm : Intracomm
        The base MPI communicator
    node_comm : Intracomm
        The node-level shared-memory MPI communicator
    node_rank : int
        The process rank within the node-level communicator
    node_size : int
        The total number of processes on the current node
    node_root : int
        The root rank on the current node communicator
    list_windows : dict[int, list[MPI.Win]]
        Tracks allocated shared-memory MPI windows mapped by communicator handle
    list_arrays : dict[int, list[npt.NDArray]]
        Tracks allocated shared-memory NumPy array views mapped by
        communicator handle
    """

    def __init__(
        self,
        base_comm: "Intracomm",
        node_root: int = 0,
    ) -> None:
        if "MPI" not in globals():
            raise ImportError(
                "mpi4py is not installed or enabled, but shared memory allocation was requested."
            )

        self._base_comm = base_comm
        self._node_comm: Intracomm = cast(
            Intracomm, self._base_comm.Split_type(MPI.COMM_TYPE_SHARED)
        )
        self._node_rank = self._node_comm.rank
        self._node_size = self._node_comm.size
        self._node_root = node_root

        # List of MPI shared memory windows
        self._list_windows: dict[int, list[MPI.Win]] = {}
        self._list_arrays: dict[int, list[npt.NDArray]] = {}

    @property
    def base_comm(self) -> "Intracomm":
        """The base global MPI communicator"""
        return self._base_comm

    @property
    def node_comm(self) -> "Intracomm":
        """The node-level shared-memory MPI communicator"""
        return self._node_comm

    @property
    def node_rank(self) -> int:
        """The process rank within the node-level communicator"""
        return self._node_rank

    @property
    def node_size(self) -> int:
        """The total number of processes on the current node"""
        return self._node_size

    @property
    def node_root(self) -> int:
        """The root rank on the current node"""
        return self._node_root

    @property
    def list_windows(self) -> dict:
        """The dictionary mapping communicators to list of allocated
        shared-memory windows for that communicator
        """
        return self._list_windows

    @property
    def list_arrays(self) -> dict:
        """The dictionary mapping communicators to list of allocated
        shared-memory arrays for that communicator
        """
        return self._list_arrays

    def alloc_shared_comm(
        self,
        size: int,
        dtype: npt.DTypeLike,
        comm: "Intracomm",
        comm_root: int = 0,
    ) -> tuple[npt.NDArray, "MPI.Win"]:
        """Allocates a shared-memory MPI window-backed 1D NumPy array for a
        communicator.
        """
        dtype = np.dtype(dtype)
        dtype_bytes = dtype.itemsize
        arr_bytes = size * dtype_bytes if comm.rank == comm_root else 0

        win = MPI.Win.Allocate_shared(
            arr_bytes,
            dtype_bytes,
            comm=comm,
        )
        buf, _ = win.Shared_query(rank=comm_root)
        # np.ndarray provides the view, it doesn't owns the memory
        array = np.ndarray(shape=size, dtype=dtype, buffer=buf)

        handle = comm.handle
        if handle not in self._list_windows:
            self._list_windows[handle] = []
        if handle not in self._list_arrays:
            self._list_arrays[handle] = []

        self._list_windows[handle].append(win)
        self._list_arrays[handle].append(array)
        return array, win

    def alloc_shared_node(
        self,
        size: int,
        dtype: npt.DTypeLike,
    ) -> tuple[npt.NDArray, "MPI.Win"]:
        """Allocates a shared-memory MPI window-backed 1D NumPy array for the
        node-level communicator
        """
        return self.alloc_shared_comm(
            size=size,
            dtype=dtype,
            comm=self.node_comm,
            comm_root=self.node_root,
        )

    def alloc_shared_zeros_comm(
        self,
        size: int,
        dtype: npt.DTypeLike,
        comm: "Intracomm",
        comm_root: int = 0,
    ):
        """Allocates a shared-memory MPI window-backed 1D NumPy array for a
        communicator, initialized to zeros.
        """
        array, win = self.alloc_shared_comm(
            size=size,
            dtype=dtype,
            comm=comm,
            comm_root=comm_root,
        )

        if comm.rank == 0:
            array[:] = 0

        return array, win

    def alloc_shared_zeros_node(
        self,
        size: int,
        dtype: npt.DTypeLike,
    ):
        """Allocates a shared-memory MPI window-backed 1D NumPy array for the
        node-level communicator, initialized to zeros.
        """
        return self.alloc_shared_zeros_comm(
            size=size,
            dtype=dtype,
            comm=self.node_comm,
            comm_root=self.node_root,
        )

    def alloc_shared_ones_comm(
        self,
        size: int,
        dtype: npt.DTypeLike,
        comm: "Intracomm",
        comm_root: int = 0,
    ):
        """Allocates a shared-memory MPI window-backed 1D NumPy array for a
        communicator, initialized to ones.
        """
        array, win = self.alloc_shared_comm(
            size=size,
            dtype=dtype,
            comm=comm,
            comm_root=comm_root,
        )

        if comm.rank == 0:
            array[:] = 1

        return array, win

    def alloc_shared_ones_node(
        self,
        size: int,
        dtype: npt.DTypeLike,
    ):
        """Allocates a shared-memory MPI window-backed 1D NumPy array for the
        node-level communicator, initialized to ones.
        """
        return self.alloc_shared_ones_comm(
            size=size,
            dtype=dtype,
            comm=self.node_comm,
            comm_root=self.node_root,
        )

    def fence_comm_all(self, comm: "Intracomm", assertion: int = 0) -> None:
        """Call MPI.Win.Fence on all windows allocated on the given
        communicator.
        """
        handle = comm.handle
        if handle in self._list_windows:
            for win in self._list_windows[handle]:
                win.Fence(assertion)

    def free_shared_arrays_all(self) -> None:
        """Frees all allocated shared-memory MPI windows and clears manager
        state.
        """
        for comm, wins in self._list_windows.items():
            for win in wins:
                win.Free()
        self._list_windows = {}
        self._list_arrays = {}

    def free_shared_arrays_comm(self, comm: "Intracomm") -> None:
        """Frees all shared-memory MPI windows allocated for a specific
        communicator.
        """
        handle = comm.handle
        if handle in self._list_windows:
            for win in self._list_windows[handle]:
                win.Free()
            del self._list_windows[handle]
        if handle in self._list_arrays:
            del self._list_arrays[handle]

    def free_shared_array(self, comm: "Intracomm", win: "MPI.Win") -> None:
        """Frees a specific shared-memory MPI window and removes its associated
        array view and window from the manager's tracking lists.
        """
        handle = comm.handle
        if handle in self._list_windows and win in self._list_windows[handle]:
            idx = self._list_windows[handle].index(win)
            win.Free()
            self._list_windows[handle].pop(idx)
            self._list_arrays[handle].pop(idx)
            if not self._list_windows[handle]:
                del self._list_windows[handle]
            if not self._list_arrays[handle]:
                del self._list_arrays[handle]
