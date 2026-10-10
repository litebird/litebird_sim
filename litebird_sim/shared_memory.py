import atexit
from typing import TYPE_CHECKING, cast
import numpy as np
import numpy.typing as npt

from .mpi import MPI_ENABLED

# Import mpi4py only if MPI is enabled: importing `mpi4py.MPI` initializes MPI,
# which must not happen when the user sets LITEBIRD_SIM_MPI=0
if MPI_ENABLED or TYPE_CHECKING:
    from mpi4py import MPI
    from mpi4py.MPI import Intracomm


class SharedMemoryManager:
    """Manages MPI shared-memory communicators and window allocations

    This manager splits a base MPI communicator into a node-level
    shared-memory communicator and allocates MPI window-backed shared NumPy
    arrays. The windows are freed when the interpreter exits.

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
        if not MPI_ENABLED:
            raise RuntimeError(
                "Shared memory allocation requires MPI, but MPI is not enabled "
                "(mpi4py is not installed or LITEBIRD_SIM_MPI is set to 0)"
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

        # MPI requires the windows to be freed before MPI_Finalize, which mpi4py
        # calls when the interpreter exits: otherwise, some MPI implementations
        # (e.g., Intel MPI) hang at exit
        atexit.register(self._free_at_exit)

    def _free_at_exit(self) -> None:
        if not MPI.Is_finalized():
            self.free_shared_arrays_all()

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
        array = np.ndarray(shape=size, dtype=dtype, buffer=cast(memoryview, buf))

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

    def free_shared_arrays_all(self) -> None:
        """Frees all allocated shared-memory MPI windows and clears manager
        state.
        """
        for comm, wins in self._list_windows.items():
            for win in wins:
                win.Free()
        self._list_windows = {}
        self._list_arrays = {}
