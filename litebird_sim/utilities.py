import os

from .constants import NUM_THREADS_ENVVAR, NUMBA_NUM_THREADS_ENVVAR


def _compute_nthreads() -> int:
    if NUM_THREADS_ENVVAR in os.environ:
        return int(os.environ[NUM_THREADS_ENVVAR])
    # No explicit thread count was requested. Default to a single thread
    # rather than every hardware thread available to the process: MPI/OpenMP
    # jobs are expected to set NUM_THREADS_ENVVAR (OMP_NUM_THREADS) per rank,
    # so this fallback only matters for un-configured runs (e.g. a laptop),
    # where using every core by default would otherwise silently starve
    # other processes on the machine.
    return 1


def _compute_numba_nthreads() -> int:
    if NUMBA_NUM_THREADS_ENVVAR in os.environ:
        return int(os.environ[NUMBA_NUM_THREADS_ENVVAR])
    return _compute_nthreads()


# Resolved once at import time (each MPI process reads its own environment),
# instead of re-reading the environment on every call. Mirrors the
# lbs.MPI_COMM_WORLD pattern: a single, process-wide value computed once and
# reused everywhere `nthreads` isn't explicitly overridden.
NUM_THREADS = _compute_nthreads()
NUMBA_NUM_THREADS = _compute_numba_nthreads()


def resolve_nthreads(nthreads: int | None) -> int:
    """Resolve the number of threads to use for ducc0/general parallel code.

    If ``nthreads`` is given explicitly, return it unchanged. Otherwise
    return :data:`NUM_THREADS`, which was resolved once at import time from
    :data:`.constants.NUM_THREADS_ENVVAR` (``OMP_NUM_THREADS``), falling back
    to ``1`` if the environment variable is not set.
    """

    if nthreads is not None:
        return nthreads
    return NUM_THREADS


def resolve_numba_nthreads(nthreads: int | None) -> int:
    """Resolve the number of threads to use for Numba.

    If ``nthreads`` is given explicitly, return it unchanged. Otherwise
    return :data:`NUMBA_NUM_THREADS`, resolved once at import time from
    :data:`.constants.NUMBA_NUM_THREADS_ENVVAR` (``NUMBA_NUM_THREADS``) if
    set, falling back to :data:`NUM_THREADS` (``OMP_NUM_THREADS``) so Numba
    and ducc0 agree by default while still allowing independent tuning.
    """

    if nthreads is not None:
        return nthreads
    return NUMBA_NUM_THREADS
