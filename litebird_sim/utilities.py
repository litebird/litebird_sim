import os

import ducc0.misc

from .constants import NUM_THREADS_ENVVAR, NUMBA_NUM_THREADS_ENVVAR


def _hardware_threads() -> int:
    """Number of threads available to this process (affinity-aware on Linux)."""

    return ducc0.misc.available_hardware_threads()


def _compute_nthreads() -> int:
    if NUM_THREADS_ENVVAR in os.environ:
        return int(os.environ[NUM_THREADS_ENVVAR])
    return _hardware_threads()


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
    to the number of hardware threads available to this process.
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
