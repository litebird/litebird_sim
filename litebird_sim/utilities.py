import os

import numpy as np
from numba import njit

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


# cos and sin on a grid of [0, 2π), for cos_sin
_TRIG_N = 4096
_TRIG_STEP = 2 * np.pi / _TRIG_N
_TRIG_COS = np.cos(np.arange(_TRIG_N) * _TRIG_STEP)
_TRIG_SIN = np.sin(np.arange(_TRIG_N) * _TRIG_STEP)


@njit(inline="always")
def cos_sin(x: float) -> tuple[float, float]:
    """Return ``(cos(x), sin(x))``, for use inside numba kernels.

    The angle is reduced to [0, 2π), then cos and sin come from the grid
    point below it and a Taylor series of the remainder (< 2π/4096), as in
    pysanepic. Numba without SVML calls a scalar libm: this is ~5-9 times
    faster on Galileo100, ~2 times on Apple M3. Accuracy is ~1e-16, plus
    ~|x|·1e-16 from the range reduction.
    """
    x = x - 2 * np.pi * np.floor(x * (1 / (2 * np.pi)))
    k = min(int(x * (1 / _TRIG_STEP)), _TRIG_N - 1)
    d = x - k * _TRIG_STEP
    d2 = d * d
    cd = 1.0 - d2 * (0.5 - d2 / 24.0)
    sd = d * (1.0 - d2 * (1.0 / 6.0 - d2 / 120.0))
    return (
        _TRIG_COS[k] * cd - _TRIG_SIN[k] * sd,
        _TRIG_SIN[k] * cd + _TRIG_COS[k] * sd,
    )
