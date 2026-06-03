import os

from .constants import NUM_THREADS_ENVVAR


def resolve_nthreads(nthreads: int | None) -> int:
    """Resolve thread count from an explicit value or environment.

    If ``nthreads`` is not ``None``, return it unchanged. Otherwise read
    :data:`NUM_THREADS_ENVVAR` and fall back to 0 when the variable is unset.
    """

    if nthreads is not None:
        return nthreads

    return int(os.environ.get(NUM_THREADS_ENVVAR, 0))
