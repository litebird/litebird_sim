"""Flags of the samples in the TODs.

Flags are bitmasks of dtype :data:`FLAG_DTYPE`: zero means a good sample,
and each bit is a different reason why a sample is flagged. Modules that
flag samples set their own bit with ``|=``, and consumers select the
reasons to exclude with ``(flags & mask) != 0``.

``Observation.local_flags`` (shape ``(n_detectors, n_samples)``) and
``Observation.global_flags`` (shape ``(n_samples,)``) are ``None`` until
somebody writes them, meaning that all the samples are good: use
:func:`get_local_flags` and :func:`get_global_flags` to allocate them on
the first write.
"""

import numpy as np
import numpy.typing as npt

from .observations import Observation

FLAG_DTYPE = np.uint16

# Bits of the flags, one per reason
FLAG_PLANET = 1 << 0


def get_local_flags(obs: Observation) -> npt.NDArray:
    """Return ``obs.local_flags``, allocating it with zeros if it is ``None``"""
    if obs.local_flags is None:
        obs.local_flags = np.zeros((obs.n_detectors, obs.n_samples), dtype=FLAG_DTYPE)
    return obs.local_flags


def get_global_flags(obs: Observation) -> npt.NDArray:
    """Return ``obs.global_flags``, allocating it with zeros if it is ``None``"""
    if obs.global_flags is None:
        obs.global_flags = np.zeros(obs.n_samples, dtype=FLAG_DTYPE)
    return obs.global_flags
