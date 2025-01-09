# -*- encoding: utf-8 -*-

from dataclasses import dataclass

import numpy as np


@dataclass
class SphericalHarmonics:
    """
    A class that wraps spherical harmonics coefficients

    The convention used in libraries like HealPy is to keep the a_ℓm coefficients
    of a spherical harmonic expansion in a plain NumPy array. However, this is
    ambiguous because it is not possible to uniquely determine the value of
    ℓ_max and m_max from the size of the array (unless you assume that ℓ_max == m_max).

    This small data class keeps the array and the values `l_max` and `m_max` together.
    """

    values: np.ndarray
    l_max: int
    m_max: int
