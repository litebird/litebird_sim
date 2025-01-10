# -*- encoding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional

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
    The shape of `values` is *always* ``(nstokes, ncoeff)``, even if ``nstokes == 1``.
    """

    values: np.ndarray
    lmax: int
    mmax: int = None
    nstokes: int = field(init=False)

    def __post_init__(self):
        if not self.mmax:
            self.mmax = self.lmax

        if len(self.values.shape) == 1:
            np.reshape(self.values, (1, -1))

        self.nstokes = self.values.shape[0]
        if self.nstokes != 1 and self.nstokes != 3:
            raise ValueError(
                "The number of Stokes parameters in SphericalHarmonics should be 1 or 3"
            )

        expected_shape = SphericalHarmonics.alm_array_size(
            self.lmax, self.mmax, self.nstokes
        )

        if not self.values.shape == expected_shape:
            raise ValueError(
                (
                    "Wrong shape for the a_ℓm array: it is {actual} instead of {expected}"
                ).format(actual=self.values.shape, expected=expected_shape)
            )

    @staticmethod
    def num_of_alm_coefficients(lmax: int, mmax: Optional[int] = None) -> int:
        """Given a value for ℓ_max and m_max, return the size of the array a_ℓm

        If `mmax` is not provided, it is set equal to `lmax`
        """
        if mmax is None:
            mmax = lmax
        else:
            assert mmax >= 0
            assert mmax <= lmax

        return mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1

    @staticmethod
    def alm_array_size(
        lmax: int, mmax: Optional[int] = None, nstokes: int = 3
    ) -> tuple[int, int]:
        return nstokes, SphericalHarmonics.num_of_alm_coefficients(lmax, mmax)
