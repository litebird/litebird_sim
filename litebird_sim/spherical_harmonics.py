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


    Attributes
    ----------
    values : np.ndarray
        The spherical harmonics coefficients, stored in a NumPy array of shape `(nstokes, ncoeff)`.
    lmax : int
        The maximum degree ℓ_max of the expansion.
    mmax : int, optional
        The maximum order m_max of the expansion. If None, it is set equal to `lmax`.
    nstokes : int
        The number of Stokes parameters (1 for intensity-only, 3 for polarization).

    Raises
    ------
    ValueError
        If `nstokes` is not 1 or 3.
        If the shape of `values` does not match the expected shape for given `lmax` and `mmax`.
    """

    values: np.ndarray
    lmax: int
    mmax: int = None
    nstokes: int = field(init=False)

    def __post_init__(self):
        """
        Initializes the `SphericalHarmonics` instance by validating and reshaping the input data.

        - If `mmax` is not provided, it is set equal to `lmax`.
        - If `values` is a tuple of three arrays, it is converted into a NumPy array.
        - Ensures `values` has shape `(nstokes, ncoeff)`, reshaping if necessary.
        - Validates that `nstokes` is either 1 or 3.
        - Checks that the shape of `values` matches the expected shape.

        Raises
        ------
        ValueError
            If `nstokes` is not 1 or 3.
            If `values` does not have the expected shape.
        """

        if self.mmax is None:
            self.mmax = self.lmax

        if isinstance(self.values, tuple):
            # if self.values is a 3-tuple containing three NumPy arrays
            # it gets converted in NumPy array
            self.values = np.array([self.values[i] for i in range(len(self.values))])

        if len(self.values.shape) == 1:
            self.values = np.reshape(self.values, (1, len(self.values)))

        self.nstokes = self.values.shape[0]
        if self.nstokes != 1 and self.nstokes != 3:
            raise ValueError(
                (
                    "The number of Stokes parameters in "
                    "SphericalHarmonics should be 1 or 3 instead of {}."
                ).format(self.nstokes)
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

    @property
    def num_of_alm_per_stokes(self):
        """
        Returns the number of a_ℓm coefficients per Stokes component.

        Returns
        -------
        int
            The number of coefficients per Stokes parameter.
        """
        return self.values.shape[1]

    @staticmethod
    def num_of_alm_from_lmax(lmax: int, mmax: Optional[int] = None) -> int:
        """
        Computes the number of a_ℓm coefficients for given ℓ_max and m_max.
        If `mmax` is not provided, it is set equal to `lmax`

        Parameters
        ----------
        lmax : int
            The maximum degree ℓ_max.
        mmax : int, optional
            The maximum order m_max. If None, it is set equal to `lmax`.

        Returns
        -------
        int
            The number of a_ℓm coefficients.

        Raises
        ------
        AssertionError
            If `mmax` is negative or greater than `lmax`.
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
        """
        Computes the expected shape of the a_ℓm array.

        Parameters
        ----------
        lmax : int
            The maximum degree ℓ_max.
        mmax : int, optional
            The maximum order m_max. If None, it is set equal to `lmax`.
        nstokes : int, default=3
            The number of Stokes parameters (1 for intensity-only, 3 for full polarization).

        Returns
        -------
        tuple[int, int]
            The expected shape `(nstokes, ncoeff)`, where `ncoeff` is the number of a_ℓm coefficients.
        """
        return nstokes, SphericalHarmonics.num_of_alm_from_lmax(lmax, mmax)

    def resize_alm(
        self,
        lmax_out: int,
        mmax_out: int = None,
        inplace: bool = False,
    ):
        """
        Resizes the spherical harmonics coefficients, either truncating or padding them with zeros.

        If `inplace` is False (default), returns a new instance of `SphericalHarmonics` with the adjusted size.
        Otherwise, modifies the current object in place.

        Parameters
        ----------
        lmax_out : int
            The new maximum degree ℓ_max.
        mmax_out : int, optional
            The new maximum order m_max. If None, it is set equal to `lmax_out`.
        inplace : bool, default=False
            If True, modifies the current object instead of returning a new one.

        Returns
        -------
        SphericalHarmonics or None
            A new `SphericalHarmonics` instance with resized coefficients if `inplace` is False.
            Otherwise, modifies the current instance in place and returns None.

        Notes
        -----
        - The method resizes the coefficient array while preserving existing values.
        - Adapted from HealPy's spherical harmonics resizing implementation.
          https://github.com/healpy/healpy/blob/a57770262788bb72281d48fdfe427d1098898d53/lib/healpy/sphtfunc.py#L1436
        """

        lmax_in = self.lmax
        mmax_in = self.mmax

        if mmax_out is None:
            mmax_out = lmax_out

        res = np.zeros(
            (self.nstokes, SphericalHarmonics.num_of_alm_from_lmax(lmax_out, mmax_out)),
            dtype=self.values.dtype,
        )

        lmaxmin = min(lmax_in, lmax_out)
        mmaxmin = min(mmax_in, mmax_out)

        ofs_i, ofs_o = 0, 0
        for m in range(0, mmaxmin + 1):
            nval = lmaxmin - m + 1
            res[:, ofs_o : ofs_o + nval] = self.values[:, ofs_i : ofs_i + nval]
            ofs_i += lmax_in - m + 1
            ofs_o += lmax_out - m + 1

        if inplace:
            self.values = res
            self.lmax = lmax_out
            self.mmax = mmax_out
        else:
            return SphericalHarmonics(
                values=res,
                lmax=lmax_out,
                mmax=mmax_out,
            )
