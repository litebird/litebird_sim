from dataclasses import dataclass, field

import numpy as np
import healpy as hp


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

    It also provides algebraic operations and I/O utilities compatible with Healpy.

    Attributes
    ----------
    values : np.ndarray
        The spherical harmonics coefficients, stored in a NumPy array of shape `(nstokes, ncoeff)`.
    lmax : int
        The maximum degree ℓ_max of the expansion.
    mmax : int, optional
        The maximum order m_max of the expansion. If None, it is set equal to `lmax`.
    nstokes : int
        The number of Stokes parameters (1 for intensity-only, 3 for TEB).

    Methods
    -------
    - num_of_alm_from_lmax(lmax, mmax=None)
        Returns the number of a_ℓm coefficients for given `lmax` and `mmax`.
    - lmax_from_num_of_alm(nalm, mmax=None)
        Infers `lmax` from the number of coefficients and `mmax`.
    - alm_array_size(lmax, mmax=None, nstokes=3)
        Computes the expected shape of the coefficient array.
    - alm_l_array(lmax, mmax=None)
        Returns an array mapping coefficient indices to their ℓ values.

    Raises
    ------
    ValueError
        If `nstokes` is not 1 or 3.
        If the shape of `values` does not match the expected shape for given `lmax` and `mmax`.

    Arithmetic
    ----------
    The following operations are supported:
    - `+`, `-` between two SphericalHarmonics (same `lmax`, `mmax`, `nstokes`)
    - `*` with scalar or Stokes-vector (array of shape `(nstokes,)`)
    - `.convolve(f_ell)` applies a filter f_ell(ℓ) or f_ell^i(ℓ) per Stokes

    I/O
    ---
    - `.write_fits(filename, overwrite=True)`
        Saves the coefficients in a Healpy-compatible `.fits` file using `hp.write_alm`.
    - `.read_fits(filename)`
        Loads from a `.fits` file written by Healpy. Automatically supports 1 or 3 Stokes.

    Example
    -------
    >>> from litebird_sim import SphericalHarmonics
    >>> import numpy as np
    >>> alm = np.ones((3, 55))  # (nstokes=3, nalm=55)
    >>> sh = SphericalHarmonics(alm, lmax=9)
    >>> sh_convolved = sh.convolve(np.arange(10))
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
    def num_of_alm_from_lmax(lmax: int, mmax: int | None = None) -> int:
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
    def lmax_from_num_of_alm(nalm: int, mmax: int | None = None) -> int:
        """
        Returns the lmax corresponding to a given array size.

        Parameters
        ----------
        nalm : int
            Number of alm coefficients.
        mmax : int, optional
            Maximum m. If None, assumes full alm with mmax = lmax.

        Returns
        -------
        int
            The corresponding lmax, or -1 if `nalm` is not consistent.
        """
        if mmax is not None and mmax >= 0:
            x = (2 * nalm + mmax**2 - mmax - 2) / (2 * mmax + 2)
        else:
            x = (-3 + np.sqrt(1 + 8 * nalm)) / 2

        if not np.isclose(x, np.round(x)):
            return -1
        return int(round(x))

    @staticmethod
    def alm_array_size(
        lmax: int, mmax: int | None = None, nstokes: int = 3
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

    @staticmethod
    def alm_l_array(lmax: int, mmax: int | None = None) -> np.ndarray:
        """
        Return the ℓ values corresponding to each a_{ℓm} coefficient in Healpy's flattened alm format.

        This function reproduces the ℓ-indexing of Healpy's `alm` array layout, assuming the coefficients
        are stored in the usual `(ℓ, m)` ordering with ℓ ≥ m.

        Parameters
        ----------
        lmax : int
            Maximum multipole ℓ included in the harmonic expansion.
        mmax : int, optional
            Maximum azimuthal number m. If None, defaults to `lmax`.

        Returns
        -------
        np.ndarray
            A 1D array of length equal to the number of alm coefficients, where each element
            contains the corresponding ℓ value (degree) of that coefficient.

        Examples
        --------
        >>> SphericalHarmonics.alm_l_array(3)
        array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3])
        """
        if mmax is None:
            mmax = lmax

        l_arr = []
        for m in range(mmax + 1):
            l_arr.extend(range(m, lmax + 1))
        return np.array(l_arr, dtype=int)

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

    # ============================================================
    # Algebraic Operations
    # ============================================================

    def is_consistent(self, other: "SphericalHarmonics") -> bool:
        """Check if two SphericalHarmonics objects are compatible for algebraic operations."""
        return (
            self.lmax == other.lmax
            and self.mmax == other.mmax
            and self.nstokes == other.nstokes
        )

    def __add__(self, other):
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Can only add another SphericalHarmonics object")

        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )

        return SphericalHarmonics(
            values=self.values + other.values, lmax=self.lmax, mmax=self.mmax
        )

    def __iadd__(self, other):
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Can only add another SphericalHarmonics object")

        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )

        self.values += other.values
        return self

    def __mul__(self, other: float | np.ndarray):
        """
        Supports:
        - scalar multiplication: SH * A
        - stokes-vector multiplication: SH * [A_T, A_E, A_B]
        """
        if isinstance(other, (float, int, complex)):
            new_values = self.values * other
        elif isinstance(other, np.ndarray):
            if other.shape != (self.nstokes,):
                raise ValueError(
                    f"Stokes multiplier must have shape ({self.nstokes},), got {other.shape}"
                )
            new_values = self.values * other[:, None]
        else:
            raise TypeError("Can only multiply by scalar or Stokes vector")

        return SphericalHarmonics(values=new_values, lmax=self.lmax, mmax=self.mmax)

    def __rmul__(self, other: float | np.ndarray):
        return self.__mul__(other)

    def convolve(self, f_ell: np.ndarray | list[np.ndarray]) -> "SphericalHarmonics":
        """
        Apply a beam or filter to the SH coefficients.

        Parameters
        ----------
        f_ell : np.ndarray or list[np.ndarray]
            The ℓ-dependent filter(s). Must be of shape (lmax+1,) or (nstokes, lmax+1)

        Returns
        -------
        SphericalHarmonics
            A new SphericalHarmonics object with filtered coefficients.
        """
        l_arr = self.alm_l_array(self.lmax, self.mmax)

        if isinstance(f_ell, np.ndarray):
            if f_ell.ndim == 1:
                kernel = np.broadcast_to(f_ell[l_arr], self.values.shape)
            elif f_ell.ndim == 2 and f_ell.shape[0] == self.nstokes:
                kernel = np.stack([f[l_arr] for f in f_ell])
            else:
                raise ValueError(f"Invalid shape for f_ell: {f_ell.shape}")
        elif isinstance(f_ell, list):
            if len(f_ell) != self.nstokes:
                raise ValueError(f"Expected {self.nstokes} filters, got {len(f_ell)}")
            kernel = np.stack([np.asarray(f)[l_arr] for f in f_ell])
        else:
            raise TypeError("f_ell must be a numpy array or a list of numpy arrays")

        return SphericalHarmonics(
            values=self.values * kernel, lmax=self.lmax, mmax=self.mmax
        )

    def copy(self):
        """Returns a deep copy of this SphericalHarmonics object."""
        return SphericalHarmonics(
            values=self.values.copy(), lmax=self.lmax, mmax=self.mmax
        )

    def __sub__(self, other):
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Subtraction requires another SphericalHarmonics object")
        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )
        return SphericalHarmonics(
            values=self.values - other.values, lmax=self.lmax, mmax=self.mmax
        )

    def __isub__(self, other):
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Subtraction requires another SphericalHarmonics object")
        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )
        self.values -= other.values
        return self

    def __eq__(self, other):
        if not isinstance(other, SphericalHarmonics):
            raise ValueError("Can only compare with another SphericalHarmonics object.")
        return self.is_consistent(other) and np.array_equal(self.values, other.values)

    def allclose(self, other, rtol=1e-5, atol=1e-8):
        """Compares SH values with tolerance."""
        if not isinstance(other, SphericalHarmonics):
            raise ValueError("Can only compare with another SphericalHarmonics object.")
        return self.is_consistent(other) and np.allclose(
            self.values, other.values, rtol=rtol, atol=atol
        )

    def write_fits(self, filename: str, overwrite: bool = True):
        """
        Save the SphericalHarmonics object to a Healpy-compatible .fits file.

        Parameters
        ----------
        filename : str
            The path to the output .fits file.
        overwrite : bool
            Whether to overwrite an existing file.
        """
        hp.write_alm(
            filename,
            self.values if self.nstokes == 3 else self.values[0],
            lmax=self.lmax,
            mmax=self.mmax,
            overwrite=overwrite,
            mmax_in=self.mmax,
        )

    @staticmethod
    def read_fits(filename: str) -> "SphericalHarmonics":
        """
        Load a SphericalHarmonics object from a Healpy .fits file using only hp.read_alm.

        This supports both 1-Stokes and 3-Stokes files written using hp.write_alm.

        Parameters
        ----------
        filename : str
            Path to the .fits file.

        Returns
        -------
        SphericalHarmonics
        """
        try:
            # Try to read all three Stokes components (T=1, E=2, B=3)
            T, mmax = hp.read_alm(filename, hdu=1, return_mmax=True)

            values = np.array(
                [
                    T,
                    hp.read_alm(filename, hdu=2),
                    hp.read_alm(filename, hdu=3),
                ]
            )
            nalm = values.shape[1]
        except Exception:
            # Fallback to a single-Stokes file
            alm, mmax = hp.read_alm(filename, return_mmax=True)
            values = alm[np.newaxis, :]
            nalm = values.shape[1]

        # Compute lmax from nalm and mmax
        lmax = SphericalHarmonics.lmax_from_num_of_alm(nalm, mmax)

        return SphericalHarmonics(values=values, lmax=lmax, mmax=mmax)


@dataclass
class HealpixMap:
    """
    A small container class for HEALPix maps, with shape checks and algebra.

    This class stores one- or three-component HEALPix maps and their NSIDE
    consistently. Maps are always stored as a 2D NumPy array of shape
    ``(nstokes, npix)``, even if ``nstokes == 1``.

    It also provides basic algebraic operations and a few static helpers
    for working with HEALPix geometry, without depending on ``healpy``.

    Attributes
    ----------
    values : np.ndarray
        Map values, stored as a NumPy array of shape ``(nstokes, npix)``.
        If a 1D array is passed, it is promoted to shape ``(1, npix)``.
        If a tuple of arrays is passed (e.g. (I, Q, U)), it is stacked
        along the first axis.
    nside : int
        HEALPix NSIDE resolution parameter. Must be a power of two, otherwise
        an AssertionError is raised by :meth:`HealpixMap.nside_to_npix`.
    nest : bool, optional
        If True, the map is in NESTED ordering; if False, in RING (default).
        This is just metadata here, no indexing operations are performed.
    nstokes : int
        Number of Stokes parameters (1 for intensity-only, 3 for IQU).

    Static helpers
    --------------
    - :meth:`HealpixMap.nside_to_npix`
    - :meth:`HealpixMap.npix_to_nside`
    - :meth:`HealpixMap.is_npix_ok`
    - :meth:`HealpixMap.nside_to_pixel_solid_angle_sterad`
    - :meth:`HealpixMap.nside_to_resolution_rad`

    Arithmetic
    ----------
    The following operations are supported:

    - `+`, `-` between two HealpixMap (same `nside`, `nest`, `nstokes`)
    - `*` with scalar or Stokes-vector (array of shape `(nstokes,)`)
    - in-place variants `+=`, `-=`
    - equality checks via `==` or `.allclose(...)`
    """

    values: np.ndarray
    nside: int
    nest: bool = False
    nstokes: int = field(init=False)

    def __post_init__(self):
        """
        Initialize the `HealpixMap` instance by validating and reshaping input.

        - Validates `nside` using :meth:`HealpixMap.nside_to_npix`.
        - If `values` is a tuple of arrays, it is converted to a NumPy array.
        - If `values` is 1D, it is reshaped to `(1, npix)`.
        - Sets `nstokes` from the first dimension of `values`.
        - Validates that `nstokes` is either 1 or 3.
        - Checks that `npix` matches :meth:`HealpixMap.nside_to_npix(self.nside)`.

        Raises
        ------
        AssertionError
            If `nside` is not a valid HEALPix NSIDE value.
        ValueError
            If `nstokes` is not 1 or 3, or if the number of pixels does
            not match the NSIDE.
        """
        # Validate NSIDE (raises AssertionError if invalid)
        _ = HealpixMap.nside_to_npix(self.nside)

        # Convert tuple of arrays (e.g. (I, Q, U)) into a stacked ndarray
        if isinstance(self.values, tuple):
            self.values = np.array([self.values[i] for i in range(len(self.values))])

        # Ensure values have shape (nstokes, npix)
        if self.values.ndim == 1:
            self.values = self.values[np.newaxis, :]

        self.nstokes = self.values.shape[0]
        if self.nstokes not in (1, 3):
            raise ValueError(
                "The number of Stokes parameters in HealpixMap should be 1 or 3 "
                f"instead of {self.nstokes}."
            )

        npix = self.values.shape[1]
        expected_npix = HealpixMap.nside_to_npix(self.nside)
        if npix != expected_npix:
            raise ValueError(
                "Wrong number of pixels for HealpixMap: it is "
                f"{npix} instead of {expected_npix} for nside={self.nside}."
            )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def npix(self) -> int:
        """Return the number of pixels in the map."""
        return self.values.shape[1]

    # ------------------------------------------------------------------
    # Static HEALPix helpers (no healpy dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def nside_to_npix(nside: int) -> int:
        """Return the number of pixels in a Healpix map with the specified NSIDE.

        If the value of `nside` is not valid (power of two), an
        `AssertionError` exception is raised.

        .. doctest::

            >>> HealpixMap.nside_to_npix(1)
            12
        """
        assert 2 ** np.log2(nside) == nside, f"Invalid value for NSIDE: {nside}"
        return 12 * nside * nside

    @staticmethod
    def is_npix_ok(num_of_pixels) -> bool:
        """Return True or False whenever num_of_pixels is a valid number.

        The function checks if the number of pixels provided as an
        argument conforms to the Healpix standard, which means that the
        number must be in the form 12 * NSIDE^2.

        .. doctest::

            >>> HealpixMap.is_npix_ok(48)
            True
            >>> HealpixMap.is_npix_ok(49)
            False
        """
        nside = np.sqrt(np.asarray(num_of_pixels) / 12.0)
        return nside == np.floor(nside)

    @staticmethod
    def npix_to_nside(num_of_pixels: int) -> int:
        """Return NSIDE for a Healpix map containing `num_of_pixels` pixels.

        If the number of pixels does not conform to the Healpix standard,
        an `AssertionError` exception is raised.

        .. doctest::

            >>> HealpixMap.npix_to_nside(48)
            2
        """
        assert HealpixMap.is_npix_ok(
            num_of_pixels
        ), f"Invalid number of pixels: {num_of_pixels}"
        return int(np.sqrt(num_of_pixels / 12))

    @staticmethod
    def nside_to_pixel_solid_angle_sterad(nside: int) -> float:
        """Return the value of the solid angle of a pixel.

        The result is exact, as all pixels in a Healpix map have the same area.

        The result is in steradians.
        """
        return 4 * np.pi / HealpixMap.nside_to_npix(nside)

    @staticmethod
    def nside_to_resolution_rad(nside: int) -> float:
        """Return an approximated resolution of a Healpix map, given its NSIDE.

        The value is the square root of the pixel area (which is measured in
        steradians); see :meth:`HealpixMap.nside_to_pixel_solid_angle_sterad`.

        The result is an angle in radians.
        """
        return np.sqrt(HealpixMap.nside_to_pixel_solid_angle_sterad(nside))

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------

    def is_consistent(self, other: "HealpixMap") -> bool:
        """Check if two HealpixMap objects are compatible for algebraic operations.

        Two maps are considered consistent if they share the same
        `nside`, `nest`, and `nstokes`.
        """
        return (
            self.nside == other.nside
            and self.nest == other.nest
            and self.nstokes == other.nstokes
        )

    def __add__(self, other: "HealpixMap") -> "HealpixMap":
        """Add two maps with the same geometry."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Can only add another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        return HealpixMap(
            values=self.values + other.values,
            nside=self.nside,
            nest=self.nest,
        )

    def __iadd__(self, other: "HealpixMap") -> "HealpixMap":
        """In-place addition of two maps with the same geometry."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Can only add another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        self.values += other.values
        return self

    def __sub__(self, other: "HealpixMap") -> "HealpixMap":
        """Subtract two maps with the same geometry."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Subtraction requires another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        return HealpixMap(
            values=self.values - other.values,
            nside=self.nside,
            nest=self.nest,
        )

    def __isub__(self, other: "HealpixMap") -> "HealpixMap":
        """In-place subtraction of two maps with the same geometry."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Subtraction requires another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        self.values -= other.values
        return self

    def __mul__(self, other: float | np.ndarray) -> "HealpixMap":
        """
        Multiply a map by a scalar or a Stokes vector.

        Supported cases
        ---------------
        - scalar multiplication: ``map * A`` where ``A`` is a float/int/complex
        - Stokes-vector multiplication: ``map * [A_I, A_Q, A_U]``

        Parameters
        ----------
        other : float, int, complex or np.ndarray
            Either a scalar, or an array of shape ``(nstokes,)``.

        Returns
        -------
        HealpixMap
            A new HealpixMap with scaled values.

        Raises
        ------
        TypeError
            If `other` has an unsupported type.
        ValueError
            If `other` is an array with invalid shape.
        """
        if isinstance(other, (float, int, complex)):
            new_values = self.values * other
        elif isinstance(other, np.ndarray):
            if other.shape != (self.nstokes,):
                raise ValueError(
                    f"Stokes multiplier must have shape ({self.nstokes},), "
                    f"got {other.shape}"
                )
            new_values = self.values * other[:, None]
        else:
            raise TypeError("Can only multiply by scalar or Stokes vector")

        return HealpixMap(
            values=new_values,
            nside=self.nside,
            nest=self.nest,
        )

    def __rmul__(self, other: float | np.ndarray) -> "HealpixMap":
        """Right-multiplication is the same as left-multiplication."""
        return self.__mul__(other)

    # ------------------------------------------------------------------
    # Comparison and utilities
    # ------------------------------------------------------------------

    def copy(self) -> "HealpixMap":
        """Return a deep copy of this HealpixMap object."""
        return HealpixMap(
            values=self.values.copy(),
            nside=self.nside,
            nest=self.nest,
        )

    def __eq__(self, other: object) -> bool:
        """
        Exact equality: same geometry and identical pixel values.

        Note
        ----
        This uses exact `np.array_equal` for the data, so it is usually
        better to use :meth:`allclose` when comparing floating-point maps.
        """
        if not isinstance(other, HealpixMap):
            raise ValueError("Can only compare with another HealpixMap object.")
        return self.is_consistent(other) and np.array_equal(self.values, other.values)

    def allclose(self, other: "HealpixMap", rtol=1e-5, atol=1e-8) -> bool:
        """
        Compare map values with tolerance.

        Parameters
        ----------
        other : HealpixMap
            Another map to compare with.
        rtol : float, default=1e-5
            Relative tolerance.
        atol : float, default=1e-8
            Absolute tolerance.

        Returns
        -------
        bool
            True if geometry is consistent and the pixel values are close.
        """
        if not isinstance(other, HealpixMap):
            raise ValueError("Can only compare with another HealpixMap object.")
        return self.is_consistent(other) and np.allclose(
            self.values, other.values, rtol=rtol, atol=atol
        )


def interpolate_alm(
    alms: SphericalHarmonics,
    locations: np.ndarray (:,2)

    ):

    return T[:],Q[:],U[:] 