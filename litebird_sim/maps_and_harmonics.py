from dataclasses import dataclass, field
from typing import Any, Tuple, Optional

import numpy as np
import healpy as hp

from .units import Units
from .coordinates import CoordinateSystem

import ducc0.sht as sht
import ducc0.healpix as dh


# ======================================================================
# SphericalHarmonics
# ======================================================================


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
        The spherical harmonics coefficients, stored in a NumPy array of shape
        ``(nstokes, ncoeff)``.
    lmax : int
        The maximum degree ℓ_max of the expansion.
    mmax : int, optional
        The maximum order m_max of the expansion. If None, it is set equal to `lmax`.
    units : Units or None
        Physical units of the coefficients. If set to :data:``None``,
        the object is treated as unitless / unspecified.
    coordinates : CoordinateSystem or None
        Sky coordinate system of these coefficients (e.g. Galactic or Ecliptic).
        If ``None``, coordinates are unspecified.
    nstokes : int
        The number of Stokes parameters (1 for intensity-only, 3 for TEB).

    Arithmetic
    ----------
    The following operations are supported:
    - `+`, `-` between two SphericalHarmonics (same `lmax`, `mmax`, `nstokes`
      and compatible units / coordinates)
    - `*` with scalar or Stokes-vector (array of shape `(nstokes,)`)
      with optional units override: ``sh * a`` or ``sh.__mul__(a, units=...)``
    - `.convolve(f_ell)` applies a filter f_ell(ℓ) or f_ell^i(ℓ) per Stokes
      (units are preserved)
    """

    values: np.ndarray
    lmax: int
    mmax: int | None = None
    nstokes: int = field(init=False)
    units: Units | None = None
    coordinates: CoordinateSystem | None = None

    def __post_init__(self):
        """
        Initializes the `SphericalHarmonics` instance by validating and reshaping the input data.

        - If `mmax` is not provided, it is set equal to `lmax`.
        - If `values` is a tuple of three arrays, it is converted into a NumPy array.
        - Ensures `values` has shape `(nstokes, ncoeff)`, reshaping if necessary.
        - Validates that `nstokes` is either 1 or 3.
        - Checks that the shape of `values` matches the expected shape.
        - Normalizes `units` to either a :class:`Units` member or ``None``.
        - Normalizes `coordinates` to either a :class:`CoordinateSystem` member or ``None``.

        Raises
        ------
        ValueError
            If `nstokes` is not 1 or 3.
            If `values` does not have the expected shape.
            If `units` is not an instance of Units or None.
            If `coordinates` is not an instance of CoordinateSystem or None.
        """

        if self.mmax is None:
            self.mmax = self.lmax

        if isinstance(self.values, tuple):
            # If self.values is a tuple containing NumPy arrays, convert to a stacked array
            self.values = np.array([self.values[i] for i in range(len(self.values))])

        if self.values.ndim == 1:
            self.values = np.reshape(self.values, (1, self.values.size))

        # Normalize units
        if self.units is not None and not isinstance(self.units, Units):
            raise ValueError(
                f"units must be an instance of Units or None, got {type(self.units)!r}"
            )

        # Normalize coordinates
        if self.coordinates is not None and not isinstance(
            self.coordinates, CoordinateSystem
        ):
            raise ValueError(
                "coordinates must be an instance of CoordinateSystem or None, "
                f"got {type(self.coordinates)!r}"
            )

        self.nstokes = self.values.shape[0]
        if self.nstokes not in (1, 3):
            raise ValueError(
                "The number of Stokes parameters in SphericalHarmonics should be 1 "
                f"or 3 instead of {self.nstokes}."
            )

        expected_shape = SphericalHarmonics.alm_array_size(
            self.lmax, self.mmax, self.nstokes
        )

        if self.values.shape != expected_shape:
            raise ValueError(
                (
                    "Wrong shape for the a_ℓm array: it is {actual} instead of {expected}"
                ).format(actual=self.values.shape, expected=expected_shape)
            )

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @property
    def num_of_alm_per_stokes(self) -> int:
        """Returns the number of a_ℓm coefficients per Stokes component."""
        return self.values.shape[1]

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def num_of_alm_from_lmax(lmax: int, mmax: int | None = None) -> int:
        """
        Computes the number of a_ℓm coefficients for given ℓ_max and m_max.
        If `mmax` is not provided, it is set equal to `lmax`.
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

        Returns
        -------
        tuple[int, int]
            The expected shape `(nstokes, ncoeff)`.
        """
        return nstokes, SphericalHarmonics.num_of_alm_from_lmax(lmax, mmax)

    @staticmethod
    def alm_l_array(lmax: int, mmax: int | None = None) -> np.ndarray:
        """
        Return the ℓ values corresponding to each a_{ℓm} coefficient in Healpy's flattened alm format.
        """
        if mmax is None:
            mmax = lmax

        l_arr: list[int] = []
        for m in range(mmax + 1):
            l_arr.extend(range(m, lmax + 1))
        return np.array(l_arr, dtype=int)

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resize_alm(
        self,
        lmax_out: int,
        mmax_out: int | None = None,
        inplace: bool = False,
    ):
        """
        Resizes the spherical harmonics coefficients, either truncating or padding them with zeros.

        Units and coordinates are preserved.
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
            # units and coordinates unchanged
        else:
            return SphericalHarmonics(
                values=res,
                lmax=lmax_out,
                mmax=mmax_out,
                units=self.units,
                coordinates=self.coordinates,
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

    def _units_compatible(self, other: "SphericalHarmonics") -> bool:
        """
        Return True if units are compatible for algebraic operations.

        Rules
        -----
        - If either units is None → always compatible.
        - Otherwise, units must be exactly equal.
        """
        u1 = self.units
        u2 = other.units

        is_none_1 = u1 is None
        is_none_2 = u2 is None

        if is_none_1 or is_none_2:
            return True

        return u1 == u2

    def _coordinates_compatible(self, other: "SphericalHarmonics") -> bool:
        """
        Return True if coordinates are compatible for algebraic operations.

        Rules
        -----
        - If either coordinates is None → always compatible.
        - Otherwise, coordinates must be exactly equal.
        """
        c1 = self.coordinates
        c2 = other.coordinates

        if c1 is None or c2 is None:
            return True

        return c1 == c2

    # ----------------------
    # Addition / subtraction
    # ----------------------

    def __add__(self, other: "SphericalHarmonics") -> "SphericalHarmonics":
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Can only add another SphericalHarmonics object")

        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for addition: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for addition: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        # Result units: if self has a "real" units, keep it, otherwise inherit other.units
        if self.units is None:
            result_units = self.units
        else:
            result_units = other.units

        # Result coordinates: keep self if defined, otherwise inherit other
        result_coordinates = (
            self.coordinates if self.coordinates is not None else other.coordinates
        )

        return SphericalHarmonics(
            values=self.values + other.values,
            lmax=self.lmax,
            mmax=self.mmax,
            units=result_units,
            coordinates=result_coordinates,
        )

    def __iadd__(self, other: "SphericalHarmonics") -> "SphericalHarmonics":
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Can only add another SphericalHarmonics object")

        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for addition: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for addition: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        self.values += other.values

        # If self was unitless and other has a real units, inherit it
        if self.units is None and other.units is not None:
            self.units = other.units

        # If self had no coordinates and other has, inherit them
        if self.coordinates is None and other.coordinates is not None:
            self.coordinates = other.coordinates

        return self

    def __sub__(self, other: "SphericalHarmonics") -> "SphericalHarmonics":
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Subtraction requires another SphericalHarmonics object")

        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for subtraction: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for subtraction: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        if self.units is not None:
            result_units = self.units
        else:
            result_units = other.units

        result_coordinates = (
            self.coordinates if self.coordinates is not None else other.coordinates
        )

        return SphericalHarmonics(
            values=self.values - other.values,
            lmax=self.lmax,
            mmax=self.mmax,
            units=result_units,
            coordinates=result_coordinates,
        )

    def __isub__(self, other: "SphericalHarmonics") -> "SphericalHarmonics":
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Subtraction requires another SphericalHarmonics object")

        if not self.is_consistent(other):
            raise ValueError(
                "SphericalHarmonics objects must have matching lmax, mmax, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for subtraction: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for subtraction: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        self.values -= other.values

        if self.units is None and other.units is not None:
            self.units = other.units

        if self.coordinates is None and other.coordinates is not None:
            self.coordinates = other.coordinates

        return self

    # -------------
    # Multiplication
    # -------------

    def __mul__(
        self,
        other: float | np.ndarray,
        *,
        units: Units | None = None,
    ) -> "SphericalHarmonics":
        """
        Supports:
        - scalar multiplication: SH * A
        - stokes-vector multiplication: SH * [A_T, A_E, A_B]

        Parameters
        ----------
        other : float, int, complex or np.ndarray
            Either a scalar, or an array of shape ``(nstokes,)``.
        units : Units or None, keyword-only
            Units to assign to the resulting object. If ``None`` (default),
            the units of ``self`` are preserved.
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

        new_units = self.units if units is None else units

        return SphericalHarmonics(
            values=new_values,
            lmax=self.lmax,
            mmax=self.mmax,
            units=new_units,
            coordinates=self.coordinates,
        )

    def __rmul__(
        self,
        other: float | np.ndarray,
        *,
        units: Units | None = None,
    ) -> "SphericalHarmonics":
        """
        Right-multiplication: allow units override just like in __mul__.

        Normal uses like ``2 * sh`` cannot pass ``units=...`` because of Python
        syntax, but explicit calls to ``sh.__rmul__(2.0, units=...)`` are allowed.
        """
        return self.__mul__(other, units=units)

    # -------------
    # Convolution
    # -------------

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
            The units and coordinates are preserved.
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
            values=self.values * kernel,
            lmax=self.lmax,
            mmax=self.mmax,
            units=self.units,
            coordinates=self.coordinates,
        )

    # -------------
    # Copy / compare
    # -------------

    def copy(self) -> "SphericalHarmonics":
        """Returns a deep copy of this SphericalHarmonics object."""
        return SphericalHarmonics(
            values=self.values.copy(),
            lmax=self.lmax,
            mmax=self.mmax,
            units=self.units,
            coordinates=self.coordinates,
        )

    def __eq__(self, other: Any) -> bool:
        """
        Exact equality: same geometry, same units, same coordinates
        and identical coefficients.
        """
        if not isinstance(other, SphericalHarmonics):
            raise ValueError("Can only compare with another SphericalHarmonics object.")
        return (
            self.is_consistent(other)
            and self.units == other.units
            and self.coordinates == other.coordinates
            and np.array_equal(self.values, other.values)
        )

    def allclose(self, other: "SphericalHarmonics", rtol=1e-5, atol=1e-8) -> bool:
        """
        Compare SH values with tolerance.

        Units must be compatible (same, or one/both None/Units.None).
        Coordinates must also be compatible (same, or one/both None).
        """
        if not isinstance(other, SphericalHarmonics):
            raise ValueError("Can only compare with another SphericalHarmonics object.")

        if not self.is_consistent(other):
            return False

        if not self._units_compatible(other):
            return False

        if not self._coordinates_compatible(other):
            return False

        return np.allclose(self.values, other.values, rtol=rtol, atol=atol)

    # -------------
    # I/O
    # -------------

    def write_fits(self, filename: str, overwrite: bool = True):
        """
        Save the SphericalHarmonics object to a Healpy-compatible .fits file.

        Parameters
        ----------
        filename : str
            The path to the output .fits file.
        overwrite : bool
            Whether to overwrite an existing file.

        Notes
        -----
        The units and coordinates metadata are *not* written to the FITS
        file by this method and must be tracked separately if needed.
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

        Notes
        -----
        The returned object has units and coordinates set to ``None``,
        as this metadata is not stored/read here.
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

        return SphericalHarmonics(
            values=values,
            lmax=lmax,
            mmax=mmax,
            units=None,
            coordinates=None,
        )


# ======================================================================
# HealpixMap
# ======================================================================


@dataclass
class HealpixMap:
    """
    A small container class for HEALPix maps, with shape checks and algebra.

    This class stores one- or three-component HEALPix maps and their NSIDE
    consistently. Maps are always stored as a 2D NumPy array of shape
    ``(nstokes, npix)``, even if ``nstokes == 1``.

    If ``nside`` is not provided, it is inferred from the size of ``values``.

    Attributes
    ----------
    values : np.ndarray
        Map values, stored as a NumPy array of shape ``(nstokes, npix)``.
        If a 1D array is passed, it is promoted to shape ``(1, npix)``.
        If a tuple of arrays is passed (e.g. (I, Q, U)), it is stacked
        along the first axis.
    nside : int | None
        HEALPix NSIDE resolution parameter.
        If ``None``, it is automatically inferred from the last dimension of ``values``.
        If provided, it is checked against the size of ``values``.
    units : Units or None
        Physical units of the map. If set to :data:`Units.None` or ``None``,
        the map is treated as unitless / unspecified.
    coordinates : CoordinateSystem or None
        Sky coordinate system of the map (e.g. Galactic or Ecliptic).
        If ``None``, coordinates are unspecified.
    nest : bool, optional
        If True, the map is in NESTED ordering; if False, in RING (default).
        This is just metadata here, no indexing operations are performed.
    nstokes : int
        Number of Stokes parameters (1 for intensity-only, 3 for IQU).
    """

    values: np.ndarray
    nside: int | None = None
    units: Units | None = None
    coordinates: CoordinateSystem | None = None
    nest: bool = False
    nstokes: int = field(init=False)

    def __post_init__(self):
        """
        Initialize the `HealpixMap` instance by validating and reshaping input.

        - Converts `values` tuple/list to ndarray and shapes 1D input to `(1, npix)`.
        - Infers `nside` from `npix` if `nside` is None.
        - Validates `nside` against `npix` if `nside` is provided.
        - Normalizes `units` and `coordinates`.
        - Sets `nstokes`.

        Raises
        ------
        AssertionError
            If `nside` (provided or inferred) is not a valid HEALPix NSIDE value.
        ValueError
            If `nstokes` is not 1 or 3, or if the number of pixels does
            not match the provided `nside`.
        """

        # 1. Normalize values shape first (we need npix to check/infer nside)
        # Convert tuple of arrays (e.g. (I, Q, U)) into a stacked ndarray
        if isinstance(self.values, tuple):
            self.values = np.array([self.values[i] for i in range(len(self.values))])

        # Ensure values have shape (nstokes, npix)
        if self.values.ndim == 1:
            self.values = self.values[np.newaxis, :]

        # Get the actual number of pixels from the data
        npix = self.values.shape[1]

        # 2. Handle NSIDE inference or validation
        if self.nside is None:
            # Infer NSIDE from npix
            # npix_to_nside raises AssertionError if npix is not valid (12 * nside^2)
            try:
                self.nside = HealpixMap.npix_to_nside(npix)
            except AssertionError as e:
                raise ValueError(
                    f"Input values have {npix} pixels, which is not a valid HEALPix map size."
                ) from e
        else:
            # Check provided NSIDE validity
            _ = HealpixMap.nside_to_npix(
                self.nside
            )  # raises AssertionError if invalid power of 2

            # Check consistency between provided NSIDE and data
            expected_npix = HealpixMap.nside_to_npix(self.nside)
            if npix != expected_npix:
                raise ValueError(
                    f"Wrong number of pixels for HealpixMap: data has "
                    f"{npix}, but expected {expected_npix} for nside={self.nside}."
                )

        # 3. Normalize metadata (units, coordinates, nstokes)
        if self.units is not None and not isinstance(self.units, Units):
            raise ValueError(
                f"units must be an instance of Units or None, got {type(self.units)!r}"
            )

        if self.coordinates is not None and not isinstance(
            self.coordinates, CoordinateSystem
        ):
            raise ValueError(
                "coordinates must be an instance of CoordinateSystem or None, "
                f"got {type(self.coordinates)!r}"
            )

        self.nstokes = self.values.shape[0]
        if self.nstokes not in (1, 3):
            raise ValueError(
                "The number of Stokes parameters in HealpixMap should be 1 or 3 "
                f"instead of {self.nstokes}."
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
    def _is_power_of_two(n: int) -> bool:
        """Helper to check if an integer is a valid power of 2."""
        return (n > 0) and ((n & (n - 1)) == 0)

    @staticmethod
    def nside_to_npix(nside: int) -> int:
        """Return the number of pixels in a Healpix map with the specified NSIDE.

        If the value of `nside` is not valid (power of two), an
        `AssertionError` exception is raised.

        .. doctest::

            >>> HealpixMap.nside_to_npix(1)
            12
        """
        assert HealpixMap._is_power_of_two(nside), (
            f"Invalid value for NSIDE: {nside} (must be power of 2)"
        )
        return 12 * nside * nside

    @staticmethod
    def is_npix_ok(num_of_pixels) -> bool:
        """Return True or False whenever num_of_pixels is a valid number.

        The function checks if the number of pixels provided as an
        argument conforms to the Healpix standard, which means that the
        number must be in the form 12 * NSIDE^2, where NSIDE is a power of 2.

        .. doctest::

            >>> HealpixMap.is_npix_ok(48)
            True
            >>> HealpixMap.is_npix_ok(108) # 12 * 3^2, but 3 not power of 2
            False
        """
        # Calculate potential nside
        nside_float = np.sqrt(np.asarray(num_of_pixels) / 12.0)
        nside = int(nside_float)

        # Check if nside is integer AND is a power of 2
        return (nside_float == nside) and HealpixMap._is_power_of_two(nside)

    @staticmethod
    def npix_to_nside(num_of_pixels: int) -> int:
        """Return NSIDE for a Healpix map containing `num_of_pixels` pixels.

        If the number of pixels does not conform to the Healpix standard,
        an `AssertionError` exception is raised.

        .. doctest::

            >>> HealpixMap.npix_to_nside(48)
            2
        """
        assert HealpixMap.is_npix_ok(num_of_pixels), (
            f"Invalid number of pixels: {num_of_pixels}"
        )
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

    def _units_compatible(self, other: "HealpixMap") -> bool:
        """
        Return True if units are compatible for algebraic operations.

        Rules
        -----
        - If either units is None → always compatible.
        - Otherwise, units must be exactly equal.
        """
        u1 = self.units
        u2 = other.units

        # Treat Units.None and Python None as "no units / don't care"
        is_none_1 = u1 is None
        is_none_2 = u2 is None

        if is_none_1 or is_none_2:
            return True

        return u1 == u2

    def _coordinates_compatible(self, other: "HealpixMap") -> bool:
        """
        Return True if coordinates are compatible for algebraic operations.

        Rules
        -----
        - If either coordinates is None → always compatible.
        - Otherwise, coordinates must be exactly equal.
        """
        c1 = self.coordinates
        c2 = other.coordinates

        if c1 is None or c2 is None:
            return True

        return c1 == c2

    def __add__(self, other: "HealpixMap") -> "HealpixMap":
        """Add two maps with the same geometry and compatible units / coordinates."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Can only add another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for addition: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for addition: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        # Resulting units:
        # - if self has "real" units, keep it
        # - otherwise, inherit other's units (could be None)
        if self.units is not None:
            result_units = self.units
        else:
            result_units = other.units

        # Resulting coordinates: keep self if defined, otherwise inherit other
        result_coordinates = (
            self.coordinates if self.coordinates is not None else other.coordinates
        )

        return HealpixMap(
            values=self.values + other.values,
            nside=self.nside,
            units=result_units,
            coordinates=result_coordinates,
            nest=self.nest,
        )

    def __iadd__(self, other: "HealpixMap") -> "HealpixMap":
        """In-place addition of two maps with the same geometry and units."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Can only add another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for addition: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for addition: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        self.values += other.values

        # Optionally update units if self was unitless and other had a units
        if self.units is None and other.units is not None:
            self.units = other.units

        # Optionally update coordinates if self had none and other has them
        if self.coordinates is None and other.coordinates is not None:
            self.coordinates = other.coordinates

        return self

    def __sub__(self, other: "HealpixMap") -> "HealpixMap":
        """Subtract two maps with the same geometry and compatible units / coordinates."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Subtraction requires another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for subtraction: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for subtraction: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        # For subtraction, use the same units resolution logic as for addition
        if self.units is not None:
            result_units = self.units
        else:
            result_units = other.units

        result_coordinates = (
            self.coordinates if self.coordinates is not None else other.coordinates
        )

        return HealpixMap(
            values=self.values - other.values,
            nside=self.nside,
            units=result_units,
            coordinates=result_coordinates,
            nest=self.nest,
        )

    def __isub__(self, other: "HealpixMap") -> "HealpixMap":
        """In-place subtraction of two maps with the same geometry and units."""
        if not isinstance(other, HealpixMap):
            raise TypeError("Subtraction requires another HealpixMap object")

        if not self.is_consistent(other):
            raise ValueError(
                "HealpixMap objects must have matching nside, nest, and nstokes"
            )

        if not self._units_compatible(other):
            raise ValueError(
                f"Incompatible units for subtraction: {self.units} vs {other.units}"
            )

        if not self._coordinates_compatible(other):
            raise ValueError(
                f"Incompatible coordinates for subtraction: "
                f"{self.coordinates} vs {other.coordinates}"
            )

        self.values -= other.values
        # Same optional units update as in __iadd__
        if self.units is None and other.units is not None:
            self.units = other.units

        if self.coordinates is None and other.coordinates is not None:
            self.coordinates = other.coordinates

        return self

    def __mul__(
        self,
        other: float | np.ndarray,
        *,
        units: Units | None = None,
    ) -> "HealpixMap":
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
        units : Units or None, keyword-only
            Units to assign to the resulting map. If ``None`` (default),
            the units of ``self`` is preserved.

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

        # If units is not provided, keep the original units
        new_units = self.units if units is None else units

        return HealpixMap(
            values=new_values,
            nside=self.nside,
            units=new_units,
            coordinates=self.coordinates,
            nest=self.nest,
        )

    def __rmul__(
        self,
        other: float | np.ndarray,
        *,
        units: Units | None = None,
    ) -> "HealpixMap":
        """
        Right-multiplication: allow units override just like in __mul__.

        Parameters
        ----------
        other : float, int, complex or np.ndarray
            Multiplier.
        units : Units or None, keyword-only
            Units to assign to the resulting map. If None, keeps self.units.

        Notes
        -----
        Normal uses like ``2 * map`` cannot pass ``units=...`` because the
        Python operator syntax does not allow it. But explicit calls to
        ``__rmul__`` can override units:

            newmap = map.__rmul__(2.0, units=Units.K_CMB)
        """
        return self.__mul__(other, units=units)

    # ------------------------------------------------------------------
    # Comparison and utilities
    # ------------------------------------------------------------------

    def copy(self) -> "HealpixMap":
        """Return a deep copy of this HealpixMap object."""
        return HealpixMap(
            values=self.values.copy(),
            nside=self.nside,
            units=self.units,
            coordinates=self.coordinates,
            nest=self.nest,
        )

    def __eq__(self, other: object) -> bool:
        """
        Exact equality: same geometry, same units, same coordinates
        and identical pixel values.
        """
        if not isinstance(other, HealpixMap):
            raise ValueError("Can only compare with another HealpixMap object.")
        return (
            self.is_consistent(other)
            and self.units == other.units
            and self.coordinates == other.coordinates
            and np.array_equal(self.values, other.values)
        )

    def allclose(self, other: "HealpixMap", rtol=1e-5, atol=1e-8) -> bool:
        """
        Compare map values with tolerance.

        Units and coordinates must be compatible. Units are compatible if
        both match or one/both are None; coordinates are
        compatible if both match or one/both are None.
        """
        if not isinstance(other, HealpixMap):
            raise ValueError("Can only compare with another HealpixMap object.")

        if not self.is_consistent(other):
            return False

        if not self._units_compatible(other):
            return False

        if not self._coordinates_compatible(other):
            return False

        return np.allclose(self.values, other.values, rtol=rtol, atol=atol)


# ======================================================================
# ducc0-based helpers
# ======================================================================


def interpolate_alm(
    alms: "SphericalHarmonics",
    locations: np.ndarray,
    *,
    epsilon: float | None = None,
    nthreads: int = 0,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Interpolate spherical-harmonic coefficients at arbitrary positions
    using :func:`ducc0.sht.synthesis_general`.

    This wrapper supports both scalar and polarized fields:

    - If ``alms.nstokes == 1``: a single spin-0 field is synthesized and
      the function returns a 1D array ``T`` of length ``N``.
    - If ``alms.nstokes == 3``: the first component is interpreted as
      a spin-0 field (T), and the remaining two as a spin-2 field,
      returning ``T, Q, U``.

    Parameters
    ----------
    alms : SphericalHarmonics
        Object exposing at least the attributes:

        * ``values``: ndarray, shape ``(nstokes, nalm)``, complex64/complex128
        * ``lmax``: int
        * ``mmax``: int
        * ``nstokes``: int (1 or 3)

        The last dimension of ``values`` must follow the standard triangular
        (healpy/ducc) ordering.

    locations : ndarray, shape (N, 2)
        Target positions on the sphere, in radians.
        ``locations[:, 0]`` = colatitude ``theta`` (0 .. π),
        ``locations[:, 1]`` = longitude ``phi`` (0 .. 2π).

    epsilon : float, optional
        Desired accuracy passed to :func:`synthesis_general`.
        If ``None``, a safe default is chosen depending on the dtype:

        * complex64  → ``1e-6``
        * complex128 → ``1e-13``

    nthreads : int, optional
        Number of threads for ducc. If 0 (default), ducc uses the
        number of hardware threads.

    Returns
    -------
    T : ndarray
        If ``alms.nstokes == 1``, a 1D array of length ``N`` with the
        interpolated scalar field.
    T, Q, U : ndarray
        If ``alms.nstokes == 3``, three 1D arrays of length ``N`` with the
        interpolated Stokes parameters.

    Raises
    ------
    ValueError
        If the input shapes are inconsistent or ``nstokes`` is not 1 or 3.
    """
    alm = np.asarray(alms.values)

    if alm.ndim != 2:
        raise ValueError(
            f"`alms.values` must be 2D (nstokes, nalm); got shape {alm.shape!r}"
        )

    nstokes = alm.shape[0]
    if nstokes not in (1, 3):
        raise ValueError(f"`alms.nstokes` must be 1 (scalar) or 3 (IQU); got {nstokes}")

    loc = np.asarray(locations, dtype=np.float64)
    if loc.ndim != 2 or loc.shape[1] != 2:
        raise ValueError(
            f"`locations` must have shape (N, 2) [theta, phi]; got {loc.shape!r}"
        )

    # Choose epsilon if not given, respecting ducc constraints
    if epsilon is None:
        if alm.dtype == np.complex64:
            epsilon = 1e-6
        else:  # assume complex128 or higher precision
            epsilon = 1e-13

    # --- Temperature (spin-0) ---
    T_map = sht.synthesis_general(
        alm=alm[0:1],
        spin=0,
        lmax=alms.lmax,
        loc=loc,
        epsilon=epsilon,
        mmax=alms.mmax,
        nthreads=nthreads,
    )[0]  # (1, N) -> (N,)

    # If scalar-only, we're done
    if nstokes == 1:
        return T_map

    # --- Polarization (spin-2) ---
    # ducc returns nmaps = 2 for spin>0, so this gives (2, N).
    QU_maps = sht.synthesis_general(
        alm=alm[1:3],
        spin=2,
        lmax=alms.lmax,
        loc=loc,
        epsilon=epsilon,
        mmax=alms.mmax,
        nthreads=nthreads,
    )
    Q_map, U_map = QU_maps

    return T_map, Q_map, U_map


def pixelize_alm(
    alms: "SphericalHarmonics",
    nside: int,
    *,
    nest: bool = False,
    lmax: Optional[int] = None,
    mmax: Optional[int] = None,
    nthreads: int = 0,
) -> "HealpixMap":
    r"""
    Convert spherical harmonics coefficients to a HEALPix map using
    :func:`ducc0.sht.synthesis` on the HEALPix geometry.

    This is essentially an ``pixelize_alm`` implemented on top of ducc0, with
    support for both scalar and polarized fields:

    - ``alms.nstokes == 1`` → spin-0 synthesis, returns a 1-component map
    - ``alms.nstokes == 3`` → spin-0 for T and spin-2 for (Q, U),
      returns a 3-component map

    Geometry is obtained from :class:`ducc0.healpix.Healpix_Base` via
    its :meth:`sht_info` method, which provides the arrays
    ``theta``, ``nphi``, ``phi0`` and (optionally) ``ringstart``
    needed by ducc's SHT routines.

    The units and coordinates metadata are propagated from ``alms`` to
    the returned :class:`HealpixMap`.

    Parameters
    ----------
    alms : SphericalHarmonics
        Harmonic coefficients. Must satisfy:

        * ``alms.values.shape == (nstokes, nalm)``
        * ``alms.nstokes in (1, 3)``
        * triangular Healpy/ducc layout in the last dimension
          (i.e. increasing m, l ≥ m).

    nside : int
        Target HEALPix resolution (power of two). The output map will have
        ``npix = 12 * nside**2`` pixels.

    nest : bool, optional
        If ``False`` (default), the map is in RING ordering.
        If ``True``, the map is in NESTED ordering. The underlying SHT is
        always performed in RING geometry; for NESTED, the geometry is built
        from a NESTED :class:`Healpix_Base` but pixel ordering in the
        returned :class:`HealpixMap` is marked accordingly.

    lmax, mmax : int, optional
        Maximum multipoles to use in the synthesis. If omitted, the values
        from ``alms`` are used:

        * ``lmax_eff = alms.lmax`` if ``lmax`` is ``None``
        * ``mmax_eff = alms.mmax`` if ``mmax`` is ``None``

        Both must be ≤ the corresponding values in ``alms``. If a smaller
        ``lmax``/``mmax`` is requested than what is stored in ``alms``,
        the coefficient array is truncated accordingly.

    nthreads : int, optional
        Number of threads passed to ducc. If zero (default), ducc chooses
        the number of threads.

    Returns
    -------
    HealpixMap
        A HEALPix map with shape ``(nstokes, npix)`` and metadata

        * ``nside`` as specified
        * ``nest`` as specified
        * ``units = alms.units``
        * ``coordinates = alms.coordinates``

    Raises
    ------
    ValueError
        If the SphericalHarmonics object has unsupported ``nstokes``,
        or if requested ``lmax``/``mmax`` are inconsistent with the
        available coefficients.
    """
    # --- basic checks / effective lmax, mmax -------------------------------
    if alms.nstokes not in (1, 3):
        raise ValueError(
            f"SphericalHarmonics.nstokes must be 1 (T) or 3 (T,Q,U), got {alms.nstokes}"
        )

    # Check NSIDE via HealpixMap helper (will raise on invalid NSIDE)
    npix = HealpixMap.nside_to_npix(nside)

    lmax_eff = alms.lmax if lmax is None else lmax
    mmax_eff = alms.mmax if mmax is None else mmax

    if lmax_eff > alms.lmax:
        raise ValueError(
            f"Requested lmax={lmax_eff} exceeds available alms.lmax={alms.lmax}"
        )
    if mmax_eff > alms.mmax:
        raise ValueError(
            f"Requested mmax={mmax_eff} exceeds available alms.mmax={alms.mmax}"
        )

    # Expected number of alm per Stokes for the requested (lmax,mmax)
    expected_nalm = SphericalHarmonics.num_of_alm_from_lmax(lmax_eff, mmax_eff)
    if expected_nalm > alms.num_of_alm_per_stokes:
        raise ValueError(
            "Not enough coefficients in SphericalHarmonics.values for "
            f"lmax={lmax_eff}, mmax={mmax_eff}: need {expected_nalm}, "
            f"have {alms.num_of_alm_per_stokes}"
        )

    alm_all = np.asarray(alms.values)
    if alm_all.shape[1] > expected_nalm:
        alm_all = alm_all[:, :expected_nalm]

    # Choose map dtype from alm dtype: complex64 -> float32, else float64
    if alm_all.dtype == np.complex64:
        map_dtype = np.float32
    else:
        map_dtype = np.float64

    # --- build Healpix geometry from ducc0.healpix -------------------------
    # RING or NESTED scheme
    scheme = "NESTED" if nest else "RING"
    base = dh.Healpix_Base(nside, scheme)
    geom = base.sht_info()

    # --- allocate output map -----------------------------------------------
    values = np.empty((alms.nstokes, npix), dtype=map_dtype)

    # --- spin-0 (temperature / scalar) -------------------------------------
    alm_T = alm_all[0:1]  # (1, nalm)
    sht.synthesis(
        alm=alm_T,
        map=values[0:1],
        **geom,
        spin=0,
        lmax=lmax_eff,
        mmax=mmax_eff,
        nthreads=nthreads,
    )

    # --- spin-2 (polarization Q,U) if present ------------------------------
    if alms.nstokes == 3:
        alm_pol = alm_all[1:3]  # (2, nalm)
        sht.synthesis(
            alm=alm_pol,
            map=values[1:3],
            **geom,
            spin=2,
            lmax=lmax_eff,
            mmax=mmax_eff,
            nthreads=nthreads,
        )

    # --- wrap into HealpixMap, propagating units and coordinates -----------
    return HealpixMap(
        values=values,
        nside=nside,
        nest=nest,
        units=alms.units,
        coordinates=alms.coordinates,
    )


def estimate_alm(
    map: "HealpixMap",
    *,
    lmax: Optional[int] = None,
    mmax: Optional[int] = None,
    nthreads: int = 0,
) -> "SphericalHarmonics":
    r"""
    Convert a HEALPix map to spherical harmonics coefficients using
    :func:`ducc0.sht.adjoint_synthesis` on the HEALPix geometry.

    This is essentially a ``estimate_alm`` implemented on top of ducc0, with
    support for both scalar and polarized fields:

    - ``map.nstokes == 1`` → spin-0 transform, returns a 1-component alm
    - ``map.nstokes == 3`` → spin-0 for T and spin-2 for (Q, U),
      returns a 3-component alm (T, Q, U interpreted as spin-2)

    Geometry is obtained from :class:`ducc0.healpix.Healpix_Base` via
    its :meth:`sht_info` method, which provides the arrays
    ``theta``, ``nphi``, ``phi0`` and (optionally) ``ringstart``
    needed by ducc's SHT routines.

    The units and coordinates metadata are propagated from ``map`` to
    the returned :class:`SphericalHarmonics`.

    Parameters
    ----------
    map : HealpixMap
        Input HEALPix map. Must satisfy:

        * ``map.values.shape == (nstokes, npix)``
        * ``map.nstokes in (1, 3)``
        * ``map.nside`` is a power of two.

    lmax, mmax : int, optional
        Maximum multipoles to compute. Defaults are chosen in a
        Healpy-like way:

        * if ``lmax`` is None → ``lmax_eff = 3 * nside - 1``
        * if ``mmax`` is None → ``mmax_eff = lmax_eff``

        ``mmax_eff`` must be ≤ ``lmax_eff``.

    nthreads : int, optional
        Number of threads passed to ducc. If zero (default), ducc chooses
        the number of threads.

    Returns
    -------
    SphericalHarmonics
        A spherical harmonics object with

        * ``values`` of shape ``(nstokes, nalm)``
        * ``lmax = lmax_eff``
        * ``mmax = mmax_eff``
        * ``units = map.units``
        * ``coordinates = map.coordinates``

    Raises
    ------
    ValueError
        If the HealpixMap object has unsupported ``nstokes``,
        or if requested ``lmax``/``mmax`` are inconsistent.
    """
    # --- checks and effective lmax, mmax -----------------------------------
    if map.nstokes not in (1, 3):
        raise ValueError(
            f"HealpixMap.nstokes must be 1 (T) or 3 (T,Q,U), got {map.nstokes}"
        )

    # Validate NSIDE
    npix = HealpixMap.nside_to_npix(map.nside)
    if map.npix != npix:
        raise ValueError(
            f"Inconsistent number of pixels: map has {map.npix}, "
            f"but nside={map.nside} implies {npix}"
        )

    nside = map.nside

    # Healpy-like defaults
    lmax_eff = 3 * nside - 1 if lmax is None else lmax
    mmax_eff = lmax_eff if mmax is None else mmax

    if mmax_eff < 0 or mmax_eff > lmax_eff:
        raise ValueError(
            f"Invalid mmax={mmax_eff} for lmax={lmax_eff}: must satisfy 0 <= mmax <= lmax"
        )

    # Expected number of alm per Stokes
    nalm = SphericalHarmonics.num_of_alm_from_lmax(lmax_eff, mmax_eff)

    # --- choose alm dtype from map dtype -----------------------------------
    map_arr = np.asarray(map.values)
    if map_arr.dtype == np.float32:
        alm_dtype = np.complex64
    else:
        # promote everything else to float64 -> complex128
        if map_arr.dtype != np.float64:
            map_arr = map_arr.astype(np.float64)
        alm_dtype = np.complex128

    alm = np.zeros((map.nstokes, nalm), dtype=alm_dtype)

    # --- build Healpix geometry from ducc0.healpix -------------------------
    scheme = "NESTED" if map.nest else "RING"
    base = dh.Healpix_Base(nside, scheme)
    geom = base.sht_info()  # provides theta, nphi, phi0, ringstart, etc.

    # --- spin-0 (T or scalar) ----------------------------------------------
    sht.adjoint_synthesis(
        alm=alm[0:1],
        map=map_arr[0:1],
        **geom,
        spin=0,
        lmax=lmax_eff,
        mmax=mmax_eff,
        nthreads=nthreads,
    )

    # --- spin-2 (Q,U) if present -------------------------------------------
    if map.nstokes == 3:
        sht.adjoint_synthesis(
            alm=alm[1:3],
            map=map_arr[1:3],
            **geom,
            spin=2,
            lmax=lmax_eff,
            mmax=mmax_eff,
            nthreads=nthreads,
        )

    # --- wrap into SphericalHarmonics, propagating units and coordinates ---
    return SphericalHarmonics(
        values=alm,
        lmax=lmax_eff,
        mmax=mmax_eff,
        units=map.units,
        coordinates=map.coordinates,
    )
