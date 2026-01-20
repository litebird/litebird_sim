import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging as log

import ducc0.healpix as dh
import ducc0.sht as sht
import numpy as np
from astropy.io import fits

from .coordinates import ECL_TO_GAL_EULER, GAL_TO_ECL_EULER, CoordinateSystem
from .units import Units

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

    **Ordering Convention (Healpix Scheme):**
    The coefficients are stored in **m-major** order.
    1. The outer loop iterates over `m` from 0 to `mmax`.
    2. The inner loop iterates over `l` from `m` to `lmax`.

    Sequence of coefficients in the array:
    Index 0: (l=0, m=0)
    Index 1: (l=1, m=0)
    ...
    Index lmax: (l=lmax, m=0)
    Index lmax+1: (l=1, m=1)   <-- Note: l starts at m
    Index lmax+2: (l=2, m=1)
    ...

    This specific ordering is required for operations involving `healpy.alm2map`
    or `ducc0.sht`.

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
    # Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def zeros(
        cls,
        lmax: int,
        mmax: int | None = None,
        nstokes: int = 3,
        dtype: type = np.complex128,
        units: Units | None = None,
        coordinates: CoordinateSystem | None = None,
    ) -> "SphericalHarmonics":
        """
        Create a SphericalHarmonics object filled with zeros.

        Parameters
        ----------
        lmax : int
            Maximum degree.
        mmax : int, optional
            Maximum order. Defaults to lmax.
        nstokes : int, default=3
            Number of Stokes parameters (1 or 3).
        dtype : type, default=np.complex128
            Data type for the array. Must be np.complex64 or np.complex128.
        units : Units, optional
            Physical units.
        coordinates : CoordinateSystem, optional
            Sky coordinate system.

        Returns
        -------
        SphericalHarmonics
            Instance initialized with zeros.

        Raises
        ------
        ValueError
            If dtype is not np.complex64 or np.complex128.
        """

        # Validazione del dtype
        if dtype not in (np.complex64, np.complex128):
            raise ValueError(
                f"dtype must be either np.complex64 or np.complex128, got {dtype!r}"
            )

        shape = cls.alm_array_size(lmax, mmax, nstokes)
        return cls(
            values=np.zeros(shape, dtype=dtype),
            lmax=lmax,
            mmax=mmax if mmax is not None else lmax,
            units=units,
            coordinates=coordinates,
        )

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

    @staticmethod
    def get_index(lmax: int, l: Any, m: Any) -> Any:
        """
        Calculate the 1D linear index in the standard Healpix (m-major) layout
        for the given (l, m) pairs.

        Formula: index = m * (2 * lmax + 1 - m) / 2 + l

        Parameters
        ----------
        lmax : int
            Maximum degree of the expansion.
        l : int or np.ndarray
            Degree l.
        m : int or np.ndarray
            Order m.

        Returns
        -------
        int or np.ndarray
            The 1D index. Returns an integer if inputs are scalars,
            otherwise a numpy array.
        """
        # Calculate the index using integer division
        idx = m * (2 * lmax + 1 - m) // 2 + l

        # Robustly handle both scalar and array results
        if np.isscalar(idx):
            return int(idx)
        return np.asarray(idx, dtype=int)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def get_lm_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate arrays of l and m for every coefficient stored in this object.

        This reconstructs the explicit (l, m) coordinates for the flattened
        data array, following the m-major ordering used internally.

        Returns
        -------
        l_arr : np.ndarray
            Array containing the l (degree) for each element.
        m_arr : np.ndarray
            Array containing the m (order) for each element.
        """
        ls = []
        ms = []

        # In Healpix layout (m-major), m ranges from 0 to mmax.
        # For each m, l ranges from m to lmax.
        for m_val in range(self.mmax + 1):
            l_vals = np.arange(m_val, self.lmax + 1)
            ls.append(l_vals)
            ms.append(np.full_like(l_vals, m_val))

        return np.concatenate(ls), np.concatenate(ms)

    def print_ordering_example(self):
        """Prints the first few (l, m) pairs to demonstrate ordering."""
        idx = 0
        print(f"Ordering for lmax={self.lmax}, mmax={self.mmax}:")
        for m in range(self.mmax + 1):
            for l in range(m, self.lmax + 1):
                print(f"Index {idx}: (l={l}, m={m})")
                idx += 1
                if idx > 10:  # Just show the start
                    print("...")
                    return

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
            return self
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
        if self.units:
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

    def convolve(
        self, f_ell: np.ndarray | list[np.ndarray], inplace: bool = True
    ) -> "SphericalHarmonics":
        """
        Apply a beam or filter to the SH coefficients.

        Parameters
        ----------
        f_ell : np.ndarray or list[np.ndarray]
            The ℓ-dependent filter(s).
        inplace : bool, optional
            If True, modifies the coefficients of the current object.
            If False, returns a new SphericalHarmonics object.
            Default is True.

        Returns
        -------
        SphericalHarmonics
            The object itself (if inplace=True) or a new object (if inplace=False).
        """
        # The filter must be defined at least up to self.lmax
        required_size = self.lmax + 1

        # Get the l-index for every coefficient in the alm array
        l_arr = self.alm_l_array(self.lmax, self.mmax)

        if isinstance(f_ell, np.ndarray):
            if f_ell.ndim == 1:
                # Case: Single filter applied to all maps (T, E, B...)
                if f_ell.shape[0] < required_size:
                    raise ValueError(
                        f"Filter size ({f_ell.shape[0]}) is smaller than required lmax+1 ({required_size})"
                    )
                # Broadcast 1D filter to (nstokes, ncoeff)
                kernel = f_ell[l_arr]

            elif f_ell.ndim == 2:
                # Case: Specific filter for each component
                if f_ell.shape[0] != self.nstokes:
                    raise ValueError(
                        f"Filter nstokes ({f_ell.shape[0]}) does not match object nstokes ({self.nstokes})"
                    )
                if f_ell.shape[1] < required_size:
                    raise ValueError(
                        f"Filter length ({f_ell.shape[1]}) is smaller than required lmax+1 ({required_size})"
                    )
                # Extract correct l values for each component
                kernel = np.stack([f_row[l_arr] for f_row in f_ell])

            else:
                raise ValueError(
                    f"Invalid shape for f_ell: {f_ell.shape}. Expected 1D or 2D array."
                )

        elif isinstance(f_ell, list):
            # Case: List of filters
            if len(f_ell) != self.nstokes:
                raise ValueError(
                    f"Expected {self.nstokes} filters in list, got {len(f_ell)}"
                )

            kernel_list = []
            for i, f in enumerate(f_ell):
                f_arr = np.asarray(f)
                if f_arr.ndim != 1:
                    raise ValueError(f"Filter at index {i} must be 1D.")
                if f_arr.shape[0] < required_size:
                    raise ValueError(
                        f"Filter {i} size ({f_arr.shape[0]}) is smaller than required lmax+1 ({required_size})"
                    )
                kernel_list.append(f_arr[l_arr])

            kernel = np.stack(kernel_list)
        else:
            raise TypeError("f_ell must be a numpy array or a list of numpy arrays")

        # Apply the kernel
        if inplace:
            self.values *= kernel
            return self
        else:
            return SphericalHarmonics(
                values=self.values * kernel,
                lmax=self.lmax,
                mmax=self.mmax,
                units=self.units,
                coordinates=self.coordinates,
            )

    def apply_gaussian_smoothing(
        self, fwhm_rad: float, inplace: bool = True
    ) -> "SphericalHarmonics":
        """
        Apply Gaussian smoothing to the spherical harmonics coefficients.

        Parameters
        ----------
        fwhm_rad : float
            Full Width at Half Maximum (FWHM) of the Gaussian beam in radians.
        inplace : bool, optional
            If True, modifies the object in place. Default is True.

        Returns
        -------
        SphericalHarmonics
            The smoothed object.
        """

        # LOCAL IMPORT to prevent circular dependency
        from .beam_synthesis import gauss_bl

        nstokes = self.values.shape[0]
        use_pol = nstokes > 1

        bl = gauss_bl(lmax=self.lmax, fwhm_rad=fwhm_rad, pol=use_pol)

        return self.convolve(bl, inplace=inplace)

    def apply_pixel_window(
        self, nside: int, inplace: bool = True
    ) -> "SphericalHarmonics":
        """
        Apply the HEALPix pixel window function.

        Parameters
        ----------
        nside : int
            The HEALPix Nside resolution parameter.
        inplace : bool, optional
            If True, modifies the object in place. Default is True.

        Returns
        -------
        SphericalHarmonics
            The object with pixel window applied.
        """
        nstokes = self.values.shape[0]
        use_pol = nstokes > 1

        pw_ell = pixel_window(nside, lmax=self.lmax, pol=use_pol)

        return self.convolve(pw_ell, inplace=inplace)

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
            return NotImplemented

        return (
            self.is_consistent(other)
            and self.units == other.units
            and self.coordinates == other.coordinates
            and np.array_equal(self.values, other.values)
        )

    def allclose(
        self, other: "SphericalHarmonics", rtol: float = 1.0e-5, atol: float = 1.0e-8
    ) -> bool:
        """
        Check if two SphericalHarmonics objects are equal within a tolerance.

        Units must be compatible (same, or one/both None/Units.None).
        Coordinates must also be compatible (same, or one/both None).

        Returns
        -------
        bool
            True if the objects are structurally consistent and their values are
            close within the specified tolerance.
            If the objects are inconsistent (e.g., different lmax, units, etc.),
            a warning is logged with the specific reason, and False is returned.
        """
        if not isinstance(other, SphericalHarmonics):
            raise TypeError("Can only compare with another SphericalHarmonics object.")

        # 1. Check structural consistency (lmax, mmax, nstokes)
        # We check attributes individually to log specific mismatch reasons
        if self.lmax != other.lmax:
            log.warning(
                f"allclose mismatch: lmax differs ({self.lmax} vs {other.lmax})"
            )
            return False

        if self.mmax != other.mmax:
            log.warning(
                f"allclose mismatch: mmax differs ({self.mmax} vs {other.mmax})"
            )
            return False

        if self.nstokes != other.nstokes:
            log.warning(
                f"allclose mismatch: nstokes differs ({self.nstokes} vs {other.nstokes})"
            )
            return False

        # 2. Check units compatibility
        if not self._units_compatible(other):
            log.warning(
                f"allclose mismatch: Incompatible units ({self.units} vs {other.units})"
            )
            return False

        # 3. Check coordinates compatibility
        if not self._coordinates_compatible(other):
            log.warning(
                f"allclose mismatch: Incompatible coordinates ({self.coordinates} vs {other.coordinates})"
            )
            return False

        # 4. Check numerical equality
        return np.allclose(self.values, other.values, rtol=rtol, atol=atol)

    # -------------
    # I/O
    # -------------

    def write_fits(self, filename: str, overwrite: bool = True):
        """
        Save the SphericalHarmonics object to a FITS file using the Explicit Index scheme.

        Format:
        - Separate HDUs for TEMPERATURE, E_MODE, B_MODE.
        - Each HDU has 3 columns: INDEX, REAL, IMAG.
        - INDEX = l^2 + l + m + 1 (FITS standard for sparse alm).
        """
        # 1. Retrieve l, m for each coefficient currently in memory
        l, m = self.get_lm_arrays()

        # 2. Compute the internal index (location of data in our numpy array)
        idx_internal = self.get_index(self.lmax, l, m)

        # 3. Compute the FITS Explicit Index (l^2 + l + m + 1)
        idx_fits = l**2 + l + m + 1

        # Handle components (1 for T, 3 for T,E,B)
        if self.nstokes == 3:
            components = self.values  # shape (3, Nalms)
            ext_names = ["TEMPERATURE", "E_MODE", "B_MODE"]
        else:
            # Handle scalar case or 1D array
            if self.values.ndim == 2:
                components = [self.values[0]]
            else:
                components = [self.values]
            ext_names = ["TEMPERATURE"]

        hdus = [fits.PrimaryHDU()]

        for comp_data, ext_name in zip(components, ext_names):
            # Extract data from internal array using the internal index
            real_part = comp_data.real[idx_internal]
            imag_part = comp_data.imag[idx_internal]

            # Create FITS columns ('J'=int32, 'D'=float64)
            col_idx = fits.Column(name="INDEX", format="J", array=idx_fits)
            col_real = fits.Column(name="REAL", format="D", array=real_part)
            col_imag = fits.Column(name="IMAG", format="D", array=imag_part)

            hdu = fits.BinTableHDU.from_columns([col_idx, col_real, col_imag])
            hdu.name = ext_name

            # Standard Healpix Header keywords
            hdu.header["PIXTYPE"] = ("HEALPIX", "HEALPIX pixelisation")
            hdu.header["MAX-LEN"] = (self.lmax, "Maximum l index")
            hdu.header["MAX-M"] = (self.mmax, "Maximum m index")

            hdus.append(hdu)

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(filename, overwrite=overwrite)

    @staticmethod
    def read_fits(filename: str) -> "SphericalHarmonics":
        """
        Load a SphericalHarmonics object from a FITS file.
        Compatible with both Explicit Index format and Standard Healpix format.
        """
        values_list = []
        lmax_file = 0
        mmax_file = 0
        is_explicit = False

        with fits.open(filename) as hdul:
            # Quick format check
            try:
                first_cols = [c.name.lower() for c in hdul[1].data.columns]
                is_explicit = "index" in first_cols and "real" in first_cols
            except IndexError:
                pass  # Fallback or error if file is empty

            if is_explicit:
                # --- EXPLICIT FORMAT (INDEX, REAL, IMAG) ---
                target_hdus = []
                # Attempt to read up to 3 data HDUs (T, E, B)
                for i in range(1, min(len(hdul), 4)):
                    target_hdus.append(hdul[i])

                for hdu in target_hdus:
                    data = hdu.data
                    idx = data["INDEX"]
                    re = data["REAL"]
                    im = data["IMAG"]

                    # Inverse conversion: from FITS INDEX to (l, m)
                    # formula: l = floor(sqrt(idx - 1))
                    l = np.floor(np.sqrt(idx - 1)).astype(int)
                    m = idx - l**2 - l - 1

                    # Update lmax/mmax
                    lmax_file = max(lmax_file, l.max())
                    mmax_file = max(mmax_file, m.max())

                    values_list.append({"l": l, "m": m, "re": re, "im": im})

            else:
                # --- STANDARD HEALPIX FORMAT (Complex columns) ---
                header = hdul[1].header
                data = hdul[1].data

                lmax_file = header.get("MAX-LEN", header.get("LMAX", -1))
                mmax_file = header.get("MAX-M", header.get("MMAX", -1))

                # Read complex columns (T, and optionally E, B)
                for col in data.columns:
                    # Filter for numeric/complex columns if necessary
                    c_data = data[col.name].astype(np.complex128)
                    values_list.append(c_data)
                    if len(values_list) == 3:
                        break  # Max 3 Stokes

                # Infer lmax if missing
                if lmax_file == -1:
                    nalm = len(values_list[0])
                    # Approximate inverse estimation
                    lmax_file = int(np.sqrt(2 * nalm)) - 1
                    if mmax_file == -1:
                        mmax_file = lmax_file

        if mmax_file <= 0:
            mmax_file = lmax_file

        # --- FINAL ARRAY CONSTRUCTION ---

        # 1. Use YOUR existing static method
        nalm_theory = SphericalHarmonics.num_of_alm_from_lmax(lmax_file, mmax_file)
        num_stokes = len(values_list)

        final_values = np.zeros((num_stokes, nalm_theory), dtype=np.complex128)

        if is_explicit:
            for i, v in enumerate(values_list):
                # Calculate destination index in the flat array (Healpix layout)
                dest_idx = SphericalHarmonics.get_index(lmax_file, v["l"], v["m"])
                final_values[i, dest_idx] = v["re"] + 1j * v["im"]
        else:
            # If standard, assume order is already correct
            for i, arr in enumerate(values_list):
                # Truncate or pad if necessary (safety check)
                limit = min(len(arr), nalm_theory)
                final_values[i, :limit] = arr[:limit]

        return SphericalHarmonics(
            values=final_values,
            lmax=lmax_file,
            mmax=mmax_file,
            units=None,  # Handle if present in header
            coordinates=None,  # Handle if present
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
    # Factory Methods
    # ------------------------------------------------------------------
    @classmethod
    def zeros(
        cls,
        nside: int,
        nstokes: int = 3,
        dtype: type = np.float64,
        units: Units | None = None,
        coordinates: CoordinateSystem | None = None,
        nest: bool = False,
    ) -> "HealpixMap":
        """
        Create a HealpixMap object filled with zeros.

        Parameters
        ----------
        nside : int
            HEALPix resolution.
        nstokes : int, default=3
            Number of Stokes parameters.
        dtype : type, default=np.float64
            Data type of the map.
        units : Units, optional
            Physical units.
        coordinates : CoordinateSystem, optional
            Sky coordinate system.
        nest : bool, default=False
            Ordering (Ring vs Nested).

        Returns
        -------
        HealpixMap
            Instance initialized with zeros.
        """
        npix = cls.nside_to_npix(nside)
        values = np.zeros((nstokes, npix), dtype=dtype)
        return cls(
            values=values,
            nside=nside,
            units=units,
            coordinates=coordinates,
            nest=nest,
        )

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

            >>> from litebird_sim import HealpixMap
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

            >>> from litebird_sim import HealpixMap
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

            >>> from litebird_sim import HealpixMap
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

    def __eq__(self, other: Any) -> bool:
        """
        Exact equality: same geometry, same units, same coordinates
        and identical pixel values.
        """
        if not isinstance(other, HealpixMap):
            return NotImplemented

        return (
            self.is_consistent(other)
            and self.units == other.units
            and self.coordinates == other.coordinates
            and np.array_equal(self.values, other.values)
        )

    def allclose(
        self, other: "HealpixMap", rtol: float = 1.0e-5, atol: float = 1.0e-8
    ) -> bool:
        """
        Check if two HealpixMap objects are equal within a tolerance.

        Checks for consistency in geometry (nside, nest, nstokes), metadata
        (units, coordinates), and numerical values.

        Returns
        -------
        bool
            True if the objects are structurally consistent and their values are
            close within the specified tolerance.
            If the objects are inconsistent, a warning is logged with the specific
            reason, and False is returned.
        """
        if not isinstance(other, HealpixMap):
            raise TypeError("Can only compare with another HealpixMap object.")

        # 1. Check structural consistency (nside, nest, nstokes)
        if self.nside != other.nside:
            log.warning(
                f"allclose mismatch: nside differs ({self.nside} vs {other.nside})"
            )
            return False

        if self.nest != other.nest:
            log.warning(
                f"allclose mismatch: nest ordering differs ({self.nest} vs {other.nest})"
            )
            return False

        if self.nstokes != other.nstokes:
            log.warning(
                f"allclose mismatch: nstokes differs ({self.nstokes} vs {other.nstokes})"
            )
            return False

        # 2. Check units compatibility
        if not self._units_compatible(other):
            log.warning(
                f"allclose mismatch: Incompatible units ({self.units} vs {other.units})"
            )
            return False

        # 3. Check coordinates compatibility
        if not self._coordinates_compatible(other):
            log.warning(
                f"allclose mismatch: Incompatible coordinates ({self.coordinates} vs {other.coordinates})"
            )
            return False

        # 4. Check numerical equality
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    lmax: int | None = None,
    mmax: int | None = None,
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
    lmax: int | None = None,
    mmax: int | None = None,
    nthreads: int = 0,
) -> "SphericalHarmonics":
    r"""
    Estimate spherical harmonic coefficients ($a_{\ell m}$) from a HEALPix map.

    This function performs a spherical harmonic analysis to transform the input map
    space into harmonic space. It uses :func:`ducc0.sht.adjoint_synthesis`
    to compute the summation over pixels and scales the result by the pixel area
    to approximate the integration over the sphere.

    Mathematically, it approximates:

    .. math::
        a_{\ell m} = \int_{\Omega} f(\hat{n}) Y_{\ell m}^*(\hat{n}) d\Omega
        \approx \Omega_{pix} \sum_{p} f(p) Y_{\ell m}^*(p)

    where :func:`ducc0.sht.adjoint_synthesis` computes the summation $\sum f Y^*$,
    and this function multiplies by $\Omega_{pix}$ (``base.pix_area()``).

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
        values=alm * base.pix_area(),
        lmax=lmax_eff,
        mmax=mmax_eff,
        units=map.units,
        coordinates=map.coordinates,
    )


def rotate_alm(
    alms: SphericalHarmonics,
    kind: str | None = None,
    *,
    psi: float | None = None,
    theta: float | None = None,
    phi: float | None = None,
    mmax_out: int | None = None,
    inplace: bool = False,
    nthreads: int = 0,
) -> SphericalHarmonics:
    """
    Rotate spherical harmonic coefficients using ducc0.

    Parameters
    ----------
    alms : SphericalHarmonics
        The spherical harmonics coefficients to rotate.
    kind : str, optional
        String specifying a predefined rotation. Supported values:
        - 'e2g': Ecliptic to Galactic transformation.
        - 'g2e': Galactic to Ecliptic transformation.
    psi : float, optional, keyword-only
        First Euler angle (Z rotation) in radians.
    theta : float, optional, keyword-only
        Second Euler angle (Y rotation) in radians.
    phi : float, optional, keyword-only
        Third Euler angle (Z rotation) in radians.
    mmax_out : int, optional, keyword-only
        Maximum m index for the output coefficients.
        If None, it defaults to alms.lmax (full triangular expansion).
        Must be <= alms.lmax.
    inplace : bool, optional, keyword-only
        If True, modifies the input `alms` object in place.
        Note: In-place rotation is only possible if `mmax_out` is equal to
        `alms.mmax` (output size must match input size).
    nthreads : int, optional, keyword-only
        Number of threads to use for the rotation. Default is 0 (use all available).

    Returns
    -------
    SphericalHarmonics
        The rotated spherical harmonics coefficients.

    Raises
    ------
    ValueError
        - If inputs are inconsistent.
        - If `mmax_out` > `alms.lmax`.
        - If `inplace=True` is requested but `mmax_out` differs from `alms.mmax`.
    """

    # 1. Retrieve Geometry from Input
    lmax_in = alms.lmax
    mmax_in = alms.mmax

    # 2. Determine and Validate mmax_out
    if mmax_out is None:
        # MODIFIED: Default to lmax (full triangular), per instruction.
        mmax_out = lmax_in

    if mmax_out > lmax_in:
        raise ValueError(
            f"Provided mmax_out ({mmax_out}) cannot be larger than input lmax ({lmax_in})."
        )

    # 3. Check for Inplace Feasibility
    # If mmax changes (even if it's just expanding from mmax_in to lmax_in),
    # the array size changes, so inplace is impossible.
    if inplace and (mmax_out != mmax_in):
        raise ValueError(
            f"Cannot perform inplace rotation when changing mmax (Input: {mmax_in}, Output: {mmax_out}). "
            "Set inplace=False to resize the output."
        )

    # 4. Check for No-Op or Invalid combinations
    has_angles = (psi is not None) or (theta is not None) or (phi is not None)

    if kind is not None and has_angles:
        raise ValueError(
            "Cannot specify both 'kind' and explicit Euler angles (psi, theta, phi)."
        )

    # 5. Coordinate determination
    # We determine target_coords BEFORE allocation to pass it to the constructor.
    target_coords = None

    if kind == "e2g":
        if (
            alms.coordinates is not None
            and alms.coordinates != CoordinateSystem.Ecliptic
        ):
            raise ValueError(
                f"Rotation 'e2g' (Ecliptic -> Galactic) requires input in Ecliptic coordinates, "
                f"but input is marked as {alms.coordinates}."
            )
        psi_rot, theta_rot, phi_rot = ECL_TO_GAL_EULER
        target_coords = CoordinateSystem.Galactic

    elif kind == "g2e":
        if (
            alms.coordinates is not None
            and alms.coordinates != CoordinateSystem.Galactic
        ):
            raise ValueError(
                f"Rotation 'g2e' (Galactic -> Ecliptic) requires input in Galactic coordinates, "
                f"but input is marked as {alms.coordinates}."
            )
        psi_rot, theta_rot, phi_rot = GAL_TO_ECL_EULER
        target_coords = CoordinateSystem.Ecliptic

    elif kind is not None:
        raise ValueError(f"Unknown rotation kind '{kind}'. Supported: 'e2g', 'g2e'.")
    else:
        # Generic rotation
        psi_rot = psi if psi is not None else 0.0
        theta_rot = theta if theta is not None else 0.0
        phi_rot = phi if phi is not None else 0.0
        # For generic rotations, the target coordinate system is considered Unknown (None)
        target_coords = None

    # Handle "No-Op" case where rotation is identity AND mmax is unchanged
    if kind is None and not has_angles:
        if mmax_out == mmax_in:
            warnings.warn(
                "No rotation specified and mmax unchanged. Returning input alms."
            )
            # If returning input/copy without rotation, we preserve original coords
            return alms if inplace else alms.copy()

    # 6. Prepare Output Container
    if inplace:
        out_alms = alms
    else:
        # MODIFIED: Allocate new container using target_coords explicitly.
        out_alms = SphericalHarmonics.zeros(
            lmax=lmax_in,
            mmax=mmax_out,
            nstokes=alms.nstokes,
            dtype=alms.values.dtype,
            units=alms.units,
            coordinates=target_coords,  # Using the determined target coords
        )

    # 7. Execution (ducc0)
    for i in range(out_alms.nstokes):
        sht.rotate_alm(
            alms.values[i],  # Input array
            lmax=lmax_in,
            psi=psi_rot,
            theta=theta_rot,
            phi=phi_rot,
            nthreads=nthreads,
            mmax_in=mmax_in,
            mmax_out=mmax_out,
            out=out_alms.values[i],  # Output array (pre-allocated)
        )

    # 8. Update metadata (Critical for inplace operations or fallback safety)
    # MODIFIED: Always set the coordinates to the determined target logic.
    out_alms.coordinates = target_coords

    return out_alms


def synthesize_alm(
    cl_dict: dict[str, np.ndarray],
    lmax: int | None = None,
    mmax: int | None = None,
    rng: np.random.Generator | None = None,
    units: Optional["Units"] = None,
    coordinates: Optional["CoordinateSystem"] = None,
) -> "SphericalHarmonics":
    """
    Generates a set of spherical harmonic coefficients (alm) from power spectra.

    Optimized behavior based on input correlations:
    - 'TT' only: Scalar generation (fastest).
    - 'TT', 'EE', 'BB', 'TE': Block-diagonal generation. Computes (T, E)
      as a correlated 2x2 pair, and B independently.
    - 'TB' or 'EB' present: Full 3x3 covariance generation.

    Parameters
    ----------
    cl_dict : Dict[str, np.ndarray]
        Dictionary containing the power spectra (e.g. 'TT', 'EE', 'BB', 'TE'...).
    lmax : int, optional
        Maximum multipole degree l. If None, it defaults to the minimum lmax
        determined from the lengths of the arrays in cl_dict (len(cl) - 1).
    mmax : int, optional
        Maximum multipole order m. If None, defaults to lmax.
    rng : numpy.random.Generator, optional
        Random number generator instance.
    units : Units, optional
        The physical units of the generated alm.
    coordinates : CoordinateSystem, optional
        The coordinate system of the generated alms (e.g. Ecliptic, Galactic).

    Returns
    -------
    SphericalHarmonics
        Dataclass with .values of shape (nstokes, ncoeffs).

    Raises
    ------
    ValueError
        If lmax is None and cl_dict is empty.
    """

    # 0. Determine lmax if not provided
    if lmax is None:
        if not cl_dict:
            raise ValueError("cl_dict is empty and lmax is not provided.")
        # Arrays in cl_dict include l=0, so max l is len(arr) - 1
        lmax = min(len(arr) for arr in cl_dict.values()) - 1

    if mmax is None:
        mmax = lmax

    if rng is None:
        rng = np.random.default_rng()

    keys = cl_dict.keys()

    # --- 1. Identify Physics Case ---
    pol_keys = {"EE", "BB", "TE", "TB", "EB"}
    has_polarization = any(k in keys for k in pol_keys)

    # Check for parity breaking correlations (TB, EB)
    has_parity_breaking = "TB" in keys or "EB" in keys
    has_bb = "BB" in keys

    # --- 2. Setup Indices (m-slow scheme) ---
    # We need n_coeffs regardless of the method
    # Create l values for every coefficient
    l_indices_list = []
    for m in range(mmax + 1):
        l_indices_list.append(np.arange(m, lmax + 1))
    l_indices = np.concatenate(l_indices_list)
    n_coeffs = len(l_indices)

    # Helper for extracting spectra
    def get_padded_cl(key):
        if key not in cl_dict:
            return np.zeros(lmax + 1)
        cl = cl_dict[key]
        if len(cl) < lmax + 1:
            warnings.warn(
                f"Cl '{key}' len {len(cl)} < lmax+1 ({lmax + 1}). Padding with zeros.",
                UserWarning,
            )
            padded = np.zeros(lmax + 1)
            padded[: len(cl)] = cl
            return padded
        return cl[: lmax + 1]

    # Helper for white noise generation
    def generate_white_noise(shape):
        # Generate Complex Normal(0, 1)
        # Real/Imag parts ~ N(0, 1/sqrt(2)) -> Sum variance is 1
        nr = rng.standard_normal(shape)
        ni = rng.standard_normal(shape)
        white = (nr + 1j * ni) / np.sqrt(2)

        # Correct m=0 modes (first lmax+1 elements) to be real
        # For real modes, we want variance 1 real.
        # Currently nr is N(0, 1).
        m0_count = lmax + 1
        white[..., :m0_count] = nr[..., :m0_count] * (1.0 + 0j)
        return white

    # --- 3. Execution based on Case ---

    # === CASE A: Intensity Only ===
    if not has_polarization:
        # nstokes = 1
        cl_tt = get_padded_cl("TT")

        # Simple scalar scaling: alm = sqrt(Cl) * white_noise
        alm_white = generate_white_noise((1, n_coeffs))

        # Map sqrt(Cl) to indices
        # Ensure non-negative Cl for sqrt
        cl_sqrt = np.sqrt(np.maximum(cl_tt, 0.0))
        scaling_factor = cl_sqrt[l_indices]

        alm_colored = alm_white * scaling_factor

    # === CASE B: Decoupled Polarization (Standard CMB) ===
    # T and E are correlated (2x2), B is independent (scalar)
    elif not has_parity_breaking:
        # nstokes = 3

        # --- Block 1: T and E (2x2 mixing) ---
        cl_tt = get_padded_cl("TT")
        cl_ee = get_padded_cl("EE")
        cl_te = get_padded_cl("TE")

        # Construct 2x2 covariances: [[TT, TE], [TE, EE]]
        # Shape (lmax+1, 2, 2)
        cov_2x2 = np.zeros((lmax + 1, 2, 2))
        cov_2x2[:, 0, 0] = cl_tt
        cov_2x2[:, 1, 1] = cl_ee
        cov_2x2[:, 0, 1] = cov_2x2[:, 1, 0] = cl_te

        # Compute mixing matrices for T, E
        L_matrices_2x2 = np.zeros_like(cov_2x2)
        for l_val in range(lmax + 1):
            C_l = cov_2x2[l_val]
            if np.allclose(C_l, 0, atol=1e-20):
                continue
            u, s, vh = np.linalg.svd(C_l, hermitian=True)
            s = np.maximum(s, 0.0)
            L_matrices_2x2[l_val] = u @ np.diag(np.sqrt(s))

        # Generate noise for T and E
        white_te = generate_white_noise((2, n_coeffs))

        # Apply coloring for T, E
        L_expanded = L_matrices_2x2[l_indices]  # (n_coeffs, 2, 2)
        colored_te_T = np.einsum("isk,ik->is", L_expanded, white_te.T)
        colored_te = colored_te_T.T  # (2, n_coeffs)

        # --- Block 2: B (Scalar) ---
        if has_bb:
            cl_bb = get_padded_cl("BB")
            white_b = generate_white_noise((1, n_coeffs))
            cl_bb_sqrt = np.sqrt(np.maximum(cl_bb, 0.0))
            colored_b = white_b * cl_bb_sqrt[l_indices]
        else:
            colored_b = np.zeros((1, n_coeffs), dtype=np.complex128)

        # Combine
        alm_colored = np.vstack([colored_te, colored_b])

    # === CASE C: Full Polarization (Parity Breaking) ===
    # Everything is correlated (3x3)
    else:
        # nstokes = 3
        cl_tt = get_padded_cl("TT")
        cl_ee = get_padded_cl("EE")
        cl_bb = get_padded_cl("BB")
        cl_te = get_padded_cl("TE")
        cl_tb = get_padded_cl("TB")
        cl_eb = get_padded_cl("EB")

        cov_3x3 = np.zeros((lmax + 1, 3, 3))
        # Diagonals
        cov_3x3[:, 0, 0] = cl_tt
        cov_3x3[:, 1, 1] = cl_ee
        cov_3x3[:, 2, 2] = cl_bb
        # Off-diagonals
        cov_3x3[:, 0, 1] = cov_3x3[:, 1, 0] = cl_te
        cov_3x3[:, 0, 2] = cov_3x3[:, 2, 0] = cl_tb
        cov_3x3[:, 1, 2] = cov_3x3[:, 2, 1] = cl_eb

        L_matrices = np.zeros_like(cov_3x3)
        for l_val in range(lmax + 1):
            C_l = cov_3x3[l_val]
            if np.allclose(C_l, 0, atol=1e-20):
                continue
            u, s, vh = np.linalg.svd(C_l, hermitian=True)
            s = np.maximum(s, 0.0)
            L_matrices[l_val] = u @ np.diag(np.sqrt(s))

        white_noise = generate_white_noise((3, n_coeffs))

        L_expanded = L_matrices[l_indices]
        colored_T = np.einsum("isk,ik->is", L_expanded, white_noise.T)
        alm_colored = colored_T.T

    return SphericalHarmonics(
        values=alm_colored, lmax=lmax, mmax=mmax, units=units, coordinates=coordinates
    )


def compute_cl(
    alm1: "SphericalHarmonics",
    alm2: Optional["SphericalHarmonics"] = None,
    lmax: int | None = None,
    mmax: int | None = None,
    symmetrize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Computes angular power spectra (Cl) from one or two sets of spherical harmonic coefficients.

    Calculates auto-spectra or cross-spectra. Automatically handles inputs with different
    sizes (lmax, mmax) by computing the spectrum up to the minimum common lmax/mmax.

    Parameters
    ----------
    alm1 : SphericalHarmonics
        The first set of spherical harmonics.
    alm2 : SphericalHarmonics, optional
        The second set of spherical harmonics. If None, the auto-spectrum of alm1
        is computed.
    lmax : int, optional
        The maximum l to compute. If None, derived from inputs.
        If larger than input dimensions, it is clamped to the input size with a warning.
    mmax : int, optional
        The maximum m to compute. If None, derived from inputs (or set to lmax).
    symmetrize : bool, default=True
        Only applies when two different objects are passed and nstokes=3.
        If True, returns symmetric cross-spectra (e.g., TE = (T1E2 + E1T2)/2).
        If False, returns all cross-combinations (TE, ET, TB, BT, EB, BE).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of power spectra (values are 1D arrays of size lmax_calc + 1).
    """

    # 1. Handle Inputs and Auto-Correlation Setup
    is_auto = False
    if alm2 is None:
        alm2 = alm1
        is_auto = True

    # 2. Determine Effective Limits for ALM 1
    # Check against alm1 physical limits
    lmax1_eff = lmax if lmax is not None else alm1.lmax
    if lmax1_eff > alm1.lmax:
        warnings.warn(
            f"Requested lmax ({lmax1_eff}) > alm1.lmax ({alm1.lmax}). using {alm1.lmax}.",
            UserWarning,
        )
        lmax1_eff = alm1.lmax

    mmax1_in = alm1.mmax if alm1.mmax is not None else alm1.lmax
    mmax1_eff = mmax if mmax is not None else mmax1_in
    # If mmax was not passed, we might have defaulted to lmax_calc later,
    # but here we check consistency with the object
    if mmax1_eff > mmax1_in:
        warnings.warn(
            f"Requested mmax ({mmax1_eff}) > alm1.mmax ({mmax1_in}). Using {mmax1_in}.",
            UserWarning,
        )
        mmax1_eff = mmax1_in

    # 3. Determine Effective Limits for ALM 2
    lmax2_eff = lmax if lmax is not None else alm2.lmax
    if lmax2_eff > alm2.lmax:
        warnings.warn(
            f"Requested lmax ({lmax2_eff}) > alm2.lmax ({alm2.lmax}). Using {alm2.lmax}.",
            UserWarning,
        )
        lmax2_eff = alm2.lmax

    mmax2_in = alm2.mmax if alm2.mmax is not None else alm2.lmax
    mmax2_eff = mmax if mmax is not None else mmax2_in
    if mmax2_eff > mmax2_in:
        warnings.warn(
            f"Requested mmax ({mmax2_eff}) > alm2.mmax ({mmax2_in}). Using {mmax2_in}.",
            UserWarning,
        )
        mmax2_eff = mmax2_in

    # 4. Compute Intersection (Calculation Limits)
    lmax_calc = min(lmax1_eff, lmax2_eff)
    mmax_calc = min(mmax1_eff, mmax2_eff)

    # Clamp mmax to lmax (physically m cannot exceed l)
    if mmax_calc > lmax_calc:
        mmax_calc = lmax_calc

    # Check Stokes dimensions
    nstokes1 = alm1.values.shape[0]
    nstokes2 = alm2.values.shape[0]
    if nstokes1 != nstokes2:
        raise ValueError(
            f"nstokes mismatch: alm1 has {nstokes1}, alm2 has {nstokes2}. "
            "Mixed dimension spectra are not currently supported."
        )

    # 5. Core Computation Helper
    def _compute_component_cl(arr1, arr2):
        cl = np.zeros(lmax_calc + 1, dtype=np.float64)

        # We need to traverse both arrays.
        # Since they might have different sizes, we track indices separately.
        idx1 = 0
        idx2 = 0

        # Pre-fetch object properties to avoid lookups in loop
        # Note: We use the *intrinsic* lmax of the objects to advance pointers correctly
        lmax1_obj = alm1.lmax
        lmax2_obj = alm2.lmax

        for m in range(mmax_calc + 1):
            # The number of elements we WANT to compute for this m
            # defined by the intersection limit lmax_calc
            count = lmax_calc - m + 1

            # Extract slices
            # arr1 and arr2 are in m-slow order: [ (l=m..lmax_obj), (l=m+1..lmax_obj), ... ]
            # We take the first 'count' elements which correspond to l=m..lmax_calc
            chunk1 = arr1[idx1 : idx1 + count]
            chunk2 = arr2[idx2 : idx2 + count]

            # Cross product
            prod = (chunk1 * chunk2.conj()).real

            # Accumulate (m=0 once, m>0 twice)
            if m == 0:
                cl[m:] += prod
            else:
                cl[m:] += 2.0 * prod

            # Advance indices by the FULL block size of each object
            # To jump to the start of the next m block
            idx1 += lmax1_obj - m + 1
            idx2 += lmax2_obj - m + 1

        # Normalize
        ell = np.arange(lmax_calc + 1, dtype=np.float64)
        cl /= 2.0 * ell + 1.0
        return cl

    # 6. Generate Spectra
    out_cls = {}

    if nstokes1 == 1:
        out_cls["TT"] = _compute_component_cl(alm1.values[0], alm2.values[0])
        return out_cls

    elif nstokes1 == 3:
        # Indices: 0=T, 1=E, 2=B
        out_cls["TT"] = _compute_component_cl(alm1.values[0], alm2.values[0])
        out_cls["EE"] = _compute_component_cl(alm1.values[1], alm2.values[1])
        out_cls["BB"] = _compute_component_cl(alm1.values[2], alm2.values[2])

        # Base Cross terms
        cl_te = _compute_component_cl(alm1.values[0], alm2.values[1])  # T1 x E2
        cl_tb = _compute_component_cl(alm1.values[0], alm2.values[2])  # T1 x B2
        cl_eb = _compute_component_cl(alm1.values[1], alm2.values[2])  # E1 x B2

        if is_auto:
            out_cls["TE"] = cl_te
            out_cls["TB"] = cl_tb
            out_cls["EB"] = cl_eb
        elif symmetrize:
            # Symmetrized Cross
            cl_et = _compute_component_cl(alm1.values[1], alm2.values[0])  # E1 x T2
            cl_bt = _compute_component_cl(alm1.values[2], alm2.values[0])  # B1 x T2
            cl_be = _compute_component_cl(alm1.values[2], alm2.values[1])  # B1 x E2

            out_cls["TE"] = 0.5 * (cl_te + cl_et)
            out_cls["TB"] = 0.5 * (cl_tb + cl_bt)
            out_cls["EB"] = 0.5 * (cl_eb + cl_be)
        else:
            # Full Cross
            out_cls["TE"] = cl_te
            out_cls["TB"] = cl_tb
            out_cls["EB"] = cl_eb
            out_cls["ET"] = _compute_component_cl(alm1.values[1], alm2.values[0])
            out_cls["BT"] = _compute_component_cl(alm1.values[2], alm2.values[0])
            out_cls["BE"] = _compute_component_cl(alm1.values[2], alm2.values[1])

        return out_cls

    else:
        raise ValueError(f"Unsupported number of Stokes parameters: {nstokes1}")


def compute_dl(alm1, alm2=None, lmax=None, mmax=None, symmetrize=True):
    """
    Computes the Dl power spectrum, defined as l(l+1)Cl/2pi.

    This function acts as a wrapper around :func:`compute_cl`, automatically
    applying the scaling factor :math:`\\ell(\\ell+1)/2\\pi` to the output spectra.

    Parameters
    ----------
    alm1 : SphericalHarmonics
        The first set of spherical harmonics coefficients.
    alm2 : SphericalHarmonics, optional
        The second set of spherical harmonics coefficients. If None, the auto-spectrum
        of `alm1` is computed.
    lmax : int, optional
        The maximum degree l to compute. If None, derived from inputs.
    mmax : int, optional
        The maximum order m to compute. If None, derived from inputs.
    symmetrize : bool, default=True
        Only applies when two different objects are passed and nstokes=3.
        If True, returns symmetric cross-spectra (e.g., TE = (T1E2 + E1T2)/2).

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the Dl spectra. The keys follow the same
        convention as compute_cl (e.g., 'TT', 'EE', 'BB', 'TE').
    """
    # 1. Compute the Cl spectra by calling the existing function
    cls = compute_cl(alm1, alm2=alm2, lmax=lmax, mmax=mmax, symmetrize=symmetrize)

    dls = {}
    # 2. Apply the scaling factor l(l+1)/2pi
    for key, cl_array in cls.items():
        # Generate the l (ell) array based on the length of the spectrum
        ell = np.arange(len(cl_array))

        # Calculate the scaling factor. Note: for l=0, the factor is 0.
        factor = ell * (ell + 1) / (2 * np.pi)

        dls[key] = cl_array * factor

    return dls


def pixel_window(nside, lmax=None, pol=False):
    """
    Returns the pixel window function compatible with SphericalHarmonics.convolve.

    Parameters
    ----------
    nside : int
        HEALPix Nside.
    lmax : int, optional
        Maximum multipole. If None, defaults to 4*nside.
    pol : bool, optional
        If True, returns TEB window functions (shape 3, lmax+1).
        If False, returns T window function (shape lmax+1,).

    Returns
    -------
    np.ndarray
        Pixel window array.
    """
    # Locate the data file
    pkl_path = Path(__file__).parent / "datautils" / "pixwin.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"pixwin.pkl not found at {pkl_path}")

    # Load the database
    db = np.load(pkl_path, allow_pickle=True)

    if nside not in db:
        raise KeyError(f"Nside {nside} not available in pixel window database.")

    # Handle lmax default
    if lmax is None:
        lmax = 4 * nside

    # 4. Fetch full arrays
    pw_t_full = db[nside]["T"]
    pw_e_full = db[nside]["E"]  # In HEALPix, E window applies to Polarization (E and B)

    # Check bounds
    available_lmax = len(pw_t_full) - 1
    if lmax > available_lmax:
        raise ValueError(
            f"Requested lmax ({lmax}) > available lmax ({available_lmax}) for Nside {nside}"
        )

    # Slice data
    pw_t = pw_t_full[: lmax + 1]

    if not pol:
        # Return T only
        return pw_t

    # Return TEB (T, E, B)
    pw_pol = pw_e_full[: lmax + 1]
    return np.array([pw_t, pw_pol, pw_pol])


def read_cls_from_fits(path: str | Path) -> dict[str, np.ndarray]:
    """
    Reads CMB power spectra from a FITS file using Astropy.

    It first attempts to map specific column names (TTYPE) to standard keys:
    TT, EE, BB, TE, TB, EB.

    If no column names match the expected mapping, it falls back to a positional
    assumption based on the number of columns:
    - 6 columns: Assumes [TT, EE, BB, TE, TB, EB]
    - 4 columns: Assumes [TT, EE, BB, TE]

    Parameters
    ----------
    path : str or Path
        Path to the FITS file containing the Cls.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the power spectra with keys 'TT', 'EE', 'BB', etc.
    """
    # Mapping for standard CAMB/Healpy FITS headers
    mapping = {
        "TEMPERATURE": "TT",
        "GRADIENT": "EE",
        "CURL": "BB",
        "G-T": "TE",
        "C-T": "TB",
        "C-G": "EB",
    }

    # Fallback orders if headers are non-standard (e.g., COL1, COL2...)
    fallback_order_6 = ["TT", "EE", "BB", "TE", "TB", "EB"]
    fallback_order_4 = ["TT", "EE", "BB", "TE"]

    cl = {}

    with fits.open(path) as hdul:
        # Assuming data is in extension 1 (standard for Cls FITS)
        data = hdul[1].data
        columns = data.columns

        # Strategy 1: Attempt to identify columns by name
        for col in columns:
            name = col.name.strip().upper()
            for key_fits, key_out in mapping.items():
                # Check if the standard keyword is part of the column name
                if key_fits in name:
                    cl[key_out] = data[col.name].flatten()
                    break

        # Strategy 2: Fallback to positional arguments if Strategy 1 failed
        if not cl:
            n_cols = len(columns)

            if n_cols == 6:
                log.warning(
                    f"No matching headers found in {path}. Assuming default 6-field order: {fallback_order_6}"
                )
                for i, key in enumerate(fallback_order_6):
                    cl[key] = data[columns[i].name].flatten()

            elif n_cols == 4:
                log.warning(
                    f"No matching headers found in {path}. Assuming default 4-field order: {fallback_order_4}"
                )
                for i, key in enumerate(fallback_order_4):
                    cl[key] = data[columns[i].name].flatten()

            else:
                raise ValueError(
                    f"Could not parse Cls from {path}. No recognized column names found "
                    f"and column count ({n_cols}) does not match standard fallback shapes (4 or 6)."
                )

    return cl


def lin_comb_cls(
    cls1: dict[str, np.ndarray],
    cls2: dict[str, np.ndarray] | None = None,
    s1: float = 1.0,
    s2: float = 1.0,
    keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute a linear combination of power spectra dictionaries or scale a single one.

    This function performs the operation:
    result = s1 * cls1 + s2 * cls2 (if cls2 is provided)
    result = s1 * cls1 (if cls2 is None)

    Parameters
    ----------
    cls1 : Dict[str, np.ndarray]
        First dictionary of power spectra {key: np.array}.
    cls2 : Dict[str, np.ndarray], optional
        Second dictionary of power spectra. If None, only cls1 is processed.
    s1 : float, optional
        Scaling factor applied to `cls1`. Default is 1.0.
    s2 : float, optional
        Scaling factor applied to `cls2`. Default is 1.0.
        Ignored if `cls2` is None.
    keys : List[str], optional
        Specific list of keys to process. If None, uses keys from cls1
        (or intersection if cls2 is provided).

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the resulting spectra.

    Raises
    ------
    ValueError
        If cls2 is provided and array lengths for the same key do not match.
    """
    # Define which keys to operate on
    if keys is None:
        if cls2 is not None:
            target_keys = cls1.keys() & cls2.keys()
        else:
            target_keys = cls1.keys()
    else:
        # Filter keys to ensure they exist in cls1 (and cls2 if provided)
        target_keys = [k for k in keys if k in cls1 and (cls2 is None or k in cls2)]
        missing = set(keys) - set(target_keys)
        if missing:
            log.warning(f"Requested keys {missing} not found in input(s).")

    result = {}
    for k in target_keys:
        val = s1 * cls1[k]

        if cls2 is not None:
            if len(cls1[k]) != len(cls2[k]):
                raise ValueError(
                    f"Length mismatch for key '{k}': {len(cls1[k])} vs {len(cls2[k])}."
                )
            val += s2 * cls2[k]

        result[k] = val

    return result
