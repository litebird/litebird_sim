# -*- encoding: utf-8 -*-

"""
This file contains a port of the Grasp2alm repository <https://github.com/yusuke-takase/grasp2alm>,
created by Yusuke Takase. We decided to incorporate it into LBS so that we can make the API more
compatible with the framework, while preserving the original source code repository.

If you are patching a bug in this file, be sure to check if the same bug was present in
Yusuke’s repository; if it is, please upload your patches there, so that both codes can benefit
from your work.
"""

import copy
import typing
import warnings
from dataclasses import dataclass

import ducc0.healpix
import ducc0.sht
from numba import njit
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .beam_convolution import SphericalHarmonics
from .healpix import npix_to_nside, nside_to_npix, num_of_alms

REASON_DESCRIPTION = {
    1: "Approximate solution found",
    2: "Approximate least-square solution found",
    3: "Too large condition number",
    7: "Maximum number of iterations reached",
}


class BeamHealpixMap:
    """Represents a beam map.

    Fields:
        nside (`int`): The resolution parameter of the map.
        map (`numpy.ndarray`): The beam map data, in RING order.
    """

    def __init__(self, healpix_map):
        self.nside = npix_to_nside(healpix_map.shape[1])
        self.map = healpix_map
        self.base = ducc0.healpix.Healpix_Base(self.nside, "RING")

    def _check_reason(
        self, reason: int, iter_count: int, residual_norm: float, quality: float
    ) -> None:
        assert reason in (1, 2), (
            f"pseudo_analysis failed, reason: {REASON_DESCRIPTION[reason]}, {iter_count=}, {residual_norm=}, {quality=}"
        )

    def to_alm(
        self,
        lmax: int,
        mmax: int,
        epsilon=1e-8,
        max_num_of_iterations=20,
    ) -> np.ndarray:
        """Converts the beam map to spherical harmonic coefficients.

        Args:
            lmax (`int`): Maximum l value for the spherical harmonic expansion.
            mmax (`int`): Maximum m value for the spherical harmonic expansion.
            epsilon (`float`): Precision of the result
            max_num_of_iterations (`int`): Maximum number of iterations

        Returns:
            `numpy.ndarray`: The spherical harmonic coefficients, as a (3, N) array.

        Raises:
            AssertionError: If ``lmax`` is greater than 3*``nside``-1 or if ``mmax`` is greater than ``lmax``.

        """

        if not self.map.shape[0] <= 3:
            raise ValueError(
                "Error in BeamMap.to_alm: map has more than 3 Stokes parameters"
            )

        geom = self.base.sht_info()

        alm = np.empty((3, num_of_alms(lmax, mmax)), dtype=np.complex128)
        (_, reason, iter_count, residual_norm, quality) = ducc0.sht.pseudo_analysis(
            map=self.map[0].reshape(1, -1),  # Make this a 2D matrix
            alm=alm[0, :].reshape(1, -1),
            lmax=lmax,
            mmax=mmax,
            spin=0,
            nthreads=0,
            maxiter=max_num_of_iterations,
            epsilon=epsilon,
            **geom,
        )
        self._check_reason(
            reason=reason,
            iter_count=iter_count,
            residual_norm=residual_norm,
            quality=quality,
        )

        ducc0.sht.pseudo_analysis(
            map=self.map[1:],
            alm=alm[1:, :],
            lmax=lmax,
            mmax=mmax,
            spin=2,
            nthreads=0,
            maxiter=max_num_of_iterations,
            epsilon=epsilon,
            **geom,
        )
        self._check_reason(
            reason=reason,
            iter_count=iter_count,
            residual_norm=residual_norm,
            quality=quality,
        )

        return alm


@njit
def _fill_points(phi_values, theta_values, points) -> None:
    # This is equivalent to the combination of numpy.meshgrid() and
    # numpy.stack(), but it uses far less memory and is much faster

    nphi = len(phi_values)
    ntheta = len(theta_values)
    npoints = points.shape[0]

    assert nphi * ntheta == npoints
    assert points.shape[1] == 2

    i = 0
    for phi in phi_values:
        for theta in theta_values:
            points[i, 0] = phi
            points[i, 1] = theta

            i += 1


@dataclass
class BeamStokesPolar:
    """Representation of a beam as a set of Stokes parameters sampled over a sphere using a θφ grid

    This class stores the four Stokes parameters ($I$, $Q$, $U$, and $V$) for a beam on a spherical
    polar grid parameterized by (:math:`\\theta`,:math:`\\phi`). This type is used as input to the
    spherical harmonic transform. Internally, this package uses the third Ludwig's definition for the
    polarization basis with the co-polar direction (positive :math:`Q`) aligned with the y-axis.

    This class assumes that the :math:`\\phi` angles spans 2π. This is the reason why there are
    no ``phi_rad_min``/``phi_rad_max`` parameters.

    Args:
        phi_values (`np.typing.ArrayLike`): List of :math:`\\phi` values.
        theta_values (`.typing.ArrayLike`): List of :math:`\theta` values.
        stokes (`numpy.ndarray`): Array of shape (4, ``nphi``, ``ntheta``) containing the four Stokes
            parameters (:math:`I`,:math:`Q`,:math:`U`,:math:`V`).
        polar_basis_flag (`bool`): If ``True``, the :math:`Q` and :math:`U` parameters have been rotated
            so that they are expressed in a polar basis. This is suitable if you are going to
            compute the harmonic coefficients on them. By default, this is ``False``, because typically
            :class:`BeamPolar` instances are created out of TICRA GRASP files, which use Ludwig's third
            definition of the polarization.
    """

    def __init__(
        self,
        phi_values: np.typing.ArrayLike,
        theta_values: np.typing.ArrayLike,
        polar_basis_flag: bool = False,
    ):
        self.phi_values = phi_values
        self.theta_values = theta_values
        self.stokes = np.zeros(
            (4, len(self.phi_values), len(self.theta_values)), dtype=np.float64
        )
        self.polar_basis_flag = polar_basis_flag

    def convert_to_polar_basis(self) -> "BeamStokesPolar":
        """Rotates :math:`Q` and :math:`U` Stokes parameters from the co-cross basis to the polar basis.

        The :math:`Q` and :math:`U` Stokes parameters are usually represented in the
        co-cross basis, where the co-polar direction is aligned with the
        y-axis (consistent with Ludwig 3 convention). For the purposes of
        extracting the spherical harmonic coefficients, it is more useful
        to represent them in the polar basis. Unlike the original LevelS's method, this function operates
        on a copy of the input :class:`.BeamPolar`, so ``self`` is not changed.

        Returns:
            :class:`.BeamPolar`: A new instance of ``BeamPolar`` with the rotated stokes beam.

        """
        beam_copy = copy.deepcopy(self)

        valid_theta_indices = self.theta_values != 0.0

        cos2phi = np.cos(2.0 * self.phi_values)
        sin2phi = np.sin(2.0 * self.phi_values)

        q = self.stokes[1, :, valid_theta_indices]
        u = self.stokes[2, :, valid_theta_indices]

        beam_copy.stokes[1, :, valid_theta_indices] = (
            q * cos2phi[None, :] + u * sin2phi[None, :]
        )
        beam_copy.stokes[2, :, valid_theta_indices] = (
            -q * sin2phi[None, :] + u * cos2phi[None, :]
        )
        beam_copy.polar_basis_flag = True
        return beam_copy

    def to_map(
        self,
        nside: int,
        nstokes: int = 3,
        unseen_pixel_value: float = 0.0,
        interp_method: str = "linear",
    ) -> BeamHealpixMap:
        """Convert the :class:`.BeamPolar` to a :class:`.BeamMap`.

        Args:
            nside (`int`): The nside parameter for the HEALPix map.
            nstokes (`int`): Number of Stokes parameters.
            unseen_pixel_value (`float`): Value to fill outside the valid theta range.
            interp_method (`str`): Interpolation method to use. Default is 'linear'.
                Supported are 'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'.

        Returns:
            :class:`.BeamMap`: A new instance of ``BeamMap`` representing the beam map.

        """
        base = ducc0.healpix.Healpix_Base(nside, "RING")
        npix = nside_to_npix(nside)

        thetaphi = base.pix2ang(pix=np.arange(npix))
        theta = thetaphi[:, 0]
        phi = thetaphi[:, 1]

        theta = theta[theta <= np.max(self.theta_values)]
        phi = phi[: len(theta)]

        beam_map = np.full((nstokes, npix), unseen_pixel_value, dtype=float)

        beam_polar = self.convert_to_polar_basis()
        _, ntheta, nphi = beam_polar.stokes.shape
        num_of_points = ntheta * nphi
        points = np.empty((num_of_points, 2), dtype=np.float64)

        _fill_points(
            phi_values=self.phi_values, theta_values=self.theta_values, points=points
        )
        for stokes_idx in range(nstokes):
            # Create a 2D interpolator for the beam stokes values
            interpolator = LinearNDInterpolator(
                points=points,
                values=beam_polar.stokes[stokes_idx, :, :].flatten(),
                fill_value=0.0,
            )

            # Use the interpolator to get the beam values at the given theta and phi
            beam_map[stokes_idx, : len(theta)] = interpolator(np.array([phi, theta]).T)

        return BeamHealpixMap(beam_map)


class BeamGrid:
    """Class to hold the data loaded from a TICRA GRASP beam grid file

    This class only supports polar spherical grids in the far field region.

    Args:
        nset (`int`): Number of field sets or beams (this class only
            supports *one* field)
        klimit (`int`): Specification of limits in a 2D grid.
        icomp (`int`): Control parameter of field components. Only types
            3 (copolar-crosspolar) and 9 (total power and
            :math:`\\sqrt(\\text{RHC}/\\text{LHC}` are supported)
        num_of_components (`int`): Number of field components (only ``ncomp==2`` is supported)
        grid_type (`int`): Control parameter of field grid type (only ``igrid==7`` is supported)
        ix (`int`): Centre of set or beam No. :math:`i`.
        iy (`int`): Centre of set or beam No. :math:`i`.
        xs (`float`): Start x-coordinate of the grid.
        ys (`float`): Start y-coordinate of the grid.
        xe (`float`): End x-coordinate of the grid.
        ye (`float`): End y-coordinate of the grid.
        nx (`int`): Number of steps for the x-coordinate
        ny (`int`): Number of steps for the y-coordinate
        frequency (`float`): Value of the frequency.
        frequency_unit (`str`): Measurement unit for ``frequency``.
        amp (`numpy.ndarray` | None): 2D array of complex amplitudes [:math:`\\theta`, :math:`\\phi`].
    """

    def __init__(self, file_obj: typing.TextIO):
        """Initialize the BeamGrid object."""

        self.num_of_components = 0
        self.grid_type = 0
        self.ix = 0
        self.iy = 0
        self.xs = 0.0
        self.ys = 0.0
        self.xe = 0.0
        self.ye = 0.0
        self.nx = 0
        self.ny = 0
        self.frequency = 0.0
        self.frequency_unit = ""
        self.amp = None

        while True:
            line = file_obj.readline().strip()
            if line[0:4] == "++++":
                break
            elif "FREQUENCIES [" in line:
                self.frequency_unit = line.split("[")[1].split("]")[0]
                self.frequency = float(file_obj.readline().strip())

        file_type = int(file_obj.readline())
        if not file_type == 1:
            raise ValueError("Unknown Grasp grid format, KTYPE != 1")

        line = file_obj.readline().split()
        number_of_field_sets = int(line[0])
        self.field_component_type = int(line[1])
        self.num_of_components = int(line[2])
        self.grid_type = int(line[3])

        if number_of_field_sets > 1:
            warnings.warn("Warning: NSET > 1, only reading first beam in file")

        line = file_obj.readline().split()
        self.ix = int(line[0])
        self.iy = int(line[1])
        i = 2
        while i <= number_of_field_sets:
            file_obj.readline()
            i += 1

        line = file_obj.readline().split()
        self.xs = float(line[0])
        self.ys = float(line[1])
        self.xe = float(line[2])
        self.ye = float(line[3])

        line = file_obj.readline().split()
        self.nx = int(line[0])
        self.ny = int(line[1])
        klimit = int(line[2])
        self.amp = np.zeros((self.num_of_components, self.nx, self.ny), dtype=complex)

        for j in range(self.ny):
            if klimit == 1:
                is_val, in_val = (int(x) for x in file_obj.readline().split())
            else:
                is_val, in_val = 1, self.nx

            for i in range(is_val, in_val + 1):
                line = file_obj.readline().split()
                if not line:
                    raise ValueError("Unexpected end of file")
                if any(np.isnan(list(map(float, line)))):
                    raise ValueError(
                        "Encountered a NaN value in Amplitude. Please check your input."
                    )

                self.amp[0, i - 1, j] = float(line[0]) + float(line[1]) * 1j
                self.amp[1, i - 1, j] = float(line[2]) + float(line[3]) * 1j

    def to_polar(self, copol_axis="x") -> BeamStokesPolar:
        """Converts beam in polar grid format into Stokes
        parameters on a polar grid. The value of copol
        specifies the alignment of the co-polar basis
        ('x' or 'y') of the input GRASP file.

        Args:
            copol_axis (`str`): The copolarization axis. Must be 'x' or 'y'.

        Returns:
            BeamStokesPolar: The beam grid in polar coordinates.

        Raises:
            ValueError: If the beam is not in the supported GRASP grid format.

        """
        copol_axis = copol_axis.lower()
        if not self.num_of_components == 2:
            raise ValueError(
                "Error in BeamGrid.to_polar: beam is not in linear 'co' and 'cx' components"
            )
        if copol_axis not in ["x", "y"]:
            raise ValueError(
                "Error in BeamGrid.to_polar: copol_axis must be 'x' or 'y'"
            )

        if self.grid_type == 7:
            nphi = self.nx - 1
            ntheta = self.ny
            theta_rad_min = np.deg2rad(self.ys)
            theta_rad_max = np.deg2rad(self.ye)
            swap_theta = theta_rad_min > theta_rad_max
            if swap_theta:
                warnings.warn("swapping theta direction")
                theta_rad_min = np.deg2rad(self.ye)
                theta_rad_max = np.deg2rad(self.ys)

            theta_values = np.linspace(theta_rad_min, theta_rad_max, ntheta)
            phi_values = np.linspace(self.xs, self.xe, nphi)
        elif self.grid_type == 5:
            azimuth_values = np.linspace(
                np.deg2rad(self.xs), np.deg2rad(self.xe), self.nx
            )
            elevation_values = np.linspace(
                np.deg2rad(self.ys), np.deg2rad(self.ye), self.ny
            )

            theta_values = np.sqrt(azimuth_values**2 + elevation_values**2)
            phi_values = np.arctan2(elevation_values, -azimuth_values)
            swap_theta = False
        else:
            raise ValueError(
                f"Error in BeamGrid.to_polar: beam is not on theta-phi grid ({self.grid_type=} ≠ 5, 7)"
            )

        beam_polar = BeamStokesPolar(
            theta_values=theta_values,
            phi_values=phi_values,
        )

        if self.field_component_type == 3:
            if copol_axis == "x":
                sign = -1
            elif copol_axis == "y":
                sign = 1
            else:
                raise ValueError("Error in bm_grid2polar: unknown value for copol")
            co = self.amp[0, :, :]
            cx = self.amp[1, :, :]
            mod_co_squared = np.abs(co) ** 2
            mod_cx_squared = np.abs(cx) ** 2
            acaxs = co * np.conj(cx)
            beam_polar.stokes[0, :, :] = mod_co_squared + mod_cx_squared
            beam_polar.stokes[1, :, :] = sign * (mod_co_squared - mod_cx_squared)
            beam_polar.stokes[2, :, :] = sign * 2.0 * np.real(acaxs)
            beam_polar.stokes[3, :, :] = 2.0 * np.imag(acaxs)
            if swap_theta:
                beam_polar.stokes = beam_polar.stokes[:, :, ::-1]
        elif self.field_component_type == 9:
            co = self.amp[1, :-1, :]
            mod_co_squared = np.abs(co) ** 2
            beam_polar.stokes[0, :, :] = mod_co_squared
            beam_polar.stokes[1, :, :] = 0.0
            beam_polar.stokes[2, :, :] = 0.0
            beam_polar.stokes[3, :, :] = 0.0
            if swap_theta:
                beam_polar.stokes = beam_polar.stokes[:, :, ::-1]
        else:
            raise ValueError(
                "Error in grid2square: beam is not in supported grid sub-format"
            )
        return beam_polar


class BeamCut:
    """Class to hold the data from a beam cut file of GRASP.

    This class only supports polar spherical cuts in the far field region.

    Args:
        vini (`float`): Initial value.
        vinc (`float`): Increment.
        vnum (`int`): Number of values in cut.
        c (`numpy.ndarray`): Constant.
        icomp (`int`): Polarization control parameter.
        icut (`int`): Control parameter of cut.
        ncomp (`int`): Number of field components.
        ncut (`int`): Number of cuts.
        amp (`numpy.ndarray` | None): Amplitude, with a shape (2, num_of_phi_cuts, n_theta).
            The two fields are the complex amplitudes of the E_co and E_cx components of the
            electric field
    """

    def __init__(self, file_obj: typing.TextIO):
        """
        Initializes a BeamCut object.

        Args:
            file_obj (`file`): A file object containing the GRASP cut file
        """

        # First pass: go through the file and count how many cuts it contains. Update
        # the list of values for φ in `phi_values`
        phi_values = []
        self.ncomp = None  # type: int | None
        self.theta0_deg = None  # type: float | None
        self.delta_theta_deg = None  # type: float | None
        self.n_theta = None  # type: int | None
        self.num_of_phi_cuts = 0
        while True:
            # Read the header line and throw it away
            line = file_obj.readline().strip()
            if not line:
                break

            # Read the line containing the metadata for this cut
            line = file_obj.readline()
            data = line.split()
            assert len(data) == 7, f'Line "{line}" is not a valid header'
            cur_vini = float(data[0])
            cur_vinc = float(data[1])
            cur_vnum = int(data[2])
            cur_phi = float(data[3])
            cur_icomp = int(data[4])
            cur_icut = int(data[5])
            cur_ncomp = int(data[6])

            if cur_icomp != 3:
                raise ValueError(
                    "Only GRASP cuts with ICOMP=3 (E_co/E_cx) are accepted"
                )

            if cur_icut != 1:
                raise ValueError(
                    "Only GRASP cuts with ICUT=1 (θ varies faster than φ) are accepted"
                )

            if cur_ncomp != 2:
                raise ValueError(
                    "Only GRASP cuts containing far field components are accepted"
                )

            if self.theta0_deg is None:
                self.theta0_deg = cur_vini
            else:
                if self.theta0_deg != cur_vini:
                    raise ValueError(
                        "The initial θ value (VINI) is not the same across cuts"
                    )

            if self.delta_theta_deg is None:
                self.delta_theta_deg = cur_vinc
            else:
                if self.delta_theta_deg != cur_vinc:
                    raise ValueError(
                        "The value of Δθ (VINC) is not the same across cuts"
                    )

            if self.n_theta is None:
                self.n_theta = cur_vnum
            else:
                if self.n_theta != cur_vnum:
                    raise ValueError(
                        "The value of n_θ (VNUM) is not the same across cuts"
                    )

            if self.ncomp is None:
                self.ncomp = cur_ncomp
            else:
                if self.ncomp != cur_ncomp:
                    raise ValueError(
                        "The value of NCOMP (either 2 or 3) is not the same across cuts"
                    )

            phi_values.append(cur_phi)
            self.num_of_phi_cuts += 1

            # Skip all the lines containing the field, for the moment:
            # we will read them back during the second pass
            for theta_idx in range(self.n_theta):
                _ = file_obj.readline()

        if self.n_theta % 2 == 0:
            raise ValueError("The number of pixels in a cut (VNUM) must be odd.")

        self.phi_values_rad = np.deg2rad(phi_values)
        self.theta_values_rad = np.deg2rad(
            self.theta0_deg + self.delta_theta_deg * np.arange(self.n_theta)
        )

        # Allocate a 3D grid for the data from all the cuts, with this shape:
        # (2, θ_steps, φ_steps)
        self.amp = np.empty(
            shape=(self.ncomp, self.num_of_phi_cuts, self.n_theta), dtype=np.complex128
        )

        # Go back to the beginning of the file and fill `self.amp`
        file_obj.seek(0)
        for phi_idx in range(len(self.phi_values_rad)):
            _ = file_obj.readline()  # Throw away the header and the metadata
            _ = file_obj.readline()

            for theta_idx in range(self.n_theta):
                line = file_obj.readline()
                data = line.split()
                tmp1, tmp2, tmp3, tmp4 = map(float, data)
                if any(np.isnan([tmp1, tmp2, tmp3, tmp4])):
                    raise ValueError(
                        "Encountered a NaN value in Amplitude. Please check your input."
                    )

                self.amp[0, phi_idx, theta_idx] = complex(
                    tmp1, tmp2
                )  # E field (complex)
                self.amp[1, phi_idx, theta_idx] = complex(
                    tmp3, tmp4
                )  # H field (complex)

    def to_polar(self, copol_axis: str = "x") -> BeamStokesPolar:
        """Converts beam in "cut" format to Stokes parameters
        on a polar grid.  Assumes that cuts are evenly spaced
        in theta. The value of `copol_axis` specifies the alignment
        of the co-polar basis ('x' or 'y') of the input GRASP file.

        Args:
            copol_axis (`str`): The axis of co-polarization. Must be either 'x' or 'y'.

        Returns:
            BeamStokesPolar: The beam in polar coordinates.

        Raises:
            ValueError: If the beam is not in the expected format.
        """
        copol_axis = copol_axis.lower()

        if copol_axis not in ("x", "y"):
            raise ValueError(
                f"Error in BeamCut.to_polar: copol_axis must be 'x' or 'y', not '{copol_axis}'"
            )

        if copol_axis == "x":
            sign = -1
        elif copol_axis == "y":
            sign = 1
        else:
            raise ValueError(f"{copol_axis=} can only be 'x' or 'y'")

        co = self.amp[0, :, :]
        cx = self.amp[1, :, :]

        mod_co_squared = np.abs(co) ** 2
        mod_cx_squared = np.abs(cx) ** 2
        cross = co * np.conj(cx)

        beam_polar = BeamStokesPolar(
            theta_values=self.theta_values_rad,
            phi_values=self.phi_values_rad,
        )
        beam_polar.stokes[0, :, :] = mod_co_squared + mod_cx_squared
        beam_polar.stokes[1, :, :] = sign * (mod_co_squared - mod_cx_squared)
        beam_polar.stokes[2, :, :] = sign * 2.0 * np.real(cross)
        beam_polar.stokes[3, :, :] = 2.0 * np.imag(cross)
        return beam_polar


def _grasp2alm(
    file_obj: typing.TextIO,
    beam_class: typing.Type[BeamCut] | typing.Type[BeamGrid],
    nside: int,
    unseen_pixel_value: float = 0.0,
    interp_method: str = "linear",
    copol_axis: str = "x",
    lmax: int | None = None,
    mmax: int | None = None,
    epsilon: float = 1e-8,
    max_num_of_iterations: int = 20,
) -> SphericalHarmonics:
    if not lmax:
        lmax = 3 * nside // 2
    if not mmax:
        mmax = lmax

    beam = beam_class(file_obj)
    beam_polar = beam.to_polar(copol_axis)
    beam_map = beam_polar.to_map(
        nside, unseen_pixel_value=unseen_pixel_value, interp_method=interp_method
    )
    alm = beam_map.to_alm(
        lmax=lmax,
        mmax=mmax,
        epsilon=epsilon,
        max_num_of_iterations=max_num_of_iterations,
    )
    return SphericalHarmonics(values=alm, lmax=lmax, mmax=mmax)


def ticra_cut_to_alm(*args, **kwargs) -> SphericalHarmonics:
    """Convert a GRASP ``.cut`` file to a spherical harmonic coefficients of beam map.

    Args:
        file_obj (`file`): File object to read
        nside (`int`): Resolution parameter for the output beam map.
        lmax (`int`): The desired lmax parameters for the analysis.
        mmax (`int`): The desired mmax parameters for the analysis.
        unseen_pixel_value (`float`): Value to assign to pixels outside
            the valid theta range.
        interp_method (`str`): Interpolation method to use. Default is 'linear'.
                Supported are 'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'.
        copol_axis (`str`, `optional`): Axis of the co-polarization
            component. Defaults to 'x'.
        lmax (`int`, optional): Maximum value for ℓ
        mmax (`int`, optional): Maximum value for m
        epsilon (`float`, optional): Target precision
        max_num_of_iterations (`int`, optional): Maximum number of iterations

    Returns:
        alm (`numpy.ndarray`): Spherical harmonic coefficients of the beam map.

    Raises:
        ValueError: If the file format is unknown.

    """
    kwargs.pop("beam_class", None)
    return _grasp2alm(
        *args,
        **kwargs,
        beam_class=BeamCut,
    )


def ticra_grid_to_alm(*args, **kwargs) -> SphericalHarmonics:
    """Convert a GRASP ``.grd`` file to a spherical harmonic coefficients of beam map.

    Args:
        file_obj (`file`): File object to read
        nside (`int`): Resolution parameter for the output beam map.
        lmax (`int`): The desired lmax parameters for the analysis.
        mmax (`int`): The desired mmax parameters for the analysis.
        outOftheta_val (`float`): Value to assign to pixels outside
            the valid theta range.
        interp_method (`str`): Interpolation method to use. Default is 'linear'.
                Supported are 'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'.
        copol_axis (`str`, `optional`): Axis of the co-polarization
            component. Defaults to 'x'.
        lmax (`int`, optional): Maximum value for ℓ
        mmax (`int`, optional): Maximum value for m
        epsilon (`float`, optional): Target precision
        max_num_of_iterations (`int`, optional): Maximum number of iterations
    Returns:
        alm (`numpy.ndarray`): Spherical harmonic coefficients of the beam map.

    Raises:
        ValueError: If the file format is unknown.

    """
    kwargs.pop("beam_class", None)
    return _grasp2alm(
        *args,
        **kwargs,
        beam_class=BeamGrid,
    )
