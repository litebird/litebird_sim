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
import numpy as np
from scipy.interpolate import RegularGridInterpolator

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


@dataclass
class BeamPolar:
    """Representation of a beam as a set of Stokes parameters sampled over a sphere

    This class stores the four Stokes parameters ($I$, $Q$, $U$, and $V$) for a beam on a spherical
    polar grid parameterized by (:math:`\\theta`,:math:`\\phi`). This type is used as input to the
    spherical harmonic transform. Internally, this package uses the third Ludwig's definition for the
    polarization basis with the co-polar direction (positive :math:`Q`) aligned with the y-axis.

    This class assumes that the :math:`\phi` angles spans 2π. This is the reason why there are
    no ``phi_rad_min``/``phi_rad_max`` parameters.

    Args:
        nphi (`int`): Number of :math:`\\phi` values.
        ntheta (`int`): Number of :math:`\theta` values.
        theta_rad_min (`float`): Minimum :math:`\\theta` value in radians.
        theta_rad_max (`float`): Maximum :math:`\\theta` value in radians.
        stokes (`numpy.ndarray`): Array of shape (4, ``nphi``, ``ntheta``) containing the four Stokes
            parameters (:math:`I`,:math:`Q`,:math:`U`,:math:`V`).
        polar_basis (`bool`): If ``True``, the :math:`Q` and :math:`U` parameters have been rotated
            so that they are expressed in a polar basis. This is suitable if you are going to
            compute the harmonic coefficients on them. By default this is ``False``, because typically
            :class:`BeamPolar` instances are created out of TICRA GRASP files, which use Ludwig's third
            definition of the polarization.
    """

    def __init__(
        self,
        nphi: int,
        ntheta: int,
        theta_rad_min: float,
        theta_rad_max: float,
        polar_basis: bool = False,
    ):
        self.nphi = nphi
        self.ntheta = ntheta
        self.theta_rad_min = theta_rad_min
        self.theta_rad_max = theta_rad_max
        self.stokes = np.zeros((4, nphi, ntheta), dtype=float)
        self.polar_basis = polar_basis

    def convert_to_polar_basis(self) -> "BeamPolar":
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
        phi_step = 2 * np.pi / self.nphi
        theta_step = (self.theta_rad_max - self.theta_rad_min) / (self.ntheta - 1)

        theta_indices = np.arange(self.ntheta)
        theta_values = self.theta_rad_min + theta_indices * theta_step
        valid_theta_indices = theta_values != 0.0

        phi_indices = np.arange(self.nphi)
        phi_values = phi_indices * phi_step

        cos2phi = np.cos(2.0 * phi_values)
        sin2phi = np.sin(2.0 * phi_values)

        q = self.stokes[1, :, valid_theta_indices]
        u = self.stokes[2, :, valid_theta_indices]

        beam_copy.stokes[1, :, valid_theta_indices] = (
            q * cos2phi[None, :] + u * sin2phi[None, :]
        )
        beam_copy.stokes[2, :, valid_theta_indices] = (
            -q * sin2phi[None, :] + u * cos2phi[None, :]
        )
        beam_copy.polar_basis = True
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
        theta = thetaphi[0, :]
        phi = thetaphi[1, :]

        beam_polar = self.convert_to_polar_basis()

        theta = theta[theta <= self.theta_rad_max]
        phi = phi[: len(theta)]

        beam_map = np.full((nstokes, npix), unseen_pixel_value, dtype=float)

        # Create a grid of theta and phi values
        theta_grid = np.linspace(self.theta_rad_min, self.theta_rad_max, self.ntheta)

        # To make a periodicity in phi, we add the first phi value to the end
        phi_grid = np.linspace(0.0, 2.0 * np.pi, self.nphi + 1)

        for stokes_idx in range(nstokes):
            # To make a periodicity in stokes, we add the first stokes value to the end
            stokes_extended = np.concatenate(
                [beam_polar.stokes[stokes_idx], beam_polar.stokes[stokes_idx][:1, :]],
                axis=0,
            )

            # Create a 2D interpolator for the beam stokes values
            interpolator = RegularGridInterpolator(
                (phi_grid, theta_grid), stokes_extended, method=interp_method
            )

            # Use the interpolator to get the beam values at the given theta and phi
            beam_map[stokes_idx, : len(theta)] = interpolator(np.array([phi, theta]).T)

        return BeamHealpixMap(beam_map)


class BeamGrid:
    """Class to hold the data loaded from a TICRA GRASP beam grid file

    This class only supports polar spherical grids in the far field region.

    Args:
        ktype (`int`): The file format (always 1)
        nset (`int`): Number of field sets or beams (this class only
            supports *one* field)
        klimit (`int`): Specification of limits in a 2D grid.
        icomp (`int`): Control parameter of field components. Only types
            3 (copolar-crosspolar) and 9 (total power and
            :math:`\\sqrt(\\text{RHC}/\\text{LHC}` are supported)
        ncomp (`int`): Number of field components (only ``ncomp==2`` is supported)
        igrid (`int`): Control parameter of field grid type (only ``igrid==7`` is supported)
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
        amp (`numpy.ndarray` | None): 2D array of complex amplitudes [:math:`\theta`, :math:`\phi`].
    """

    def __init__(self, file_obj: typing.TextIO):
        """Initialize the BeamGrid object."""

        self.ktype = 0
        self.nset = 0
        self.klimit = 0
        self.icomp = 0
        self.ncomp = 0
        self.igrid = 0
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
        self._parse_file(file_obj)

    def _parse_file(self, fi: typing.TextIO) -> None:
        """Read and parse the beam grid file."""

        while True:
            line = fi.readline()
            if line[0:4] == "++++":
                break
            elif "FREQUENCIES [" in line:
                self.frequency_unit = line.split("[")[1].split("]")[0]
                self.frequency = float(fi.readline().strip())

        self.ktype = int(fi.readline())
        if not self.ktype == 1:
            raise ValueError("Unknown Grasp grid format, ktype != 1")

        line = fi.readline().split()
        self.nset = int(line[0])
        self.icomp = int(line[1])
        self.ncomp = int(line[2])
        self.igrid = int(line[3])

        if self.nset > 1:
            warnings.warn("Warning: nset > 1, only reading first beam in file")

        line = fi.readline().split()
        self.ix = int(line[0])
        self.iy = int(line[1])
        i = 2
        while i <= self.nset:
            fi.readline()
            i += 1

        line = fi.readline().split()
        self.xs = float(line[0])
        self.ys = float(line[1])
        self.xe = float(line[2])
        self.ye = float(line[3])

        beam_solid_angle_rad = (
            np.cos(np.deg2rad(self.ys)) - np.cos(np.deg2rad(self.ye))
        ) * (np.deg2rad(self.xe) - np.deg2rad(self.xs))
        if not np.isclose(beam_solid_angle_rad, 2.0 * np.pi) and not np.isclose(
            beam_solid_angle_rad, 4.0 * np.pi
        ):
            warnings.warn(
                f"Warning: beam solid angle is not 2pi or 4pi because BeamGrid has xs={self.xs}, xe={self.xe}, ys={self.ys} and ye={self.ye}. The header should be checked."
            )

        line = fi.readline().split()
        self.nx = int(line[0])
        self.ny = int(line[1])
        self.klimit = int(line[2])
        self.amp = np.zeros((self.ncomp, self.nx, self.ny), dtype=complex)

        is_val, in_val = 0, 0
        for j in range(self.ny):
            if self.klimit == 1:
                line = fi.readline().split()
                in_val = int(line[1])
                is_val = int(line[0])
            else:
                is_val = 1
                in_val = self.nx

            for i in range(is_val, in_val + 1):
                line = fi.readline().split()
                if any(np.isnan(list(map(float, line)))):
                    raise ValueError(
                        "Encountered a NaN value in Amplitude. Please check your input."
                    )
                self.amp[0, i - 1, j] = float(line[0]) + float(line[1]) * 1j
                self.amp[1, i - 1, j] = float(line[2]) + float(line[3]) * 1j

    def to_polar(self, copol_axis="x") -> BeamPolar:
        """Converts beam in polar grid format into Stokes
        parameters on a polar grid. The value of copol
        specifies the alignment of the co-polar basis
        ('x' or 'y') of the input GRASP file.

        Args:
            copol_axis (`str`): The copolarization axis. Must be 'x' or 'y'.

        Returns:
            BeamPolar: The beam grid in polar coordinates.

        Raises:
            ValueError: If the beam is not in the supported GRASP grid format.

        """
        copol_axis = copol_axis.lower()
        if not self.ncomp == 2:
            raise ValueError(
                "Error in BeamGrid.to_polar: beam is not in linear 'co' and 'cx' components"
            )
        if not self.igrid == 7:
            raise ValueError(
                "Error in BeamGrid.to_polar: beam is not on theta-phi grid"
            )
        if not abs(self.xs) <= 1e-5:
            raise ValueError(
                "Error in BeamGrid.to_polar: phi coordinates does not start at zero"
            )
        if not abs(self.xe - self.xs - 360.0) <= 1e-5:
            raise ValueError("Error in BeamGrid.to_polar: phi range is not 360 degrees")
        if copol_axis not in ["x", "y"]:
            raise ValueError(
                "Error in BeamGrid.to_polar: copol_axis must be 'x' or 'y'"
            )

        nphi = self.nx - 1
        ntheta = self.ny
        theta_rad_min = np.deg2rad(self.ys)
        theta_rad_max = np.deg2rad(self.ye)
        swap_theta = theta_rad_min > theta_rad_max
        if swap_theta:
            warnings.warn("swapping theta direction")
            theta_rad_min = np.deg2rad(self.ye)
            theta_rad_max = np.deg2rad(self.ys)

        beam_polar = BeamPolar(
            nphi,
            ntheta,
            theta_rad_min,
            theta_rad_max,
        )

        if self.icomp == 3:
            if copol_axis == "x":
                sign = -1
            elif copol_axis == "y":
                sign = 1
            else:
                raise ValueError("Error in bm_grid2polar: unknown value for copol")
            c = self.amp[0, :-1, :]
            x = self.amp[1, :-1, :]
            modc2 = np.abs(c) ** 2
            modx2 = np.abs(x) ** 2
            acaxs = c * np.conj(x)
            beam_polar.stokes[0, :, :] = modc2 + modx2
            beam_polar.stokes[1, :, :] = sign * (modc2 - modx2)
            beam_polar.stokes[2, :, :] = sign * 2.0 * np.real(acaxs)
            beam_polar.stokes[3, :, :] = 2.0 * np.imag(acaxs)
            if swap_theta:
                beam_polar.stokes = beam_polar.stokes[:, :, ::-1]
        elif self.icomp == 9:
            c = self.amp[1, :-1, :]
            modc2 = np.abs(c) ** 2
            beam_polar.stokes[0, :, :] = modc2
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
        self.vini = 0.0
        self.vinc = 0.0
        self.vnum = 0
        self.c = np.array([])
        self.icomp = 0
        self.icut = 0
        self.ncomp = 0
        self.ncut = 0
        self.amp = None
        self._parse_file(file_obj)

    def _parse_file(self, file_obj: typing.TextIO) -> None:
        "Parse the .cut file"

        # Read the header line and throw it away
        _ = file_obj.readline().strip()

        while True:
            line = file_obj.readline()
            if not line:
                break
            data = line.split()
            if len(data) == 7:
                (
                    self.vini,
                    self.vinc,
                    self.vnum,
                    c,
                    self.icomp,
                    self.icut,
                    self.ncomp,
                ) = map(float, data)
                self.vnum, self.icomp, self.icut, self.ncomp = map(
                    int, (self.vnum, self.icomp, self.icut, self.ncomp)
                )
                self.c = np.append(self.c, c)
                self.ncut += 1
            if self.ncomp > 2:
                raise ValueError(
                    "Three field components present. Beam package can only handle two field components."
                )
            if self.vnum % 2 == 0:
                raise ValueError("The number of pixels in a cut must be odd.")

        self.amp = np.zeros((self.ncomp, self.vnum, self.ncut), dtype=complex)
        file_obj.seek(0)
        cnt = 0
        while True:
            line = file_obj.readline()
            if not line:
                break
            data = line.split()
            if len(data) == 7:
                (
                    self.vini,
                    self.vinc,
                    self.vnum,
                    _,
                    self.icomp,
                    self.icut,
                    self.ncomp,
                ) = map(float, data)
                self.vnum, self.icomp, self.icut, self.ncomp = map(
                    int, (self.vnum, self.icomp, self.icut, self.ncomp)
                )
                for i in range(self.vnum):
                    line = file_obj.readline()
                    data = line.split()
                    tmp1, tmp2, tmp3, tmp4 = map(float, data)
                    if any(np.isnan([tmp1, tmp2, tmp3, tmp4])):
                        raise ValueError(
                            "Encountered a NaN value in Amplitude. Please check your input."
                        )
                    self.amp[0, i, cnt] = complex(tmp1, tmp2)
                    self.amp[1, i, cnt] = complex(tmp3, tmp4)
                cnt += 1

    def to_polar(self, copol_axis: str = "x") -> BeamPolar:
        """Converts beam in "cut" format to Stokes parameters
        on a polar grid.  Assumes that cuts are evenly spaced
        in theta. The value of copol specifies the alignment
        of the co-polar basis ('x' or 'y') of the input GRASP file.

        Args:
            copol_axis (`str`): The axis of copolarization. Must be either 'x' or 'y'.

        Returns:
            BeamPolar: The beam in polar coordinates.

        Raises:
            ValueError: If the beam is not in the expected format.
        """
        copol_axis = copol_axis.lower()

        if self.icomp != 3:
            raise ValueError(
                "Error in BeamCut.to_polar: beam is not in linear 'co' and 'cx' components"
            )
        if self.icut != 1:
            raise ValueError("Error in BeamCut.to_polar: beam is not in phi cuts")
        if self.ncomp != 2:
            raise ValueError(
                "Error in BeamCut.to_polar: beam has the wrong number of components"
            )
        if copol_axis not in ["x", "y"]:
            raise ValueError("Error in BeamCut.to_polar: copol_axis must be 'x' or 'y'")

        nphi = int(2 * self.ncut)
        ntheta = int(self.vnum // 2) + 1
        theta_rad_min = 0.0
        theta_rad_max = np.deg2rad(np.abs(self.vini))
        beam_polar = BeamPolar(
            nphi,
            ntheta,
            theta_rad_min,
            theta_rad_max,
        )
        amp_tmp = np.zeros((2, nphi, ntheta), dtype=complex)

        for i in range(self.ncut):
            amp_tmp[:, i, :] = self.amp[:, ntheta - 1 : self.vnum + 1, i]
            amp_tmp[:, self.ncut + i, :] = self.amp[:, ntheta - 1 :: -1, i]

        if copol_axis == "x":
            sign = -1
        elif copol_axis == "y":
            sign = 1
        else:
            raise ValueError(f"{copol_axis=} can only be 'x' or 'y'")

        c = amp_tmp[0, :, :]
        x = amp_tmp[1, :, :]

        modc2 = np.abs(c) ** 2
        modx2 = np.abs(x) ** 2
        acaxs = c * np.conj(x)

        beam_polar.stokes[0, :, :] = modc2 + modx2
        beam_polar.stokes[1, :, :] = sign * (modc2 - modx2)
        beam_polar.stokes[2, :, :] = sign * 2.0 * np.real(acaxs)
        beam_polar.stokes[3, :, :] = 2.0 * np.imag(acaxs)
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
