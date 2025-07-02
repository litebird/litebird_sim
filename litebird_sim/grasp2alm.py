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
import warnings
from dataclasses import dataclass

import ducc0.healpix
import ducc0.sht
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .healpix import UNSEEN_PIXEL_VALUE, npix_to_nside, nside_to_npix, num_of_alms

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
        lmax: int | None = None,
        mmax: int | None = None,
        epsilon=1e-8,
        max_num_of_iterations=20,
    ) -> np.ndarray:
        """Converts the beam map to spherical harmonic coefficients.

        Args:
            lmax (`int`): Maximum l value for the spherical harmonic expansion. If it is not provided,
                the default ``5NSIDE - 1`` will be used.
            mmax (`int`): Maximum m value for the spherical harmonic expansion. If not provided,
                it will be assumed that ``lmax == mmax``.
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

        if not lmax:
            lmax = 5 * self.nside - 1

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
    polar grid parameterized by (:math:`\theta`,:math:`\phi`). This type is used as input to the
    spherical harmonic transform. Internally, this package uses the third Ludwig's definition for the
    polarization basis with the co-polar direction (positive :math:`Q`) aligned with the y-axis.

    This class assumes that the :math:`\phi` angles spans 2π. This is the reason why there are
    no ``phi_rad_min``/``phi_rad_max`` parameters.

    Args:
        nphi (`int`): Number of :math:`\phi` values.
        ntheta (`int`): Number of :math:`\theta` values.
        theta_rad_min (`float`): Minimum :math:`\theta` value in radians.
        theta_rad_max (`float`): Maximum :math:`\theta` value in radians.
        stokes (`numpy.ndarray`): Array of shape (4, ``nphi``, ``ntheta``) containing the four Stokes
            parameters (:math:`I`,:math:`Q`,:math:`U`,:math:`V`).
        filename (`str`): Name of the file.
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
        filename: str,
        polar_basis: bool = False,
    ):
        self.nphi = nphi
        self.ntheta = ntheta
        self.theta_rad_min = theta_rad_min
        self.theta_rad_max = theta_rad_max
        self.stokes = np.zeros((4, nphi, ntheta), dtype=float)
        self.filename = filename
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
        outOftheta_val: float = UNSEEN_PIXEL_VALUE,
        interp_method: str = "linear",
    ) -> BeamHealpixMap:
        """Convert the :class:`.BeamPolar` to a :class:`.BeamMap`.

        Args:
            nside (`int`): The nside parameter for the HEALPix map.
            nstokes (`int`): Number of Stokes parameters.
            outOftheta_val (`float`): Value to fill outside the valid theta range.
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

        beam_map = np.full((nstokes, npix), outOftheta_val, dtype=float)
        for s in range(nstokes):
            beam_map[s, : len(theta)] = beam_polar._get_interp_val(
                theta, phi, s, interp_method
            )
        return BeamHealpixMap(beam_map)

    def _get_interp_val(
        self, theta: np.ndarray, phi: np.ndarray, s: int, interp_method="linear"
    ):
        """Calculate the value of the beam at a given :math:`\theta`, :math:`\phi`, and Stokes parameter.
        The value is interpolated from `BeamPolar` by a given :math:`\theta` and :math:`\phi`.

        Args:
            theta (`float` or `array-like`): The :math:`\theta` value(s) at which to evaluate the beam.
            phi (`float` or `array-like`): The :math:`\phi` value(s) at which to evaluate the beam.
            s (`int`): The Stokes parameter index.
            interp_method (`str`): Interpolation method to use. Default is 'linear'.
                Supported are 'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'.
        Returns:
            value (`float` or `array-like`): The value(s) of the beam at the given :math:`\theta`, :math:`\phi`, and Stokes parameter.

        """
        # Create a grid of theta and phi values
        theta_grid = np.linspace(self.theta_rad_min, self.theta_rad_max, self.ntheta)
        # To make a periodicity in phi, we add the first phi value to the end
        phi_grid = np.linspace(0.0, 2.0 * np.pi, self.nphi + 1)
        # To make a periodicity in stokes, we add the first stokes value to the end
        stokes_extended = np.concatenate(
            [self.stokes[s], self.stokes[s][:1, :]], axis=0
        )

        # Create a 2D interpolator for the beam stokes values
        interpolator = RegularGridInterpolator(
            (phi_grid, theta_grid), stokes_extended, method=interp_method
        )

        # Use the interpolator to get the beam values at the given theta and phi
        value = interpolator(np.array([phi, theta]).T)

        return value


def _get_interp_val_from_polar_original(
    beam: BeamPolar, theta: np.ndarray, phi: np.ndarray, s: int
):
    """Calculate the value of the beam at a given :math:`\theta`, :math:`\phi`, and Stokes parameter.
    The value is interpolated from :class:`.BeamPolar` by a given :math:`\theta` and :math:`\phi`.

    Args:
        beam (:class:`.BeamPolar`): The polar beam object.
        theta (`float` or `array-like`): The :math:`\theta` value(s) at which to evaluate the beam.
        phi (`float` or `array-like`): The :math:`\phi` value(s) at which to evaluate the beam.
        s (`int`): The Stokes parameter index.

    Returns:
        value (`float` or `array-like`): The value(s) of the beam at the given :math:`\theta`, :math:`\phi` and Stokes parameter.

    """
    # TODO replace this function by scipy interpolation
    theta_step = (beam.theta_rad_max - beam.theta_rad_min) / (beam.ntheta - 1)
    phi_step = 2.0 * np.pi / beam.nphi
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    ith1 = (theta / theta_step).astype(int)
    ith1 = np.maximum(0, np.minimum(beam.ntheta - 2, ith1))
    ith2 = ith1 + 1

    iph1 = (phi / phi_step).astype(int)
    iph1 = np.maximum(0, np.minimum(beam.nphi - 1, iph1))
    iph2 = iph1 + 1
    iph2[iph2 >= beam.nphi] = 0

    th1 = beam.theta_rad_min + ith1 * theta_step
    wth = 1.0 - (theta - th1) / theta_step

    ph1 = iph1 * phi_step
    wph = 1.0 - (phi - ph1) / phi_step

    value = wth * (
        wph * beam.stokes[s, iph1, ith1] + (1.0 - wph) * beam.stokes[s, iph2, ith1]
    ) + (1.0 - wth) * (
        wph * beam.stokes[s, iph1, ith2] + (1.0 - wph) * beam.stokes[s, iph2, ith2]
    )
    return value


@dataclass
class BeamGrid:
    """Class to hold the data loaded from a TICRA GRASP beam grid file

    Args:
        header (`str`): Record with identification text.
        ktype (`int`): Specifies type of file format.
        nset (`int`): Number of field sets or beams.
        klimit (`int`): Specification of limits in a 2D grid.
        icomp (`int`): Control parameter of field components.
        ncomp (`int`): Number of field components.
        igrid (`int`): Control parameter of field grid type.
        ix (`int`): Centre of set or beam No. :math:`i`.
        iy (`int`): Centre of set or beam No. :math:`i`.
        xs (`float`): Start x-coordinate of the grid.
        ys (`float`): Start y-coordinate of the grid.
        xe (`float`): End x-coordinate of the grid.
        ye (`float`): End y-coordinate of the grid.
        nx (`int`): Number of columns.
        ny (`int`): Number of rows.
        freq (`float`): Frequency.
        frequnit (`str`): Frequency unit.
        amp (`numpy.ndarray`): Array of complex amplitudes [:math:`\theta`, :math:`\phi`].
    """

    header: str = ""
    ktype: int = 0
    nset: int = 0
    klimit: int = 0
    icomp: int = 0
    ncomp: int = 0
    igrid: int = 0
    ix: int = 0
    iy: int = 0
    xs: float = 0.0
    ys: float = 0.0
    xe: float = 0.0
    ye: float = 0.0
    nx: int = 0
    ny: int = 0
    frequency: float = 0.0
    frequency_unit: str = ""
    amp: np.ndarray = None

    def __init__(self, filepath):
        """Initialize the BeamGrid object."""

        super().__init__()
        self.file_path = filepath
        self.file_name = filepath.split("/")[-1]
        self.__post_init__()

    def __post_init__(self):
        """Read and parse the beam grid file."""

        if not self.file_path.endswith(".grd"):
            raise ValueError(
                "Error in BeamGrid.__post_init__: The file is not a GRASP grid file."
            )
        with open(self.file_path, "r") as fi:
            while True:
                line = fi.readline()
                if line[0:4] == "++++":
                    break
                elif "FREQUENCIES [" in line:
                    self.frequency_unit = line.split("[")[1].split("]")[0]
                    self.frequency = float(fi.readline().strip())
                else:
                    self.header += line[:-1] + "\n"

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
            nphi, ntheta, theta_rad_min, theta_rad_max, self.file_name
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


@dataclass
class BeamCut:
    """Class to hold the data from a beam cut file of GRASP.

    Args:
        header (`str`): Record with identification text.
        vini (`float`): Initial value.
        vinc (`float`): Increment.
        vnum (`int`): Number of values in cut.
        c (`numpy.ndarray`): Constant.
        icomp (`int`): Polarization control parameter.
        icut (`int`): Control parameter of cut.
        ncomp (`int`): Number of field components.
        ncut (`int`): Number of cuts.
        amp (`numpy.ndarray`): Amplitude.
    """

    header: str = ""
    vini: float = 0.0
    vinc: float = 0.0
    vnum: int = 0
    c = np.array([])

    icomp: int = 0
    icut: int = 0
    ncomp: int = 0
    ncut: int = 0
    amp: np.ndarray = None

    def __init__(self, filepath):
        """
        Initializes a BeamCut object.

        Args:
            filepath (`str`): The path to the GRASP cut file.
        """
        super().__init__()
        self.filepath = filepath
        self.filename = filepath.split("/")[-1]
        self.__post_init__()

    def __post_init__(self):
        """Performs post-initialization tasks."""

        if not self.filepath.endswith(".cut"):
            raise ValueError(
                "Error in BeamCut.__post_init__: The file is not a GRASP cut file."
            )
        with open(self.filepath, "r") as fi:
            self.header = fi.readline().strip()
            while True:
                line = fi.readline()
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
            fi.seek(0)
            cnt = 0
            while True:
                line = fi.readline()
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
                        line = fi.readline()
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
            nphi, ntheta, theta_rad_min, theta_rad_max, self.filename
        )
        amp_tmp = np.zeros((2, nphi, ntheta), dtype=complex)

        for i in range(self.ncut):
            amp_tmp[:, i, :] = self.amp[:, ntheta - 1 : self.vnum + 1, i]
            amp_tmp[:, self.ncut + i, :] = self.amp[:, ntheta - 1 :: -1, i]

        if copol_axis == "x":
            sign = -1
        elif copol_axis == "y":
            sign = 1

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


def grasp2alm(
    file_path: str,
    nside: int,
    outOftheta_val: float = UNSEEN_PIXEL_VALUE,
    interp_method: str = "linear",
    copol_axis: str = "x",
    lmax: int | None = None,
    mmax: int | None = None,
    epsilon: float = 1e-8,
    max_num_of_iterations: int = 20,
) -> np.ndarray:
    """Convert a GRASP file to a spherical harmonic coefficients of beam map.

    Args:
        filepath (`str`): Path to the GRASP file.
        nside (`int`): Resolution parameter for the output beam map.
        lmax (`int`): The desired lmax parameters for the analysis.
        mmax (`int`): The desired mmax parameters for the analysis.
        outOftheta_val (`float`): Value to assign to pixels outside
            the valid theta range.
        interp_method (`str`): Interpolation method to use. Default is 'linear'.
                Supported are 'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'.
        copol_axis (`str`, `optional`): Axis of the co-polarization
            component. Defaults to 'x'.
        iter : `int`, `scalar`, `optional`
            Number of iteration (default: 3)
        pol : `bool`, `optional`
            If `True`, assumes input maps are TQU. Output will be TEB alm's.
            (input must be 1 or 3 maps)
            If False, apply spin 0 harmonic transform to each map.
            (input can be any number of maps)
            If there is only one input map, it has no effect. Default: `True`.
        use_weights : `bool`, `scalar`, `optional`
            If `True`, use the ring weighting. Default: False.
        datapath : `None` or `str`, `optional`
            If given, the directory where to find the pixel weights.
            See in the docstring above details on how to set it up.
        gal_cut : `float` [degrees]
            pixels at latitude in [-gal_cut;+gal_cut] are not taken into account
        use_pixel_weights: `bool`, `optional`
            If `True`, use pixel by pixel weighting, healpy will automatically download the weights, if needed

    Returns:
        alm (`numpy.ndarray`): Spherical harmonic coefficients of the beam map.

    Raises:
        ValueError: If the file format is unknown.

    """
    if file_path.endswith(".grd"):
        beam = BeamGrid(file_path)
    elif file_path.endswith(".cut"):
        beam = BeamCut(file_path)
    else:
        raise ValueError("Error in grasp2alm: unknown file format")
    beam_polar = beam.to_polar(copol_axis)
    beam_map = beam_polar.to_map(
        nside, outOftheta_val=outOftheta_val, interp_method=interp_method
    )
    alm = beam_map.to_alm(
        lmax=lmax,
        mmax=mmax,
        epsilon=epsilon,
        max_num_of_iterations=max_num_of_iterations,
    )
    return alm
