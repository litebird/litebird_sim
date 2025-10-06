"""
This file contains a port of the Grasp2alm repository <https://github.com/yusuke-takase/grasp2alm>,
created by Yusuke Takase. We decided to incorporate it into LBS so that we can make the API more
compatible with the framework, while preserving the original source code repository.

If you are patching a bug in this file, be sure to check if the same bug was present in
Yusuke’s repository; if it is, please upload your patches there, so that both codes can benefit
from your work.
"""

import copy
from enum import IntEnum
import typing
import warnings
from dataclasses import dataclass

import ducc0.healpix
import ducc0.sht
from numba import njit
import numpy as np
import numpy.typing as npt

from .beam_convolution import SphericalHarmonics
from .healpix import npix_to_nside, nside_to_npix, nside_to_resolution_rad, num_of_alms

REASON_DESCRIPTION = {
    1: "Approximate solution found",
    2: "Approximate least-square solution found",
    3: "Too large condition number",
    7: "Maximum number of iterations reached",
}


class SphericalFarFieldDecomposition(IntEnum):
    ICOMP_CO_AND_CX = (3,)
    ICOMP_POWER_AND_RATIO = 9


class SphericalFarFieldGrid(IntEnum):
    IGRID_ELEVATION_AND_AZIMUTH = (5,)
    IGRID_THETA_PHI = 7


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
def _to_polar_basis(theta_phi_values_rad: npt.NDArray, stokes: npt.NDArray) -> None:
    num_of_samples = theta_phi_values_rad.shape[0]

    for i in range(num_of_samples):
        cur_theta = theta_phi_values_rad[i, 0]
        if cur_theta == 0:
            # No transformation is needed at the North Pole
            continue

        cur_phi = theta_phi_values_rad[i, 1]

        cos2phi = np.cos(2.0 * cur_phi)
        sin2phi = np.sin(2.0 * cur_phi)

        q = stokes[1, i]
        u = stokes[2, i]
        rotated_q = q * cos2phi + u * sin2phi
        rotated_u = -q * sin2phi + u * cos2phi
        stokes[1, i] = rotated_q
        stokes[2, i] = rotated_u


@njit
def beam_mapmaker(
    pixidx: npt.NDArray,
    values: npt.NDArray,
    output_map: npt.NDArray,
    hit_map: npt.NDArray,
) -> None:
    "A simple binning map-maker for Stokes parameters"

    assert len(pixidx) == len(values), (
        "The number of samples and of pixel indices must match"
    )
    assert len(output_map) == len(hit_map), (
        "The NSIDE values of the output map and the hit map must match"
    )

    # Set all the pixels in the two maps to zero
    for i in range(len(hit_map)):
        output_map[i] = 0
        hit_map[i] = 0

    # Do the binning
    for i in range(len(pixidx)):
        cur_pix = pixidx[i]
        output_map[cur_pix] += values[i]
        hit_map[cur_pix] += 1

    # Normalize the output map. Note that we save NaN for unseen pixels;
    # this will be fixed in the main code.
    for i in range(len(hit_map)):
        if hit_map[i] > 0:
            output_map[i] /= hit_map[i]
        else:
            output_map[i] = np.NaN


@dataclass
class BeamStokesPolar:
    """Representation of a beam as a set of Stokes parameters sampled over a sphere using θφ coordinates

    This class stores the four Stokes parameters (:math:`I`, :math:`Q`, :math:`U`, and :math:`V`) for a beam on a
    possibly irregular spherical grid (:math:`\\theta`, :math:`\\varphi`). This type is used as input to the
    spherical harmonic transform. Internally, this package uses the third Ludwig's definition for the
    polarization basis with the co-polar direction (positive :math:`Q`) aligned with the y-axis.

    Args:
        theta_phi_values_rad (`npt.ArrayLike`): A 2D matrix of shape :math:`(N, 2)` containing the
            values for :math:`\\theta` (in ``theta_phi_values_rad[:, 0]``) and for :math:`\\varphi`
            (in ``theta_phi_values_rad[:, 1]``.
        stokes (`numpy.ndarray`): Array of shape :math:`(4, N)` containing the four Stokes
            parameters (:math:`I`, :math:`Q`, :math:`U`, :math:`V`, in this order).
        polar_basis_flag (`bool`): If ``True``, the :math:`Q` and :math:`U` parameters have been rotated
            so that they are expressed in a polar basis. This is handy if you are going to
            compute the harmonic coefficients on them. By default, this is ``False``, because typically
            :class:`BeamPolar` instances are created out of TICRA GRASP files, which use Ludwig's third
            definition of the polarization.
    """

    def __init__(
        self,
        theta_phi_values_rad: npt.ArrayLike,
        polar_basis_flag: bool = False,
    ):
        assert theta_phi_values_rad.shape[1] == 2, (
            "Error, theta_phi_values_rad must be a matrix with shape (N, 2)"
        )
        self.theta_phi_values_rad = theta_phi_values_rad
        self.num_of_samples = theta_phi_values_rad.shape[0]
        self.stokes = np.zeros((4, self.num_of_samples), dtype=np.float64)
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

        if not beam_copy.polar_basis_flag:
            _to_polar_basis(self.theta_phi_values_rad, beam_copy.stokes)
            beam_copy.polar_basis_flag = True

        return beam_copy

    def minimum_angle_between_samples_rad(self) -> float:
        """
        Return the minimum angle between samples in the GRASP file

        This information is useful if you plan to project the samples over a
        Healpix map, as it provides a starting point for figuring out NSIDE.

        See :meth:`.find_best_nside_for_resolution()`.

        Returns:
            The minimum separation between samples, in radians
        """
        from scipy.spatial import KDTree

        theta = self.theta_phi_values_rad[:, 0]
        phi = self.theta_phi_values_rad[:, 1]

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.column_stack((x, y, z))  # shape (N, 3)

        tree = KDTree(points)
        dists, _ = tree.query(points, k=2)  # k=2 to exclude distance to self
        min_dist = np.min(dists[:, 1])  # First column is zero (self-distance)

        # Convert Euclidean to angular distance
        cos_gamma = 1 - 0.5 * min_dist**2
        min_angle = np.arccos(np.clip(cos_gamma, -1.0, 1.0))

        return min_angle

    def find_best_nside_for_resolution(self, resol_rad: float | None = None) -> int:
        """Given a typical separation between two samples, estimate a good value for NSIDE

        This function can be used to estimate which value of NSIDE to use for building
        a Healpix map with the representation of the beam, given the minimal separation
        between two samples in the GRASP file.

        Note that this function should *not* be called for GRASP grids sampled using the
        (θ, φ) grid scheme, as in this case the spacing between points varies a lot between
        different co-latitude rings.

        If the parameter `resol_rad` is not provided, the code will call the method
        :meth:`.minimum_angle_between_samples_rad`.
        """
        actual_resol_rad = (
            resol_rad if resol_rad else self.minimum_angle_between_samples_rad()
        )

        nside = 1
        old_nside = 1

        while nside_to_resolution_rad(nside) > actual_resol_rad:
            old_nside = nside
            nside *= 2

        # We downgrade the resolution by two steps here (2²) because the value
        # returned by `minimum_angle_between_samples_rad()` is a *minimum* angle,
        # and it might be aligned unfavourably with respect to the Healpix grid.
        return old_nside // 4 if old_nside > 4 else 1

    def to_map(
        self,
        nside: int,
        nstokes: int = 3,
        unseen_pixel_value: float = 0.0,
    ) -> BeamHealpixMap:
        """Convert the :class:`.BeamPolar` to a :class:`.BeamMap`.

        Args:
            nside (`int`): The nside parameter for the HEALPix map. If you are sampling the
                sphere using a roughly uniform coverage (which means you are *not* using
                the IGRID=7 theta/phi sampling!), you can estimate this value using
                :meth:`.minimum_angle_between_samples_rad`.
            nstokes (`int`): Number of Stokes parameters to project. The default is 3, which means
                that three maps are produced: I, Q, and U.
            unseen_pixel_value (`float`): Value to fill outside the valid theta range.

        Returns:
            :class:`.BeamMap`: A new instance of ``BeamMap`` representing the beam map.
        """

        assert nstokes <= 4, (
            f"Wrong value for {nstokes=}, it can only be 1 (I), 2 (I, Q), 3 (I, Q, U), or 4 (I, Q, U, V)"
        )

        base = ducc0.healpix.Healpix_Base(nside, "RING")
        npix = nside_to_npix(nside)

        # Convert Q and U from Ludwig’s third definition into θ/φ coordinates
        if not self.polar_basis_flag:
            beam_polar = self.convert_to_polar_basis()
        else:
            beam_polar = self

        # Build the Stokes maps
        pixel_indexes = base.ang2pix(self.theta_phi_values_rad)
        beam_map = np.empty((nstokes, npix), dtype=float)
        hit_map = np.empty(npix, dtype=int)
        for stokes_idx in range(nstokes):
            cur_stokes = beam_polar.stokes[stokes_idx, :]
            if np.any(np.isnan(cur_stokes)):
                raise ValueError(
                    f"NaN values in Stokes parameters at {np.arange(cur_stokes.size)[np.isnan(cur_stokes)]=}"
                )
            cur_map = beam_map[stokes_idx, :]
            beam_mapmaker(
                pixidx=pixel_indexes,
                values=self.stokes[stokes_idx, :],
                output_map=cur_map,
                hit_map=hit_map,
            )

            # TODO: it would be nice to fill solitary NaN pixels with the average of their neighbours,
            #       before filling everything with `unseen_pixel_value`…
            cur_map[np.isnan(cur_map)] = unseen_pixel_value

        return BeamHealpixMap(beam_map)


@njit
def _build_thetaphi_values_for_spherical_coordinates(
    theta_values_rad: npt.NDArray, phi_values_rad: npt.NDArray, result: npt.NDArray
) -> None:
    n_theta = len(theta_values_rad)
    n_phi = len(phi_values_rad)

    assert result.shape == (n_theta * n_phi, 2)

    i = 0
    for phi_idx in range(n_phi):
        for theta_idx in range(n_theta):
            result[i, 0] = theta_values_rad[theta_idx]
            result[i, 1] = phi_values_rad[phi_idx]
            i += 1


@njit
def _build_thetaphi_values_for_az_el_coordinates(
    azimuth_values_rad: npt.NDArray,
    elevation_values_rad: npt.NDArray,
    result: npt.NDArray,
) -> None:
    n_azimuth = len(azimuth_values_rad)
    n_elevation = len(elevation_values_rad)

    assert result.shape == (n_azimuth * n_elevation, 2)

    i = 0
    for el_idx in range(n_elevation):
        for az_idx in range(n_azimuth):
            cur_az = azimuth_values_rad[az_idx]
            cur_el = elevation_values_rad[el_idx]

            if cur_el != 0 or cur_az != 0:
                result[i, 0] = np.sqrt(cur_az * cur_az + cur_el * cur_el)
                result[i, 1] = np.arctan2(cur_el, -cur_az)

                # This ensures that all φ angles are in [0, 2π)
                # (it is required by some Ducc functions)
                if result[i, 1] < 0.0:
                    result[i, 1] += 2 * np.pi
            else:
                result[i, 0] = 0
                result[i, 1] = 0

            i += 1


class BeamGrid:
    """Class to hold the data loaded from a TICRA GRASP beam grid file

    This class only supports polar spherical grids in the far field region.

    Args:
        nset (`int`): Number of field sets or beams (this class only
            supports *one* field)
        klimit (`int`): Specification of limits in a 2D grid.
        field_component_type (`SphericalFarFieldDecomposition`): Control parameter of field components. Only types
            3 (copolar-crosspolar) and 9 (total power and
            :math:`\\sqrt(\\text{RHC}/\\text{LHC}` are supported)
        num_of_components (`int`): Number of field components (only ``ncomp==2`` is supported)
        grid_type (`SphericalFarFieldGrid`): Control parameter of field grid type (only ``igrid==7`` is supported)
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

        icomp = int(line[1])
        try:
            self.field_component_type = SphericalFarFieldDecomposition(icomp)
        except ValueError:
            raise ValueError(f"Unknown value ICOMP={icomp}")

        self.num_of_components = int(line[2])

        igrid = int(line[3])
        try:
            self.grid_type = SphericalFarFieldGrid(igrid)
        except ValueError:
            raise ValueError(f"Unknown value IGRID={igrid}")

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
        """Converts beam in polar grid format into Stokes parameters on a polar grid.

        The value of `copol` specifies the alignment of the co-polar basis ('x' or 'y')
        of the input GRASP file.

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

        theta_phi_values_rad = np.empty((self.nx * self.ny, 2), dtype=np.float64)
        if self.grid_type == SphericalFarFieldGrid.IGRID_THETA_PHI:
            theta_rad_min = np.deg2rad(self.ys)
            theta_rad_max = np.deg2rad(self.ye)

            _build_thetaphi_values_for_spherical_coordinates(
                theta_values_rad=np.linspace(theta_rad_min, theta_rad_max, self.ny),
                phi_values_rad=np.linspace(
                    np.deg2rad(self.xs), np.deg2rad(self.xe), self.nx
                ),
                result=theta_phi_values_rad,
            )
        elif self.grid_type == SphericalFarFieldGrid.IGRID_ELEVATION_AND_AZIMUTH:
            azimuth_values = np.linspace(
                np.deg2rad(self.xs), np.deg2rad(self.xe), self.nx
            )
            elevation_values = np.linspace(
                np.deg2rad(self.ys), np.deg2rad(self.ye), self.ny
            )

            _build_thetaphi_values_for_az_el_coordinates(
                azimuth_values_rad=azimuth_values,
                elevation_values_rad=elevation_values,
                result=theta_phi_values_rad,
            )
        else:
            raise ValueError(
                f"Error in BeamGrid.to_polar: beam is not on theta-phi grid ({self.grid_type=} ≠ 5, 7)"
            )

        beam_polar = BeamStokesPolar(
            theta_phi_values_rad=theta_phi_values_rad,
        )

        if self.field_component_type == SphericalFarFieldDecomposition.ICOMP_CO_AND_CX:
            if copol_axis == "x":
                sign = -1
            elif copol_axis == "y":
                sign = 1
            else:
                raise ValueError("Error in bm_grid2polar: unknown value for copol")

            co = self.amp[0, :, :].transpose().flatten()
            cx = self.amp[1, :, :].transpose().flatten()
            mod_co_squared = co.real**2 + co.imag**2
            mod_cx_squared = cx.real**2 + cx.imag**2
            acaxs = co * np.conj(cx)

            beam_polar.stokes[0, :] = mod_co_squared + mod_cx_squared
            beam_polar.stokes[1, :] = sign * (mod_co_squared - mod_cx_squared)
            beam_polar.stokes[2, :] = sign * 2.0 * np.real(acaxs)
            beam_polar.stokes[3, :] = 2.0 * np.imag(acaxs)

        elif (
            self.field_component_type
            == SphericalFarFieldDecomposition.ICOMP_POWER_AND_RATIO
        ):
            co = self.amp[1, :-1, :].flatten()
            mod_co_squared = np.abs(co) ** 2
            beam_polar.stokes[0, :] = mod_co_squared
            beam_polar.stokes[1, :] = 0.0
            beam_polar.stokes[2, :] = 0.0
            beam_polar.stokes[3, :] = 0.0
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

        # As self.amp makes θ vary faster than φ by putting it in the last (rightmost) rank,
        # transposing and flattening it will make sure that the θ angle keeps varying faster
        # than φ, which is what the GRASP .cut format assumes.
        co = self.amp[0, :, :].transpose().flatten()
        cx = self.amp[1, :, :].transpose().flatten()

        mod_co_squared = np.abs(co) ** 2
        mod_cx_squared = np.abs(cx) ** 2
        cross = co * np.conj(cx)

        theta_phi_values_rad = np.empty(
            (len(self.theta_values_rad) * len(self.phi_values_rad), 2), dtype=np.float64
        )
        _build_thetaphi_values_for_spherical_coordinates(
            theta_values_rad=self.theta_values_rad,
            phi_values_rad=self.phi_values_rad,
            result=theta_phi_values_rad,
        )
        beam_polar = BeamStokesPolar(
            theta_phi_values_rad=theta_phi_values_rad,
        )
        beam_polar.stokes[0, :] = mod_co_squared + mod_cx_squared
        beam_polar.stokes[1, :] = sign * (mod_co_squared - mod_cx_squared)
        beam_polar.stokes[2, :] = sign * 2.0 * np.real(cross)
        beam_polar.stokes[3, :] = 2.0 * np.imag(cross)
        return beam_polar


def _grasp2alm(
    file_obj: typing.TextIO,
    beam_class: type[BeamCut] | type[BeamGrid],
    nside: int,
    unseen_pixel_value: float = 0.0,
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
        nside,
        unseen_pixel_value=unseen_pixel_value,
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
