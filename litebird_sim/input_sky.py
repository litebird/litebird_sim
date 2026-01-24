import logging as log
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import ducc0.healpix as dh
import numpy as np
import pysm3
import pysm3.units as u

import litebird_sim as lbs
from litebird_sim import constants as c

from .bandpasses import BandPassInfo
from .coordinates import CoordinateSystem
from .detectors import DetectorInfo, FreqChannelInfo
from .maps_and_harmonics import (
    HealpixMap,
    estimate_alm,
    lin_comb_cls,
    pixelize_alm,
    read_cls_from_fits,
    synthesize_alm,
)
from .units import Units, UnitUtils  # <--- NEW IMPORT

# --- Utility Functions ---


def _get_cmb_unit_conversion(
    target_unit: Units,
    origin_unit: Units = Units.K_CMB,
    freq_ghz: float | None = None,
    bandpass: BandPassInfo | None = None,
    band_integration: bool = False,
) -> float:
    """
    Computes the scalar factor to convert from origin_unit into the target_unit,
    considering potential bandpass integration over a CMB spectrum.

    Parameters
    ----------
    target_unit : Units
        The target unit Enum (e.g., Units.K_CMB, Units.MJy_over_sr).
    origin_unit : Units, optional
        The unit of the input data, by default Units.K_CMB.
    freq_ghz : float | None, optional
        Frequency in GHz for monochromatic conversion. Required if bandpass is None.
    bandpass : BandPassInfo | None, optional
        Bandpass object containing frequencies and weights. Required if band_integration is True.
    band_integration : bool, optional
        If True, integrates over the bandpass using PySM3. If False, uses the band center
        (or freq_ghz) for a monochromatic conversion using Astropy equivalencies.Structure of the Output
-----------------------

The :meth:`~litebird_sim.input_sky.SkyGenerator.execute` method returns a dictionary containing the generated sky objects. Depending on the ``output_type`` parameter, these will be instances of either ``HealpixMap`` or ``SphericalHarmonics``.

If ``return_components=True`` is set in the parameters, the output will be a nested dictionary separating the components:

.. code-block:: python

    {
        "cmb": { "channel_name": map_obj, ... },
        "foregrounds": { "channel_name": map_obj, ... },
        "dipole": { "channel_name": map_obj, ... }
    }

Otherwise, it returns the sum of all requested components.

API Reference
-------------

.. automodule:: litebird_sim.input_sky
    :members:
    :undoc-members:
    :show-inheritance:
        Default is False.

    Returns
    -------
    float
        The conversion factor. Multiply the input data by this factor to get
        values in the target unit.

    Raises
    ------
    ValueError
        If neither freq_ghz nor bandpass is provided.
        If band_integration is True but bandpass is None.
    """
    # Use UnitUtils to get the actual Astropy objects needed by PySM logic
    target_astropy = UnitUtils.get_astropy_unit(target_unit)
    origin_astropy = UnitUtils.get_astropy_unit(origin_unit)

    # --- 1. Validation ---

    # Check that we have at least one source of frequency information
    if bandpass is None and freq_ghz is None:
        raise ValueError("Either 'freq_ghz' or 'bandpass' must be provided.")

    # Check for ambiguity (Warning only)
    if bandpass is not None and freq_ghz is not None:
        log.warning(
            "Both 'bandpass' and 'freq_ghz' provided to unit conversion. Using 'bandpass' information."
        )

    # Check that integration is actually possible
    if band_integration and bandpass is None:
        raise ValueError(
            "band_integration=True requires a valid 'bandpass' object, but None was provided."
        )

    # --- 2. Calculation ---

    if band_integration:
        # Full integration over the bandpass
        nonzero = np.where(bandpass.freqs_ghz != 0)[0]
        freqs = bandpass.freqs_ghz[nonzero] * u.GHz
        weights = bandpass.weights[nonzero]

        # pysm3.bandpass_unit_conversion calculates the factor C such that:
        # Value_Unit = Value_K_CMB * C
        # Therefore: C_target converts K_CMB -> Target
        factor_to_target = pysm3.bandpass_unit_conversion(
            freqs, weights, target_astropy
        )

        if origin_unit == Units.K_CMB:
            return factor_to_target.value

        # If origin is not K_CMB, we calculate the factor for the origin unit
        # and take the ratio.
        # Value_Target = Value_K_CMB * C_target
        # Value_Origin = Value_K_CMB * C_origin  =>  Value_K_CMB = Value_Origin / C_origin
        # Value_Target = (Value_Origin / C_origin) * C_target
        # Factor = C_target / C_origin
        factor_to_origin = pysm3.bandpass_unit_conversion(
            freqs, weights, origin_astropy
        )
        return (factor_to_target / factor_to_origin).value

    else:
        # Monochromatic conversion using Astropy equivalencies
        if bandpass is not None:
            effective_freq = bandpass.bandcenter_ghz
        else:
            effective_freq = freq_ghz

        equiv = u.cmb_equivalencies(effective_freq * u.GHz)
        factor = (1.0 * origin_astropy).to(target_astropy, equivalencies=equiv)
        return factor.value


# --- Configuration DataClasses ---


@dataclass
class SkyGenerationParams:
    """Parameters for sky signal generation."""

    nside: int = 512
    lmax: int | None = None

    # Output Control
    output_type: Literal["map", "alm"] = "map"
    units: Units = Units.K_CMB  # Updated to use Enum

    # Beam & Smoothing
    apply_beam: bool = False
    apply_pixel_window: bool = False

    # Bandpass
    bandpass_integration: bool = False

    # Parallelism
    nthreads: int = 0  # 0 usually means "use all available" in ducc0

    # Components to generate
    make_cmb: bool = True
    make_fg: bool = False
    make_dipole: bool = False

    return_components: bool = False

    # CMB Specifics
    # Assume input power spectrum in uK^2
    cmb_ps_file: str | Path | None = None
    seed_cmb: int | None = None
    cmb_r: float = 0.0

    # Foreground Specifics
    fg_models: list[str] = field(default_factory=list)
    fg_oversampling: int = 2

    # Dipole Specifics
    # This simulates first order dipole (no kinematic high order terms)
    sun_velocity_kms: float = c.SOLAR_VELOCITY_KM_S
    sun_direction_galactic: tuple[float, float] = (
        c.SOLAR_VELOCITY_GAL_LAT_RAD,
        c.SOLAR_VELOCITY_GAL_LON_RAD,
    )

    def __post_init__(self):
        if self.lmax is None:
            self.lmax = 3 * self.nside - 1

        # Robustness: Ensure units is Enum if user passed string
        if isinstance(self.units, str):
            self.units = Units(self.units)

        # Warning for MPI consistency
        if self.make_cmb and self.seed_cmb is None:
            log.warning(
                "seed_cmb is None. If this simulation is running across multiple MPI tasks, "
                "the generated CMB sky will NOT be coherent (identical) across tasks. "
                "Set a specific integer seed to ensure consistency."
            )


# --- Main Class ---


class SkyGenerator:
    """
    Simplified sky signal generator (CMB, FG, Dipole).
    Returns in-memory litebird_sim objects in Galactic coordinates.
    """

    def __init__(
        self,
        parameters: SkyGenerationParams,
        channels: (
            FreqChannelInfo | DetectorInfo | list[FreqChannelInfo | DetectorInfo]
        ),
    ):
        self.params = parameters

        if isinstance(channels, list):
            self.channels = channels
        else:
            self.channels = [channels]

        # --- VALIDATION (New) ---
        if not UnitUtils.is_pysm3_compatible(self.params.units):
            raise ValueError(
                f"Unit '{self.params.units.name}' is not compatible with PySM 3 generation. "
                f"Please select a Flux or Temperature unit."
            )

        # We store the Astropy Unit object for PySM calls (same name as before for consistency)
        self.pysm_units = UnitUtils.get_astropy_unit(self.params.units)

        # Coordinate system is strictly Galactic
        self.coords = CoordinateSystem.Galactic

        # Resolve Unit object for lbs objects
        # Direct assignment since we use the Enum now
        self.lbs_unit = self.params.units

    def generate_cmb(self) -> dict[str, Any]:
        """Generates CMB component."""
        log.info("Generating CMB...")

        # 1. Get Cls using Astropy
        if self.params.cmb_ps_file:
            cl_cmb = read_cls_from_fits(self.params.cmb_ps_file)
        else:
            datautils_dir = Path(__file__).parent / "datautils"

            cl_cmb_scalar = read_cls_from_fits(
                datautils_dir / "Cls_Planck2018_for_PTEP_2020_r0.fits"
            )
            cl_cmb_tensor = read_cls_from_fits(
                datautils_dir / "Cls_Planck2018_for_PTEP_2020_tensor_r1.fits"
            )
            cl_cmb = lin_comb_cls(cl_cmb_scalar, cl_cmb_tensor, s2=self.params.cmb_r)

        # Default units of the input cmb K_CMB
        cl_cmb = lin_comb_cls(cl_cmb, s1=1e-12)

        # 2. Generate ALMs using maps_and_harmonics.,size_alm
        rng = np.random.default_rng(self.params.seed_cmb)
        alm_cmb = synthesize_alm(
            cl_dict=cl_cmb,
            lmax=self.params.lmax,
            mmax=self.params.lmax,
            coordinates=self.coords,
            rng=rng,
            units=self.lbs_unit,
        )

        result = {}
        for ch in self.channels:
            name = ch.name if hasattr(ch, "name") else ch.channel
            log.debug(f"Processing CMB for {name}")

            # Copy to avoid destructive inplace modifications
            alm_obs = alm_cmb.copy()

            # 3. Apply Beam and Window
            if self.params.apply_beam and ch.fwhm_arcmin > 0:
                fwhm_rad = np.radians(ch.fwhm_arcmin / 60.0)
                alm_obs.apply_gaussian_smoothing(fwhm_rad=fwhm_rad, inplace=True)

            if self.params.apply_pixel_window:
                alm_obs.apply_pixel_window(nside=self.params.nside, inplace=True)

            # Unit conversion
            if self.params.bandpass_integration:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    bandpass=ch.band,
                    band_integration=True,
                )
            else:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    freq_ghz=ch.bandcenter_ghz,
                    band_integration=False,
                )

            alm_obs *= conv_factor

            # 5. Output (ALM or MAP)
            if self.params.output_type == "map":
                # Use maps_and_harmonics.pixelize_alm
                map_obs = pixelize_alm(
                    alm_obs,
                    nside=self.params.nside,
                    lmax=self.params.lmax,
                    nthreads=self.params.nthreads,
                )
                result[name] = map_obs
            else:
                result[name] = alm_obs

        return result

    def generate_foregrounds(self) -> dict[str, Any]:
        """Generates foregrounds using PySM3."""
        if not self.params.fg_models:
            return {}

        log.info("Generating Foregrounds...")

        nside_fg = self.params.nside * self.params.fg_oversampling
        sky = pysm3.Sky(nside=nside_fg, preset_strings=list(self.params.fg_models))

        result = {}
        for ch in self.channels:
            name = ch.name if hasattr(ch, "name") else ch.channel

            # 1. Compute Emission
            if self.params.bandpass_integration:
                nonzero = np.where(ch.band.freqs_ghz != 0)[0]
                bandpass_frequencies = ch.band.freqs_ghz[nonzero] * u.GHz
                weights = ch.band.weights[nonzero]

                m_fg = sky.get_emission(bandpass_frequencies, weights=weights)
                # Use self.pysm_units (which is now the Astropy object)
                m_fg = m_fg * pysm3.bandpass_unit_conversion(
                    bandpass_frequencies, weights, self.pysm_units
                )
            else:
                m_fg = sky.get_emission(ch.bandcenter_ghz * u.GHz)
                m_fg = m_fg.to(
                    self.pysm_units,
                    equivalencies=u.cmb_equivalencies(ch.bandcenter_ghz * u.GHz),
                )

            # Convert to target unit
            m_fg_val = m_fg.value  # Numpy array (3, npix)

            # Wrap in HealpixMap
            map_fg = HealpixMap(
                values=m_fg_val,
                nside=nside_fg,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,  # PySM uses Ring
            )

            alm_fg = estimate_alm(
                map_fg,
                lmax=self.params.lmax,
                nthreads=self.params.nthreads,
            )

            if self.params.apply_beam and ch.fwhm_arcmin > 0:
                fwhm_rad = np.radians(ch.fwhm_arcmin / 60.0)
                alm_fg.apply_gaussian_smoothing(fwhm_rad=fwhm_rad, inplace=True)

            if self.params.apply_pixel_window:
                alm_fg.apply_pixel_window(nside=self.params.nside, inplace=True)

            if self.params.output_type == "map":
                # Transform back if we went to ALM space for beam
                map_fg = pixelize_alm(
                    alm_fg,
                    nside=self.params.nside,
                    lmax=self.params.lmax,
                    nthreads=self.params.nthreads,
                )
                result[name] = map_fg
            else:
                result[name] = alm_fg

        return result

    def generate_solar_dipole(self) -> dict[str, Any]:
        """Generates solar dipole."""
        log.info("Generating Dipole...")
        velocity = self.params.sun_velocity_kms
        lat, lon = self.params.sun_direction_galactic

        # Amplitude in K_CMB
        amp = c.T_CMB_K * (velocity / c.C_LIGHT_KM_OVER_S)

        # Template Map
        hpx = dh.Healpix_Base(self.params.nside, "RING")
        npix = hpx.npix()

        # Dipole Vector
        vec = dh.ang2vec(np.array([[lat, lon]]))[0]

        pix_vecs = hpx.pix2vec(np.arange(npix))

        # Expand to IQU
        dipole_map_val = np.zeros((3, npix))
        dipole_map_val[0] = np.dot(pix_vecs, vec) * amp

        # Wrap in HealpixMap
        # Initial map is unitless/K_CMB raw

        result = {}
        for ch in self.channels:
            name = ch.name if hasattr(ch, "name") else ch.channel

            # Unit conversion
            if self.params.bandpass_integration:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    bandpass=ch.band,
                    band_integration=True,
                )
            else:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    freq_ghz=ch.bandcenter_ghz,
                    band_integration=False,
                )

            m_dip = HealpixMap(
                values=dipole_map_val * conv_factor,
                nside=self.params.nside,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
            )

            if self.params.output_type == "alm":
                result[name] = estimate_alm(
                    m_dip,
                    lmax=self.params.lmax,
                    nthreads=self.params.nthreads,
                )
            else:
                result[name] = m_dip

        return result

    def execute(self) -> dict[str, Any] | dict[str, dict[str, Any]]:
        """
        Executes the generation pipeline.
        Returns a dictionary of objects (maps or alms) or a dictionary of components if requested.
        """
        components = {}

        if self.params.make_cmb:
            components["cmb"] = self.generate_cmb()

        if self.params.make_fg:
            components["foregrounds"] = self.generate_foregrounds()

        if self.params.make_dipole:
            components["dipole"] = self.generate_solar_dipole()

        if self.params.return_components:
            components["SkyGenerationParams"] = self.params
            return components

        # Sum components
        log.info("Summing components...")
        total_maps = {}

        if not components:
            return {}

        first_comp_name = list(components.keys())[0]
        first_comp_data = components[first_comp_name]

        for name in first_comp_data:
            # Initialize with the first component (copy to avoid modification)
            total_obj = first_comp_data[name].copy()

            for comp_name in components:
                if comp_name == first_comp_name:
                    continue
                # Add other components (HealpixMap/SphericalHarmonics support + operator)
                # Assumes units and coordinates are compatible (handled by class checks)
                total_obj += components[comp_name][name]

            total_maps[name] = total_obj
        total_maps["SkyGenerationParams"] = self.params

        return total_maps
