import logging as log
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import ducc0.healpix as dh
import numpy as np
import pysm3
import pysm3.units as u

from litebird_sim import constants as c

from .bandpasses import BandPassInfo
from .coordinates import CoordinateSystem
from .detectors import DetectorInfo, FreqChannelInfo
from .maps_and_harmonics import (
    HealpixMap,
    SphericalHarmonics,
    estimate_alm,
    lin_comb_cls,
    pixelize_alm,
    read_cls_from_fits,
    synthesize_alm,
)
from .units import Units, UnitUtils

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
        nonzero = np.where(getattr(bandpass, "freqs_ghz") != 0)[0]
        freqs = getattr(bandpass, "freqs_ghz")[nonzero] * getattr(u, "GHz")
        weights = getattr(bandpass, "weights")[nonzero]

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

        equiv = u.cmb_equivalencies(effective_freq * getattr(u, "GHz"))
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
    units: str | Units = Units.K_CMB  # Updated to use Enum

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
            self.units: Units = Units(self.units)

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

    Notes
    -----
    This class supports two mutually-exclusive modes:

    1) Channel/Detector mode (legacy):
       - Provide `channels` OR `detectors`
       - Output is a dict keyed by channel/detector name, values are HealpixMap/SphericalHarmonics.

    2) Frequency mode:
       - Provide `frequencies_ghz`
       - Output is a multi-frequency HealpixMap or SphericalHarmonics object.
       - The `values` attribute has shape (nfreqs, 3, npix) or (nfreqs, 3, nalms).
       - The `frequencies_ghz` attribute contains the frequency array.
       - `return_components=True` is not supported in this mode (a warning is raised and only the total is returned).
    """

    def __init__(
        self,
        parameters: SkyGenerationParams,
        channels: FreqChannelInfo | Sequence[FreqChannelInfo] | None = None,
        detectors: DetectorInfo | Sequence[DetectorInfo] | None = None,
        frequencies_ghz: Sequence[float] | np.ndarray | None = None,
        fwhm_rad: float | Sequence[float] | np.ndarray | None = None,
    ):
        self.params: SkyGenerationParams = parameters

        provided = [
            channels is not None,
            detectors is not None,
            frequencies_ghz is not None,
        ]
        if sum(provided) == 0:
            raise ValueError(
                "You must provide exactly one of 'channels', 'detectors', or 'frequencies_ghz'."
            )
        if sum(provided) > 1:
            raise ValueError(
                "Only one of 'channels', 'detectors', or 'frequencies_ghz' can be provided at the same time."
            )

        # --- VALIDATION (Units / PySM compatibility) ---
        assert isinstance(self.params.units, Units)
        if not UnitUtils.is_pysm3_compatible(self.params.units):
            raise ValueError(
                f"Unit '{self.params.units.name}' is not compatible with PySM 3 generation. "
                f"Please select a Flux or Temperature unit."
            )

        # Store Astropy Unit object for PySM calls
        self.pysm_units = UnitUtils.get_astropy_unit(self.params.units)

        # Coordinate system is strictly Galactic
        self.coords = CoordinateSystem.Galactic

        # Unit for litebird_sim objects
        self.lbs_unit = self.params.units

        # --- Mode selection ---
        self.frequency_mode: bool = frequencies_ghz is not None

        # --- Frequency mode setup ---
        self.frequencies_ghz: np.ndarray | None = None
        self.fwhm_rad: np.ndarray | None = None

        if self.frequency_mode:
            freqs = np.asarray(frequencies_ghz, dtype=float)
            if freqs.ndim != 1 or freqs.size == 0:
                raise ValueError(
                    "'frequencies_ghz' must be a 1D non-empty list/array of frequencies in GHz."
                )
            self.frequencies_ghz = freqs
            nfreq = freqs.size

            if self.params.bandpass_integration:
                warnings.warn(
                    "'frequencies_ghz' was provided but 'bandpass_integration=True'. "
                    "Bandpass integration will be ignored in frequency mode."
                )

            if self.params.return_components:
                warnings.warn(
                    "'return_components=True' is not supported when 'frequencies_ghz' is provided. "
                    "Only the total sky will be returned."
                )

            # Beam handling rules for frequency mode
            if self.params.apply_beam:
                if fwhm_rad is None:
                    raise ValueError(
                        "apply_beam=True requires 'fwhm_rad' in frequency mode, but fwhm_rad is None."
                    )
                fwhm = np.asarray(fwhm_rad, dtype=float)
                if fwhm.ndim == 0:
                    self.fwhm_rad = np.full(nfreq, float(fwhm))
                else:
                    if fwhm.ndim != 1 or fwhm.size != nfreq:
                        raise ValueError(
                            "'fwhm_rad' must be a scalar or a 1D array/list with the same length as 'frequencies_ghz'."
                        )
                    self.fwhm_rad = fwhm
            else:
                if fwhm_rad is not None:
                    warnings.warn(
                        "'fwhm_rad' was provided but 'apply_beam=False'. The fwhm will be ignored."
                    )
                self.fwhm_rad = None

            # In frequency mode, channels/detectors are not used.
            self.channels = []
            self.detectors = []
            self.channels_or_detectors = []

        # --- Channel/Detector (legacy) mode setup ---
        else:
            if fwhm_rad is not None:
                if self.params.apply_beam:
                    warnings.warn(
                        "'fwhm_rad' was provided but 'frequencies_ghz' is None. "
                        "In channel/detector mode, the beam comes from the channel/detector definition; "
                        "the provided fwhm_rad will be ignored."
                    )
                else:
                    warnings.warn(
                        "'fwhm_rad' was provided but 'apply_beam=False'. The fwhm will be ignored."
                    )

            if channels is not None:
                if isinstance(channels, (list, tuple)):
                    self.channels = list(channels)
                else:
                    self.channels = [channels]
                self.detectors = []
                self.channels_or_detectors = self.channels

            else:
                # detectors is not None here by construction
                if isinstance(detectors, (list, tuple)):
                    self.detectors = list(detectors)
                else:
                    self.detectors = [detectors]
                self.channels = []
                self.channels_or_detectors = self.detectors

    # ------------------------------------------------------------------
    # Helpers (frequency mode)
    # ------------------------------------------------------------------

    def _empty_frequency_output(self) -> HealpixMap | SphericalHarmonics:
        """Return a zero-filled multi-frequency HealpixMap or SphericalHarmonics object."""
        assert self.frequencies_ghz is not None
        nfreq = self.frequencies_ghz.size
        if self.params.output_type == "map":
            npix = dh.Healpix_Base(self.params.nside, "RING").npix()
            values = np.zeros((nfreq, 3, npix), dtype=float)
            return HealpixMap(
                values=values,
                nside=self.params.nside,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )
        else:
            # SphericalHarmonics values are complex
            nalms = SphericalHarmonics.alm_array_size(
                self.params.lmax, self.params.lmax, 3
            )[1]
            values = np.zeros((nfreq, 3, nalms), dtype=np.complex128)
            return SphericalHarmonics(
                values=values,
                lmax=self.params.lmax,
                mmax=self.params.lmax,
                coordinates=self.coords,
                units=self.lbs_unit,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )

    def _apply_smoothing_and_windows_to_alm(
        self, alm: SphericalHarmonics, fwhm_rad: float | None
    ) -> SphericalHarmonics:
        """Apply gaussian smoothing and pixel window in-place and return the object."""
        if self.params.apply_beam and fwhm_rad is not None and fwhm_rad > 0:
            alm.apply_gaussian_smoothing(fwhm_rad=fwhm_rad, inplace=True)
        if self.params.apply_pixel_window:
            alm.apply_pixel_window(nside=self.params.nside, inplace=True)
        return alm

    # ------------------------------------------------------------------
    # Component generators (channel/detector mode)
    # ------------------------------------------------------------------

    def generate_cmb(self) -> dict[str, HealpixMap | SphericalHarmonics]:
        """Generates CMB component (channel/detector mode only)."""
        if self.frequency_mode:
            raise RuntimeError("generate_cmb() is not available in frequency mode.")

        log.info("Generating CMB...")

        # 1. Get Cls
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

        # Default units of the input cmb K_CMB (Cls in uK^2 -> K^2)
        cl_cmb = lin_comb_cls(cl_cmb, s1=1e-12)

        # 2. Generate ALMs
        rng = np.random.default_rng(self.params.seed_cmb)
        alm_cmb = synthesize_alm(
            cl_dict=cl_cmb,
            lmax=self.params.lmax,
            mmax=self.params.lmax,
            coordinates=self.coords,
            rng=rng,
            units=self.lbs_unit,
        )

        result: dict[str, HealpixMap | SphericalHarmonics] = {}

        for ch_or_det in self.channels_or_detectors:
            name = ch_or_det.name if hasattr(ch_or_det, "name") else ch_or_det.channel
            log.debug(f"Processing CMB for {name}")

            alm_obs = alm_cmb.copy()

            # Beam / window from channel/detector
            if self.params.apply_beam and getattr(ch_or_det, "fwhm_arcmin", 0) > 0:
                fwhm_rad = np.radians(ch_or_det.fwhm_arcmin / 60.0)
                alm_obs.apply_gaussian_smoothing(fwhm_rad=fwhm_rad, inplace=True)

            if self.params.apply_pixel_window:
                alm_obs.apply_pixel_window(nside=self.params.nside, inplace=True)

            # Unit conversion
            if self.params.bandpass_integration:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    bandpass=ch_or_det.band,
                    band_integration=True,
                )
            else:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    freq_ghz=ch_or_det.bandcenter_ghz,
                    band_integration=False,
                )

            alm_obs *= conv_factor

            # Output
            if self.params.output_type == "map":
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

    def generate_foregrounds(self) -> dict[str, HealpixMap | SphericalHarmonics]:
        """Generates foregrounds using PySM3 (channel/detector mode only)."""
        if self.frequency_mode:
            raise RuntimeError(
                "generate_foregrounds() is not available in frequency mode."
            )

        if not self.params.fg_models:
            return {}

        log.info("Generating Foregrounds...")

        nside_fg = self.params.nside * self.params.fg_oversampling
        sky = pysm3.Sky(nside=nside_fg, preset_strings=list(self.params.fg_models))

        result: dict[str, HealpixMap | SphericalHarmonics] = {}

        for ch_or_det in self.channels_or_detectors:
            name = ch_or_det.name if hasattr(ch_or_det, "name") else ch_or_det.channel

            # 1. Compute emission
            if self.params.bandpass_integration:
                nonzero = np.where(getattr(ch_or_det.band, "freqs_ghz") != 0)[0]
                bandpass_frequencies = getattr(ch_or_det.band, "freqs_ghz")[
                    nonzero
                ] * getattr(u, "GHz")
                weights = getattr(ch_or_det.band, "weights")[nonzero]

                m_fg = sky.get_emission(bandpass_frequencies, weights=weights)
                m_fg = m_fg * pysm3.bandpass_unit_conversion(
                    bandpass_frequencies, weights, self.pysm_units
                )
            else:
                m_fg = sky.get_emission(ch_or_det.bandcenter_ghz * getattr(u, "GHz"))
                m_fg = m_fg.to(
                    self.pysm_units,
                    equivalencies=u.cmb_equivalencies(
                        ch_or_det.bandcenter_ghz * getattr(u, "GHz")
                    ),
                )

            m_fg_val = m_fg.value  # (3, npix)
            map_fg = HealpixMap(
                values=m_fg_val,
                nside=nside_fg,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
            )

            alm_fg = estimate_alm(
                map_fg,
                lmax=self.params.lmax,
                nthreads=self.params.nthreads,
            )

            if self.params.apply_beam and getattr(ch_or_det, "fwhm_arcmin", 0) > 0:
                fwhm_rad = np.radians(ch_or_det.fwhm_arcmin / 60.0)
                alm_fg.apply_gaussian_smoothing(fwhm_rad=fwhm_rad, inplace=True)

            if self.params.apply_pixel_window:
                alm_fg.apply_pixel_window(nside=self.params.nside, inplace=True)

            if self.params.output_type == "map":
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
        """Generates solar dipole (channel/detector mode only)."""
        if self.frequency_mode:
            raise RuntimeError(
                "generate_solar_dipole() is not available in frequency mode."
            )

        log.info("Generating Dipole...")
        velocity = self.params.sun_velocity_kms
        lat, lon = self.params.sun_direction_galactic

        # Amplitude in K_CMB
        amp = c.T_CMB_K * (velocity / c.C_LIGHT_KM_OVER_S)

        hpx = dh.Healpix_Base(self.params.nside, "RING")
        npix = hpx.npix()

        vec = dh.ang2vec(np.array([[lat, lon]]))[0]
        pix_vecs = hpx.pix2vec(np.arange(npix))

        dipole_map_val = np.zeros((3, npix))
        dipole_map_val[0] = np.dot(pix_vecs, vec) * amp

        result: dict[str, Any] = {}

        for ch_or_det in self.channels_or_detectors:
            name = ch_or_det.name if hasattr(ch_or_det, "name") else ch_or_det.channel

            if self.params.bandpass_integration:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    bandpass=ch_or_det.band,
                    band_integration=True,
                )
            else:
                conv_factor = _get_cmb_unit_conversion(
                    target_unit=self.params.units,
                    freq_ghz=ch_or_det.bandcenter_ghz,
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

    # ------------------------------------------------------------------
    # Component generators (frequency mode)
    # ------------------------------------------------------------------

    def _generate_cmb_frequencies(self) -> HealpixMap | SphericalHarmonics:
        assert self.frequencies_ghz is not None

        log.info("Generating CMB (frequency mode)...")

        # 1. Get Cls
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

        # uK^2 -> K^2
        cl_cmb = lin_comb_cls(cl_cmb, s1=1e-12)

        rng = np.random.default_rng(self.params.seed_cmb)
        alm_cmb = synthesize_alm(
            cl_dict=cl_cmb,
            lmax=self.params.lmax,
            mmax=self.params.lmax,
            coordinates=self.coords,
            rng=rng,
            units=self.lbs_unit,
        )

        nfreq = self.frequencies_ghz.size
        if self.params.output_type == "map":
            npix = dh.Healpix_Base(self.params.nside, "RING").npix()
            out = np.zeros((nfreq, 3, npix), dtype=float)
        else:
            nalms = alm_cmb.values.shape[1]
            out = np.zeros((nfreq, 3, nalms), dtype=np.complex128)

        for i, nu in enumerate(self.frequencies_ghz):
            alm_obs = alm_cmb.copy()

            # Smooth/window
            fwhm = None if self.fwhm_rad is None else float(self.fwhm_rad[i])
            self._apply_smoothing_and_windows_to_alm(alm_obs, fwhm)

            # Units
            conv_factor = _get_cmb_unit_conversion(
                target_unit=self.params.units,
                freq_ghz=float(nu),
                band_integration=False,
            )
            alm_obs *= conv_factor

            if self.params.output_type == "map":
                m = pixelize_alm(
                    alm_obs,
                    nside=self.params.nside,
                    lmax=self.params.lmax,
                    nthreads=self.params.nthreads,
                )
                out[i] = m.values
            else:
                out[i] = alm_obs.values

        # Return multi-frequency object
        if self.params.output_type == "map":
            return HealpixMap(
                values=out,
                nside=self.params.nside,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )
        else:
            return SphericalHarmonics(
                values=out,
                lmax=self.params.lmax,
                mmax=self.params.lmax,
                coordinates=self.coords,
                units=self.lbs_unit,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )

    def _generate_foregrounds_frequencies(self) -> HealpixMap | SphericalHarmonics:
        assert self.frequencies_ghz is not None

        if not self.params.fg_models:
            return self._empty_frequency_output()

        log.info("Generating Foregrounds (frequency mode)...")

        nside_fg = self.params.nside * self.params.fg_oversampling
        sky = pysm3.Sky(nside=nside_fg, preset_strings=list(self.params.fg_models))

        nfreq = self.frequencies_ghz.size
        if self.params.output_type == "map":
            npix = dh.Healpix_Base(self.params.nside, "RING").npix()
            out = np.zeros((nfreq, 3, npix), dtype=float)
        else:
            nalms = SphericalHarmonics.alm_array_size(
                self.params.lmax, self.params.lmax, 3
            )[1]
            out = np.zeros((nfreq, 3, nalms), dtype=np.complex128)

        for i, nu in enumerate(self.frequencies_ghz):
            # Emission at frequency (monochromatic)
            m_fg = sky.get_emission(float(nu) * getattr(u, "GHz"))
            m_fg = m_fg.to(
                self.pysm_units,
                equivalencies=u.cmb_equivalencies(float(nu) * getattr(u, "GHz")),
            )

            map_fg = HealpixMap(
                values=m_fg.value,
                nside=nside_fg,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
            )

            alm_fg = estimate_alm(
                map_fg,
                lmax=self.params.lmax,
                nthreads=self.params.nthreads,
            )

            fwhm = None if self.fwhm_rad is None else float(self.fwhm_rad[i])
            self._apply_smoothing_and_windows_to_alm(alm_fg, fwhm)

            if self.params.output_type == "map":
                m = pixelize_alm(
                    alm_fg,
                    nside=self.params.nside,
                    lmax=self.params.lmax,
                    nthreads=self.params.nthreads,
                )
                out[i] = m.values
            else:
                out[i] = alm_fg.values

        # Return multi-frequency object
        if self.params.output_type == "map":
            return HealpixMap(
                values=out,
                nside=self.params.nside,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )
        else:
            return SphericalHarmonics(
                values=out,
                lmax=self.params.lmax,
                mmax=self.params.lmax,
                coordinates=self.coords,
                units=self.lbs_unit,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )

    def _generate_dipole_frequencies(self) -> HealpixMap | SphericalHarmonics:
        assert self.frequencies_ghz is not None

        log.info("Generating Dipole (frequency mode)...")

        velocity = self.params.sun_velocity_kms
        lat, lon = self.params.sun_direction_galactic

        amp = c.T_CMB_K * (velocity / c.C_LIGHT_KM_OVER_S)

        hpx = dh.Healpix_Base(self.params.nside, "RING")
        npix = hpx.npix()

        vec = dh.ang2vec(np.array([[lat, lon]]))[0]
        pix_vecs = hpx.pix2vec(np.arange(npix))

        dipole_map_val = np.zeros((3, npix))
        dipole_map_val[0] = np.dot(pix_vecs, vec) * amp

        nfreq = self.frequencies_ghz.size
        if self.params.output_type == "map":
            out = np.zeros((nfreq, 3, npix), dtype=float)
        else:
            nalms = SphericalHarmonics.alm_array_size(
                self.params.lmax, self.params.lmax, 3
            )[1]
            out = np.zeros((nfreq, 3, nalms), dtype=np.complex128)

        for i, nu in enumerate(self.frequencies_ghz):
            conv_factor = _get_cmb_unit_conversion(
                target_unit=self.params.units,
                freq_ghz=float(nu),
                band_integration=False,
            )

            m_dip = HealpixMap(
                values=dipole_map_val * conv_factor,
                nside=self.params.nside,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
            )

            alm_dip = estimate_alm(
                m_dip,
                lmax=self.params.lmax,
                nthreads=self.params.nthreads,
            )

            fwhm = None if self.fwhm_rad is None else float(self.fwhm_rad[i])
            self._apply_smoothing_and_windows_to_alm(alm_dip, fwhm)

            if self.params.output_type == "map":
                m = pixelize_alm(
                    alm_dip,
                    nside=self.params.nside,
                    lmax=self.params.lmax,
                    nthreads=self.params.nthreads,
                )
                out[i] = m.values
            else:
                out[i] = alm_dip.values

        # Return multi-frequency object
        if self.params.output_type == "map":
            return HealpixMap(
                values=out,
                nside=self.params.nside,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )
        else:
            return SphericalHarmonics(
                values=out,
                lmax=self.params.lmax,
                mmax=self.params.lmax,
                coordinates=self.coords,
                units=self.lbs_unit,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(
        self,
    ) -> (
        HealpixMap
        | SphericalHarmonics
        | dict[str, HealpixMap | SphericalHarmonics]
        | dict[str, dict[str, HealpixMap | SphericalHarmonics]]
    ):
        """
        Executes the generation pipeline.

        Returns
        -------
        - Frequency mode: Multi-frequency HealpixMap or SphericalHarmonics object.
        - Channel/detector mode: dict keyed by channel/detector name.
          If return_components=True, returns a dict-of-dicts with keys: cmb/foregrounds/dipole.
        """
        # -------------------------
        # Frequency mode
        # -------------------------
        if self.frequency_mode:
            if self.params.return_components:
                warnings.warn(
                    "'return_components=True' is ignored in frequency mode. Returning only the total sky."
                )

            total = self._empty_frequency_output()
            any_component = False

            if self.params.make_cmb:
                total += self._generate_cmb_frequencies()
                any_component = True

            if self.params.make_fg:
                total += self._generate_foregrounds_frequencies()
                any_component = True

            if self.params.make_dipole:
                total += self._generate_dipole_frequencies()
                any_component = True

            if not any_component:
                # Return zeros with correct shape
                return total

            return total

        # -------------------------
        # Channel/Detector mode
        # -------------------------
        components: dict[str, dict[str, HealpixMap | SphericalHarmonics]] = {}

        if self.params.make_cmb:
            components["cmb"] = self.generate_cmb()

        if self.params.make_fg:
            components["foregrounds"] = self.generate_foregrounds()

        if self.params.make_dipole:
            components["dipole"] = self.generate_solar_dipole()

        if self.params.return_components:
            components["SkyGenerationParams"] = self.params  # type: ignore[assignment]
            return components  # type: ignore[return-value]

        log.info("Summing components...")
        total_maps: dict[str, HealpixMap | SphericalHarmonics] = {}

        if not components:
            return {}

        first_comp_name = list(components.keys())[0]
        first_comp_data = components[first_comp_name]

        for name in first_comp_data:
            total_obj = first_comp_data[name].copy()
            for comp_name in components:
                if comp_name == first_comp_name:
                    continue
                total_obj += components[comp_name][name]
            total_maps[name] = total_obj

        total_maps["SkyGenerationParams"] = self.params  # keep legacy behavior

        return total_maps
