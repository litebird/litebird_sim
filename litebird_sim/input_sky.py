import logging as log
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from collections.abc import Sequence

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


@dataclass(frozen=True)
class _GenerationTarget:
    name: str | None
    freq_ghz: float
    bandpass: BandPassInfo | None
    fwhm_rad: float | None


class SkyGenerationParams:
    """Parameters for sky signal generation."""

    def __init__(
        self,
        nside: int = 512,
        lmax: int | None = None,
        maxiter: int | None = None,
        epsilon: float | None = None,
        # Output Control
        output_type: Literal["map", "alm"] = "map",
        units: str | Units = Units.K_CMB,  # Updated to use Enum
        # Beam & Smoothing
        apply_beam: bool = False,
        apply_pixel_window: bool = False,
        # Bandpass
        bandpass_integration: bool = False,
        # Parallelism
        nthreads: int = 0,  # 0 usually means "use all available" in ducc0
        # Components to generate
        make_cmb: bool = True,
        make_fg: bool = False,
        make_dipole: bool = False,
        return_components: bool = False,
        # CMB Specifics
        # Assume input power spectrum in uK^2
        cmb_ps_file: str | Path | None = None,
        seed_cmb: int | None = None,
        cmb_r: float = 0.0,
        # Foreground Specifics
        fg_models: list[str] | None = None,
        fg_oversampling: int = 2,
        # Dipole Specifics
        # This simulates first order dipole (no kinematic high order terms)
        sun_velocity_kms: float = c.SOLAR_VELOCITY_KM_S,
        sun_direction_galactic: tuple[float, float] = (
            c.SOLAR_VELOCITY_GAL_LAT_RAD,
            c.SOLAR_VELOCITY_GAL_LON_RAD,
        ),
    ):
        self.nside = nside
        self.lmax: int = lmax
        if self.lmax is None:
            self.lmax: int = 3 * self.nside - 1
        self.output_type = output_type
        self.units = Units(units) if isinstance(units, str) else units
        self.apply_beam = apply_beam
        self.apply_pixel_window = apply_pixel_window
        self.bandpass_integration = bandpass_integration
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.nthreads = nthreads
        self.make_cmb = make_cmb
        self.make_fg = make_fg
        self.make_dipole = make_dipole
        self.return_components = return_components
        self.cmb_ps_file = cmb_ps_file
        self.seed_cmb = seed_cmb
        self.cmb_r = cmb_r
        self.fg_models: list[str] = fg_models if fg_models is not None else []
        self.fg_oversampling = fg_oversampling
        self.sun_velocity_kms = sun_velocity_kms
        self.sun_direction_galactic = sun_direction_galactic


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
       - Provide `frequencies_ghz` (single float or array of frequencies)
       - Single frequency output: HealpixMap or SphericalHarmonics with 2D arrays (nstokes, size)
       - Multi-frequency output: HealpixMap or SphericalHarmonics with 3D arrays (nfreqs, nstokes, size)
       - The `frequencies_ghz` attribute contains the frequency array.
       - `return_components=True` is not supported in this mode (a warning is raised and only the total is returned).
    """

    def __init__(
        self,
        parameters: SkyGenerationParams,
        channels: FreqChannelInfo | Sequence[FreqChannelInfo] | None = None,
        detectors: DetectorInfo | Sequence[DetectorInfo] | None = None,
        frequencies_ghz: float | Sequence[float] | np.ndarray | None = None,
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
            freqs = np.atleast_1d(np.asarray(frequencies_ghz, dtype=float))
            if freqs.ndim != 1 or freqs.size == 0:
                raise ValueError(
                    "'frequencies_ghz' must be a frequency in GHz (float) or a 1D non-empty list/array of frequencies in GHz."
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
            self.channels_or_detectors: list[FreqChannelInfo] | list[DetectorInfo]
            if channels is not None:
                if isinstance(channels, FreqChannelInfo):
                    self.channels: list[FreqChannelInfo] = [channels]
                else:
                    self.channels: list[FreqChannelInfo] = list(channels)
                self.detectors: list[DetectorInfo] = []
                self.channels_or_detectors = self.channels

            elif detectors is not None:
                if isinstance(detectors, DetectorInfo):
                    self.detectors: list[DetectorInfo] = [detectors]
                else:
                    self.detectors: list[DetectorInfo] = list(detectors)
                self.channels: list[FreqChannelInfo] = []
                self.channels_or_detectors = self.detectors
            else:
                raise ValueError(
                    "Internal error: No channels or detectors provided, but not in frequency mode. This should have been caught by validation."
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate(
        total: HealpixMap | SphericalHarmonics,
        component: HealpixMap | SphericalHarmonics,
    ) -> HealpixMap | SphericalHarmonics:
        if isinstance(total, HealpixMap) and isinstance(component, HealpixMap):
            total += component
        elif isinstance(total, SphericalHarmonics) and isinstance(
            component, SphericalHarmonics
        ):
            total += component
        else:
            raise TypeError(
                f"Cannot add {type(component).__name__} to {type(total).__name__}"
            )
        return total

    @staticmethod
    def _channel_or_detector_name(ch_or_det: FreqChannelInfo | DetectorInfo) -> str:
        return str(ch_or_det.name if hasattr(ch_or_det, "name") else ch_or_det.channel)

    @staticmethod
    def _channel_or_detector_fwhm_rad(
        ch_or_det: FreqChannelInfo | DetectorInfo,
    ) -> float | None:
        fwhm_arcmin = getattr(ch_or_det, "fwhm_arcmin", 0)
        if fwhm_arcmin <= 0:
            return None
        return float(np.radians(fwhm_arcmin / 60.0))

    def _get_generation_targets(self) -> list[_GenerationTarget]:
        if self.frequency_mode:
            assert self.frequencies_ghz is not None
            targets: list[_GenerationTarget] = []
            for i, freq_ghz in enumerate(self.frequencies_ghz):
                fwhm_rad = None if self.fwhm_rad is None else float(self.fwhm_rad[i])
                targets.append(
                    _GenerationTarget(
                        name=None,
                        freq_ghz=float(freq_ghz),
                        bandpass=None,
                        fwhm_rad=fwhm_rad,
                    )
                )
            return targets

        return [
            _GenerationTarget(
                name=self._channel_or_detector_name(ch_or_det),
                freq_ghz=float(ch_or_det.bandcenter_ghz),
                bandpass=ch_or_det.band,
                fwhm_rad=self._channel_or_detector_fwhm_rad(ch_or_det),
            )
            for ch_or_det in self.channels_or_detectors
        ]

    def _build_frequency_output(
        self, values: np.ndarray
    ) -> HealpixMap | SphericalHarmonics:
        assert self.frequencies_ghz is not None
        nfreq = self.frequencies_ghz.size
        if self.params.output_type == "map":
            return HealpixMap(
                values=values,
                nside=self.params.nside,
                nfreqs=nfreq,
                units=self.lbs_unit,
                coordinates=self.coords,
                nest=False,
                frequencies_ghz=self.frequencies_ghz.copy(),
            )

        return SphericalHarmonics(
            values=values,
            lmax=self.params.lmax,
            mmax=self.params.lmax,
            nfreqs=nfreq,
            coordinates=self.coords,
            units=self.lbs_unit,
            frequencies_ghz=self.frequencies_ghz.copy(),
        )

    def _stack_frequency_outputs(
        self, outputs: Sequence[HealpixMap | SphericalHarmonics]
    ) -> HealpixMap | SphericalHarmonics:
        values = np.stack([output.values for output in outputs], axis=0)
        return self._build_frequency_output(values)

    def _collect_target_outputs(self, build_output):
        targets = self._get_generation_targets()

        if self.frequency_mode:
            outputs = [build_output(target) for target in targets]
            return self._stack_frequency_outputs(outputs)

        result: dict[str, HealpixMap | SphericalHarmonics] = {}
        for target in targets:
            assert target.name is not None
            result[target.name] = build_output(target)
        return result

    def _cmb_unit_conversion_for_target(self, target: _GenerationTarget) -> float:
        if self.params.bandpass_integration and target.bandpass is not None:
            return _get_cmb_unit_conversion(
                target_unit=self.params.units,
                bandpass=target.bandpass,
                band_integration=True,
            )

        return _get_cmb_unit_conversion(
            target_unit=self.params.units,
            freq_ghz=target.freq_ghz,
            band_integration=False,
        )

    def _healpix_map_from_values(self, values: np.ndarray, nside: int) -> HealpixMap:
        return HealpixMap(
            values=values,
            nside=nside,
            units=self.lbs_unit,
            coordinates=self.coords,
            nest=False,
        )

    def _alm_to_output(self, alm: SphericalHarmonics) -> HealpixMap | SphericalHarmonics:
        if self.params.output_type == "map":
            return pixelize_alm(
                alm,
                nside=self.params.nside,
                lmax=self.params.lmax,
                nthreads=self.params.nthreads,
            )
        return alm

    def _map_to_output(self, map_obj: HealpixMap) -> HealpixMap | SphericalHarmonics:
        if self.params.output_type == "alm":
            return estimate_alm(
                map_obj,
                lmax=self.params.lmax,
                nthreads=self.params.nthreads,
                maxiter=self.params.maxiter,
                epsilon=self.params.epsilon,
            )
        return map_obj

    def _empty_frequency_output(self) -> HealpixMap | SphericalHarmonics:
        """Return a zero-filled multi-frequency HealpixMap or SphericalHarmonics object."""
        assert self.frequencies_ghz is not None
        nfreq = self.frequencies_ghz.size
        if self.params.output_type == "map":
            npix = dh.Healpix_Base(self.params.nside, "RING").npix()
            values = np.zeros((nfreq, 3, npix), dtype=float)
            return self._build_frequency_output(values)
        else:
            # SphericalHarmonics values are complex
            nalms = SphericalHarmonics.alm_array_size(
                self.params.lmax, self.params.lmax, 3
            )[1]
            values = np.zeros((nfreq, 3, nalms), dtype=np.complex128)
            return self._build_frequency_output(values)

    def _apply_smoothing_and_windows_to_alm(
        self, alm: SphericalHarmonics, fwhm_rad: float | None
    ) -> SphericalHarmonics:
        """Apply gaussian smoothing and pixel window in-place and return the object."""
        if self.params.apply_beam and fwhm_rad is not None and fwhm_rad > 0:
            alm.apply_gaussian_smoothing(fwhm_rad=fwhm_rad, inplace=True)
        if self.params.apply_pixel_window:
            alm.apply_pixel_window(nside=self.params.nside, inplace=True)
        return alm

    def _generate_cmb_common(
        self,
    ) -> HealpixMap | SphericalHarmonics | dict[str, HealpixMap | SphericalHarmonics]:
        log.info(
            "Generating CMB%s...",
            " (frequency mode)" if self.frequency_mode else "",
        )

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

        def build_output(target: _GenerationTarget) -> HealpixMap | SphericalHarmonics:
            alm_obs = alm_cmb.copy()
            self._apply_smoothing_and_windows_to_alm(alm_obs, target.fwhm_rad)
            alm_obs *= self._cmb_unit_conversion_for_target(target)
            return self._alm_to_output(alm_obs)

        return self._collect_target_outputs(build_output)

    def _foreground_map_for_target(
        self, sky: pysm3.Sky, target: _GenerationTarget, nside_fg: int
    ) -> HealpixMap:
        if self.params.bandpass_integration and target.bandpass is not None:
            nonzero = np.where(getattr(target.bandpass, "freqs_ghz") != 0)[0]
            bandpass_frequencies = getattr(target.bandpass, "freqs_ghz")[nonzero] * getattr(
                u, "GHz"
            )
            weights = getattr(target.bandpass, "weights")[nonzero]

            m_fg = sky.get_emission(bandpass_frequencies, weights=weights)
            m_fg = m_fg * pysm3.bandpass_unit_conversion(
                bandpass_frequencies, weights, self.pysm_units
            )
        else:
            m_fg = sky.get_emission(target.freq_ghz * getattr(u, "GHz"))
            m_fg = m_fg.to(
                self.pysm_units,
                equivalencies=u.cmb_equivalencies(target.freq_ghz * getattr(u, "GHz")),
            )

        return self._healpix_map_from_values(m_fg.value, nside=nside_fg)

    def _generate_foregrounds_common(
        self,
    ) -> HealpixMap | SphericalHarmonics | dict[str, HealpixMap | SphericalHarmonics]:
        if not self.params.fg_models:
            if self.frequency_mode:
                return self._empty_frequency_output()
            return {}

        log.info(
            "Generating Foregrounds%s...",
            " (frequency mode)" if self.frequency_mode else "",
        )

        nside_fg = self.params.nside * self.params.fg_oversampling
        sky = pysm3.Sky(nside=nside_fg, preset_strings=list(self.params.fg_models))

        def build_output(target: _GenerationTarget) -> HealpixMap | SphericalHarmonics:
            map_fg = self._foreground_map_for_target(sky, target, nside_fg)
            alm_fg = estimate_alm(
                map_fg,
                lmax=self.params.lmax,
                nthreads=self.params.nthreads,
                maxiter=self.params.maxiter,
                epsilon=self.params.epsilon,
            )
            self._apply_smoothing_and_windows_to_alm(alm_fg, target.fwhm_rad)
            return self._alm_to_output(alm_fg)

        return self._collect_target_outputs(build_output)

    def _dipole_map_values(self) -> np.ndarray:
        velocity = self.params.sun_velocity_kms
        lat, lon = self.params.sun_direction_galactic

        amp = c.T_CMB_K * (velocity / c.C_LIGHT_KM_OVER_S)

        hpx = dh.Healpix_Base(self.params.nside, "RING")
        npix = hpx.npix()

        vec = dh.ang2vec(np.array([[lat, lon]]))[0]
        pix_vecs = hpx.pix2vec(np.arange(npix))

        dipole_map_val = np.zeros((3, npix))
        dipole_map_val[0] = np.dot(pix_vecs, vec) * amp
        return dipole_map_val

    def _generate_dipole_common(
        self,
    ) -> HealpixMap | SphericalHarmonics | dict[str, HealpixMap | SphericalHarmonics]:
        log.info(
            "Generating Dipole%s...",
            " (frequency mode)" if self.frequency_mode else "",
        )

        dipole_map_val = self._dipole_map_values()

        def build_output(target: _GenerationTarget) -> HealpixMap | SphericalHarmonics:
            map_obj = self._healpix_map_from_values(
                dipole_map_val * self._cmb_unit_conversion_for_target(target),
                nside=self.params.nside,
            )
            return self._map_to_output(map_obj)

        return self._collect_target_outputs(build_output)

    # ------------------------------------------------------------------
    # Component generators (channel/detector mode)
    # ------------------------------------------------------------------

    def generate_cmb(self) -> dict[str, HealpixMap | SphericalHarmonics]:
        """Generates CMB component (channel/detector mode only)."""
        if self.frequency_mode:
            raise RuntimeError("generate_cmb() is not available in frequency mode.")
        return cast(dict[str, HealpixMap | SphericalHarmonics], self._generate_cmb_common())

    def generate_foregrounds(self) -> dict[str, HealpixMap | SphericalHarmonics]:
        """Generates foregrounds using PySM3 (channel/detector mode only)."""
        if self.frequency_mode:
            raise RuntimeError(
                "generate_foregrounds() is not available in frequency mode."
            )
        return cast(
            dict[str, HealpixMap | SphericalHarmonics],
            self._generate_foregrounds_common(),
        )

    def generate_solar_dipole(self) -> dict[str, HealpixMap | SphericalHarmonics]:
        """Generates solar dipole (channel/detector mode only)."""
        if self.frequency_mode:
            raise RuntimeError(
                "generate_solar_dipole() is not available in frequency mode."
            )
        return cast(
            dict[str, HealpixMap | SphericalHarmonics],
            self._generate_dipole_common(),
        )

    # ------------------------------------------------------------------
    # Component generators (frequency mode)
    # ------------------------------------------------------------------

    def _generate_cmb_frequencies(self) -> HealpixMap | SphericalHarmonics:
        assert self.frequencies_ghz is not None
        return cast(HealpixMap | SphericalHarmonics, self._generate_cmb_common())

    def _generate_foregrounds_frequencies(self) -> HealpixMap | SphericalHarmonics:
        assert self.frequencies_ghz is not None
        return cast(
            HealpixMap | SphericalHarmonics, self._generate_foregrounds_common()
        )

    def _generate_dipole_frequencies(self) -> HealpixMap | SphericalHarmonics:
        assert self.frequencies_ghz is not None
        return cast(HealpixMap | SphericalHarmonics, self._generate_dipole_common())

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(
        self,
    ) -> (
        HealpixMap
        | SphericalHarmonics
        | dict[str, HealpixMap | SphericalHarmonics | SkyGenerationParams]
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
                total = self._accumulate(total, self._generate_cmb_frequencies())
                any_component = True

            if self.params.make_fg:
                total = self._accumulate(
                    total, self._generate_foregrounds_frequencies()
                )
                any_component = True

            if self.params.make_dipole:
                total = self._accumulate(total, self._generate_dipole_frequencies())
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
            return components

        log.info("Summing components...")
        total_maps: dict[
            str, HealpixMap | SphericalHarmonics | SkyGenerationParams
        ] = {}

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
