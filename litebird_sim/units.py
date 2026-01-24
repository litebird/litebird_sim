from enum import Enum
import pysm3.units as u  # Wraps astropy.units


# --- 1. The Container ---
class Units(str, Enum):
    """
    Enum for supported units in litebird_sim.

    Inheriting from str allows these members to be passed directly to functions
    expecting standard unit strings (e.g., 'uK_CMB' for PySM or Astropy),
    while ensuring type safety within the simulation pipeline.
    """

    K_CMB = "K_CMB"
    mK_CMB = "mK_CMB"
    uK_CMB = "uK_CMB"
    K_RJ = "K_RJ"
    mK_RJ = "mK_RJ"
    uK_RJ = "uK_RJ"
    MJy_over_sr = "MJy/sr"  # Valid string representation for Astropy/PySM
    Jy_over_sr = "Jy/sr"  # Valid string representation for Astropy/PySM
    ADU = "ADU"
    Pure = "dimensionless_unscaled"


# --- 2. The Functional Manager ---
class UnitUtils:
    """
    Static utility class to handle physical logic, conversions, and
    formatting for the Units Enum.

    It delegates physical constants and conversion formulas to astropy.units
    to ensure consistency with the input model generation (PySM 3).
    """

    # Mapping for LaTeX plot labels
    _LABELS = {
        Units.K_CMB: r"$K_{\mathrm{CMB}}$",
        Units.mK_CMB: r"$\mathrm{mK}_{\mathrm{CMB}}$",
        Units.uK_CMB: r"$\mu K_{\mathrm{CMB}}$",
        Units.K_RJ: r"$K_{\mathrm{RJ}}$",
        Units.mK_RJ: r"$\mathrm{mK}_{\mathrm{RJ}}$",
        Units.uK_RJ: r"$\mu K_{\mathrm{RJ}}$",
        Units.MJy_over_sr: r"$\mathrm{MJy}/\mathrm{sr}$",
        Units.Jy_over_sr: r"$\mathrm{Jy}/\mathrm{sr}$",
        Units.ADU: r"$\mathrm{ADU}$",
        Units.Pure: "Dimensionless",
    }

    @staticmethod
    def get_label(unit: Units) -> str:
        """Returns the LaTeX string label for the unit (e.g., for plotting)."""
        return UnitUtils._LABELS.get(unit, unit.value)

    @staticmethod
    def is_temperature(unit: Units) -> bool:
        """Returns True if the unit is a thermodynamic or brightness temperature."""
        return unit in [
            Units.K_CMB,
            Units.mK_CMB,
            Units.uK_CMB,
            Units.K_RJ,
            Units.mK_RJ,
            Units.uK_RJ,
        ]

    @staticmethod
    def is_flux(unit: Units) -> bool:
        """Returns True if the unit is flux density or surface brightness."""
        return unit in [Units.MJy_over_sr, Units.Jy_over_sr]

    @staticmethod
    def is_pysm3_compatible(unit: Units) -> bool:
        """Returns True if the unit is pysm3 compatible."""
        return unit in [
            Units.K_CMB,
            Units.mK_CMB,
            Units.uK_CMB,
            Units.K_RJ,
            Units.mK_RJ,
            Units.uK_RJ,
            Units.MJy_over_sr,
            Units.Jy_over_sr,
        ]

    @staticmethod
    def get_astropy_unit(unit: Units):
        """
        Returns the actual astropy.unit object corresponding to the Enum.

        This handles the instantiation of the unit object from the string value.
        """
        if unit == Units.ADU:
            # ADU is not a standard base unit in astropy; we define it locally
            # to allow quantity creation, though conversion requires Gain.
            return u.def_unit("ADU")
        return u.Unit(unit.value)

    @staticmethod
    def get_conversion_factor(
        unit_from: Units, unit_to: Units, freq_ghz: float = None
    ) -> float:
        """
        Calculates the multiplicative factor to convert between units using Astropy equivalencies.

        Formula: value_out = value_in * factor

        Parameters
        ----------
        unit_from : Units
            The source unit.
        unit_to : Units
            The destination unit.
        freq_ghz : float, optional
            Frequency in Gigahertz (GHz). Strictly required for conversions involving
            Flux (MJy/sr) or transformations between CMB and RJ temperatures.

        Returns
        -------
        float
            The scalar conversion factor.

        Raises
        ------
        ValueError
            If ADU is involved (requires instrument gain) or if frequency is missing
            for spectral conversions.
        """
        if unit_from == unit_to:
            return 1.0

        if Units.ADU in [unit_from, unit_to]:
            raise ValueError(
                "Conversion involving ADU requires instrument Gain and cannot be handled by simple unit conversion."
            )

        # Define CMB equivalencies (required for T_RJ <-> T_CMB <-> Flux)
        eq = []
        if freq_ghz is not None:
            eq = u.cmb_equivalencies(freq_ghz * u.GHz)

        # We rely on Astropy to handle the physics.
        # We create a Quantity of "1.0 unit_from" and convert it to "unit_to".
        try:
            inp_quantity = 1.0 * UnitUtils.get_astropy_unit(unit_from)
            out_value = inp_quantity.to_value(
                UnitUtils.get_astropy_unit(unit_to), equivalencies=eq
            )
            return out_value

        except u.UnitConversionError as e:
            # Raise a more descriptive error if the frequency is missing for spectral conversions
            if freq_ghz is None and (
                UnitUtils.is_flux(unit_from)
                or UnitUtils.is_flux(unit_to)
                or "CMB" in unit_from.value
                or "CMB" in unit_to.value
            ):
                raise ValueError(
                    f"Frequency (freq_ghz) is required to convert between {unit_from.name} and {unit_to.name}"
                ) from e
            raise e
