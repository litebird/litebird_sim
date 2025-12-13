import numpy as np
from enum import Enum
from .constants import C_LIGHT_M_OVER_S, K_B, H_OVER_K_B, T_CMB_K

# --- 1. The Container ---
Units = Enum(
    "Units",
    [
        "K_CMB",
        "uK_CMB",
        "K_RJ",
        "uK_RJ",
        "MJy_over_sr",
        "Jy_over_sr",
        "ADU",
        "Pure",
    ],
)


# --- 2. The Functional Manager ---
class UnitUtils:
    """
    Static utility class to provide physical logic, conversions, and
    formatting for the Units Enum in litebird_sim.
    """

    # Mapping for LaTeX plot labels
    _LABELS = {
        Units.K_CMB: r"$K_{\mathrm{CMB}}$",
        Units.uK_CMB: r"$\mu K_{\mathrm{CMB}}$",
        Units.K_RJ: r"$K_{\mathrm{RJ}}$",
        Units.uK_RJ: r"$\mu K_{\mathrm{RJ}}$",
        Units.MJy_over_sr: r"$\mathrm{MJy}/\mathrm{sr}$",
        Units.Jy_over_sr: r"$\mathrm{Jy}/\mathrm{sr}$",
        Units.ADU: r"$\mathrm{ADU}$",
        Units.Pure: "Dimensionless",
    }

    @staticmethod
    def get_label(unit: Units) -> str:
        """Returns the LaTeX string label for the unit (e.g., for plots)."""
        return UnitUtils._LABELS.get(unit, unit.name)

    @staticmethod
    def is_temperature(unit: Units) -> bool:
        """True if unit is thermodynamic or antenna temperature."""
        return unit in [Units.K_CMB, Units.uK_CMB, Units.K_RJ, Units.uK_RJ]

    @staticmethod
    def is_flux(unit: Units) -> bool:
        """True if unit is flux density or surface brightness."""
        return unit in [Units.MJy_over_sr, Units.Jy_over_sr]

    @staticmethod
    def get_conversion_factor(
        unit_from: Units, unit_to: Units, freq_ghz: float = None
    ) -> float:
        """
        Calculates the multiplicative factor to convert between units.

        Formula: val_out = val_in * factor

        Parameters
        ----------
        unit_from : Units
            The starting unit.
        unit_to : Units
            The destination unit.
        freq_ghz : float, optional
            Frequency in Gigahertz (GHz). Required for conversions involving flux or
            transformations between CMB and RJ temperatures.
        """
        if unit_from == unit_to:
            return 1.0

        if Units.ADU in [unit_from, unit_to]:
            raise ValueError("Conversion involving ADU requires instrument Gain.")

        if Units.Pure in [unit_from, unit_to]:
            raise ValueError("Cannot convert to/from dimensionless 'Pure' units.")

        # Convert GHz to Hz for internal physics calculations
        freq_hz = freq_ghz * 1e9 if freq_ghz is not None else None

        # --- Step 1: Normalize Source to 'Pivot' (K_CMB) ---
        to_pivot = 1.0

        # Temp -> Pivot
        if unit_from == Units.uK_CMB:
            to_pivot = 1e-6
        elif unit_from == Units.K_RJ:
            to_pivot = 1.0 / UnitUtils._krj_to_kcmb_factor(freq_hz)
        elif unit_from == Units.uK_RJ:
            to_pivot = 1e-6 / UnitUtils._krj_to_kcmb_factor(freq_hz)

        # Flux -> Pivot
        elif unit_from == Units.MJy_over_sr:
            # Flux -> K_RJ -> K_CMB
            mjy_to_krj = UnitUtils._mjysr_to_krj_factor(freq_hz)
            krj_to_kcmb = 1.0 / UnitUtils._krj_to_kcmb_factor(freq_hz)
            to_pivot = mjy_to_krj * krj_to_kcmb
        elif unit_from == Units.Jy_over_sr:
            # 1 Jy = 1e-6 MJy
            mjy_to_krj = UnitUtils._mjysr_to_krj_factor(freq_hz)
            krj_to_kcmb = 1.0 / UnitUtils._krj_to_kcmb_factor(freq_hz)
            to_pivot = 1e-6 * mjy_to_krj * krj_to_kcmb

        # --- Step 2: Convert 'Pivot' (K_CMB) to Target ---
        from_pivot = 1.0

        # Pivot -> Temp
        if unit_to == Units.uK_CMB:
            from_pivot = 1e6
        elif unit_to == Units.K_RJ:
            from_pivot = UnitUtils._krj_to_kcmb_factor(freq_hz)
        elif unit_to == Units.uK_RJ:
            from_pivot = 1e6 * UnitUtils._krj_to_kcmb_factor(freq_hz)

        # Pivot -> Flux
        elif unit_to == Units.MJy_over_sr:
            # K_CMB -> K_RJ -> Flux
            kcmb_to_krj = UnitUtils._krj_to_kcmb_factor(freq_hz)
            krj_to_mjy = 1.0 / UnitUtils._mjysr_to_krj_factor(freq_hz)
            from_pivot = kcmb_to_krj * krj_to_mjy
        elif unit_to == Units.Jy_over_sr:
            kcmb_to_krj = UnitUtils._krj_to_kcmb_factor(freq_hz)
            krj_to_mjy = 1.0 / UnitUtils._mjysr_to_krj_factor(freq_hz)
            from_pivot = kcmb_to_krj * krj_to_mjy * 1e6

        return to_pivot * from_pivot

    @staticmethod
    def _krj_to_kcmb_factor(freq_hz):
        """
        Returns factor X such that T_RJ = X * T_CMB.
        Equation: (x^2 * e^x) / (e^x - 1)^2
        """
        if freq_hz is None:
            raise ValueError("Frequency is required for thermodynamic conversions.")

        # x = h * nu / (k_B * T_CMB)
        x = (H_OVER_K_B * freq_hz) / T_CMB_K

        if x < 1e-4:
            return 1.0
        return (x**2 * np.exp(x)) / ((np.exp(x) - 1) ** 2)

    @staticmethod
    def _mjysr_to_krj_factor(freq_hz):
        """
        Returns factor Y such that T_RJ [K] = Y * S [MJy/sr].
        Equation: T_RJ = (c^2 / (2 * k_B * nu^2)) * I_nu
        """
        if freq_hz is None:
            raise ValueError("Frequency is required for Flux <-> Temp conversion.")

        # 1 MJy = 10^-20 W / m^2 / Hz
        prefactor = (C_LIGHT_M_OVER_S**2) / (2 * K_B * freq_hz**2)
        return prefactor * 1e-20
