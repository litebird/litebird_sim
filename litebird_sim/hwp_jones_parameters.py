from dataclasses import dataclass, fields
from typing import IO
from uuid import UUID

import numpy as np

from litebird_sim.imo import Imo


@dataclass
class HWPJonesParams:
    """Container for frequency-dependent Jones matrix coefficients."""

    freq_ghz: np.ndarray
    Jxx_0f: np.ndarray
    Phxx_0f: np.ndarray
    Jxy_0f: np.ndarray
    Phxy_0f: np.ndarray
    Jyx_0f: np.ndarray
    Phyx_0f: np.ndarray
    Jyy_0f: np.ndarray
    Phyy_0f: np.ndarray
    Jxx_2f: np.ndarray
    Phxx_2f: np.ndarray
    Jxy_2f: np.ndarray
    Phxy_2f: np.ndarray
    Jyx_2f: np.ndarray
    Phyx_2f: np.ndarray
    Jyy_2f: np.ndarray
    Phyy_2f: np.ndarray

    @staticmethod
    def from_stream(stream: IO) -> "HWPJonesParams":
        """
        Load the Jones HWP parameters from an open text stream (CSV format).

        Expects 17 columns of floating point data as defined in LB-DFS-HWP-001.
        """
        loaded_data = np.loadtxt(
            stream,
            delimiter=",",
            dtype=np.float64,
            unpack=True,
            skiprows=1,
            comments="#",
        )

        all_fields = fields(HWPJonesParams)
        if len(loaded_data) != len(all_fields):
            raise ValueError(
                f"Expected {len(all_fields)} HWP Jones parameters, got {len(loaded_data)}"
            )

        return HWPJonesParams(
            **{f.name: data for f, data in zip(all_fields, loaded_data)}
        )

    @staticmethod
    def from_imo(imo: Imo, url: UUID | str) -> "HWPJonesParams":
        """
        Load the Jones HWP parameters from the IMo.

        The data must conform LB-DFS-HWP-001. No validation is done here,
        as it is assumed that the IMo was already validated!

        Args:
            imo: An instance to a :class:`.Imo` object
            url: UUID or path for the specific Jones HWP parameters.

        Returns:
            An instance of HWPJonesParams populated with raw data.
        """

        data_file = imo.query_data_file(url)
        assert data_file is not None, f"Reference {url} does not point to a data file"
        with imo.open_data_file(data_file) as inpf:
            return HWPJonesParams.from_stream(inpf)

    @classmethod
    def from_file(cls, file_name: str) -> "HWPJonesParams":
        """Load the Jones HWP parameters from a file."""
        with open(file_name) as stream:
            return HWPJonesParams.from_stream(stream)

    def clip_frequencies(
        self, bandcenter_ghz: float, bandwidth_ghz: float
    ) -> "HWPJonesParams":
        """Return a copy of this object with the frequencies outside the range [ν₀ - δν/2, ν + δν/2]
        masked out"""

        assert bandcenter_ghz > 0.0, "The bandcenter must be positive."
        assert bandwidth_ghz > 0.0, "The bandwidth must be positive."

        freq_min = bandcenter_ghz - bandwidth_ghz / 2
        freq_max = bandcenter_ghz + bandwidth_ghz / 2
        mask = (self.freq_ghz >= freq_min) & (self.freq_ghz <= freq_max)

        new_data = {f.name: getattr(self, f.name)[mask] for f in fields(self)}
        return HWPJonesParams(**new_data)
