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

    @classmethod
    def from_stream(
        cls,
        stream: IO,
        freq_sampling_frac: int = 1,
        freq_sampling_offset: int = 0,
    ) -> "HWPJonesParams":
        """
        Load the Jones HWP parameters from an open text stream (CSV format).

        Expects 17 columns of floating point data as defined in LB-DFS-HWP-001.

        Keeps every `freq_sampling_frac`-th data row, starting at
        `freq_sampling_offset`. The first data row has index 0; headers,
        blank lines, and comments are not counted.
        """
        if not isinstance(freq_sampling_frac, int) or isinstance(
            freq_sampling_frac, bool
        ):
            raise TypeError("freq_sampling_frac must be an integer")
        if freq_sampling_frac <= 0:
            raise ValueError("freq_sampling_frac must be positive")

        if not isinstance(freq_sampling_offset, int) or isinstance(
            freq_sampling_offset, bool
        ):
            raise TypeError("freq_sampling_offset must be an integer")
        if freq_sampling_offset < 0:
            raise ValueError("freq_sampling_offset must be non-negative")

        # Skip the CSV header.
        next(stream, None)

        selected_row_count = 0

        def selected_lines():
            nonlocal selected_row_count
            data_row = 0

            for line in stream:
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                if (
                    data_row >= freq_sampling_offset
                    and (data_row - freq_sampling_offset) % freq_sampling_frac == 0
                ):
                    selected_row_count += 1
                    yield line

                data_row += 1

        data = np.loadtxt(
            selected_lines(),
            delimiter=",",
            dtype=np.float64,
            unpack=True,
            comments="#",
            ndmin=2,
        )

        all_fields = fields(cls)
        expected_columns = len(all_fields)

        if selected_row_count == 0:
            return cls(
                **{field.name: np.empty(0, dtype=np.float64) for field in all_fields}
            )

        if data.shape[0] != expected_columns:
            raise ValueError(
                f"Expected {expected_columns} HWP Jones parameters, got {data.shape[0]}"
            )

        return cls(
            **{field.name: data[index] for index, field in enumerate(all_fields)}
        )

    @classmethod
    def from_file(
        cls,
        file_name: str,
        freq_sampling_frac: int = 1,
        freq_sampling_offset: int = 0,
    ) -> "HWPJonesParams":
        """Load sampled Jones HWP parameters from a CSV file."""
        with open(file_name) as stream:
            return cls.from_stream(
                stream,
                freq_sampling_frac=freq_sampling_frac,
                freq_sampling_offset=freq_sampling_offset,
            )

    @classmethod
    def from_imo(
        cls,
        imo: Imo,
        url: UUID | str,
        freq_sampling_frac: int = 1,
        freq_sampling_offset: int = 0,
    ) -> "HWPJonesParams":
        """Load sampled Jones HWP parameters from the IMo."""
        data_file = imo.query_data_file(url)
        if data_file is None:
            raise ValueError(f"Reference {url} does not point to a data file")

        with imo.open_data_file(data_file) as stream:
            return cls.from_stream(
                stream,
                freq_sampling_frac=freq_sampling_frac,
                freq_sampling_offset=freq_sampling_offset,
            )

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
