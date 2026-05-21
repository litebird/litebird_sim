import io
from dataclasses import fields

import numpy as np
import pytest

from litebird_sim import HWPJonesParams

# --- Test Data Setup ---

# We add three frequencies: 1.0 (outside), 2.0 (inside), 3.0 (outside)
# for a band centered at 2.0 with a width of 1.0.
CSV_CONTENT = """freq,Jxx_0f,Phxx_0f,Jxy_0f,Phxy_0f,Jyx_0f,Phyx_0f,Jyy_0f,Phyy_0f,Jxx_2f,Phxx_2f,Jxy_2f,Phxy_2f,Jyx_2f,Phyx_2f,Jyy_2f,Phyy_2f
1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0
2.0,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1,13.1,14.1,15.1,16.1,17.1
3.0,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,16.2,17.2
"""


# --- Pytest Suite ---


class TestHWPJonesParams:
    """Suite of tests for HWPJonesParams loading and manipulation."""

    @pytest.fixture
    def sample_params(self) -> HWPJonesParams:
        """Fixture to provide a populated HWPJonesParams instance."""
        stream = io.StringIO(CSV_CONTENT)
        return HWPJonesParams.from_stream(stream)

    def test_from_stream_loading(self, sample_params):
        """Verify that data is correctly parsed into the dataclass."""
        assert len(sample_params.freq_ghz) == 3
        assert sample_params.freq_ghz[0] == 1.0
        assert sample_params.Jxx_0f[1] == 2.1
        assert sample_params.Phyy_2f[2] == 17.2

    def test_clip_frequencies_logic(self, sample_params):
        """
        Verify that clipping correctly filters data within [ν₀ - δν/2, ν₀ + δν/2].
        For ν₀=2.0 and δν=1.0, range is [1.5, 2.5]. Only the 2.0 GHz row should remain.
        """
        clipped = sample_params.clip_frequencies(bandcenter_ghz=2.0, bandwidth_ghz=1.0)

        assert len(clipped.freq_ghz) == 1
        assert clipped.freq_ghz[0] == 2.0
        assert clipped.Jxx_0f[0] == 2.1
        assert clipped.Phyy_2f[0] == 17.1

    def test_clip_frequencies_immutability(self, sample_params):
        """
        Ensure that the original object is not modified by the clipping operation.
        """
        original_freqs = sample_params.freq_ghz.copy()

        # Perform clipping
        _ = sample_params.clip_frequencies(bandcenter_ghz=2.0, bandwidth_ghz=1.0)

        # Verify the original object is identical to its state before clipping
        assert len(sample_params.freq_ghz) == 3
        np.testing.assert_array_equal(sample_params.freq_ghz, original_freqs)
        assert sample_params.freq_ghz[0] == 1.0

        # Check that it is a different object instance
        clipped = sample_params.clip_frequencies(2.0, 1.0)
        assert clipped is not sample_params

    def test_clip_frequencies_validation(self, sample_params):
        """Verify that invalid frequency parameters raise appropriate errors."""
        with pytest.raises(AssertionError, match="must be positive"):
            sample_params.clip_frequencies(bandcenter_ghz=-10, bandwidth_ghz=1.0)

        with pytest.raises(AssertionError, match="must be positive"):
            sample_params.clip_frequencies(bandcenter_ghz=100, bandwidth_ghz=0.0)

    def test_all_fields_clipped(self, sample_params):
        """Ensure all fields in the dataclass are masked, not just the frequency."""
        clipped = sample_params.clip_frequencies(2.0, 1.0)

        for field in fields(HWPJonesParams):
            val = getattr(clipped, field.name)
            assert len(val) == 1, f"Field {field.name} was not clipped correctly."
