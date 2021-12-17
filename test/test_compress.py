# -*- encoding: utf-8 -*-

import numpy as np

from litebird_sim.compress import rle_compress, rle_decompress


def test_rle_compression():
    samples = np.array([1, 1, 1, 6, 6, 4, 4, 4, 4, 5, 1, 1, 1, 8], dtype="uint8")
    compressed = rle_compress(samples)

    assert np.all(
        compressed
        == np.array(
            [[3, 2, 4, 1, 3, 1], [1, 6, 4, 5, 1, 8]],
            dtype="uint8",
        )
    )

    decompressed = rle_decompress(compressed)

    assert np.all(samples == decompressed)
