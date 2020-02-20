# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs

NUMPY_TYPES = [
    (np.float32, "float32"),
    (np.float64, "float64"),
    (np.uint8, "uint8"),
    (np.int16, "int16"),
    (np.int32, "int32"),
    (np.int64, "int64"),
    (np.bool, "bool"),
]


def test_write_healpix_map_to_hdu():
    for cur_dtype, cur_name in NUMPY_TYPES:
        pixels = np.zeros(12, dtype=cur_dtype)
        hdu = lbs.write_healpix_map_to_hdu(pixels, dtype=cur_dtype, name=cur_name)

        assert hdu.header["EXTNAME"] == cur_name
        assert len(hdu.data.field(0)) == len(pixels)
        assert hdu.data.field(0).dtype == cur_dtype


def test_write_healpix_map(tmp_path):
    for cur_dtype, cur_name in NUMPY_TYPES:
        pixels = np.zeros(12, dtype=cur_dtype)
        filename = tmp_path / f"{cur_name}.fits"
        lbs.write_healpix_map_to_file(filename, pixels, dtype=cur_dtype, name=cur_name)
