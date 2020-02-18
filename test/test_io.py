# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs


def test_write_healpix_map_to_hdu():
    for cur_dtype in (
        np.float32,
        np.float64,
        np.uint8,
        np.int16,
        np.int32,
        np.int64,
        np.bool,
    ):
        pixels = np.zeros(12, dtype=cur_dtype)
        curname = str(cur_dtype)
        hdu = lbs.write_healpix_map_to_hdu(pixels, dtype=cur_dtype, name=curname)

        assert hdu.header["EXTNAME"] == curname
        assert len(hdu.data.field(0)) == len(pixels)
        assert hdu.data.field(0).dtype == cur_dtype


def test_write_healpix_map(tmp_path):
    for cur_dtype in (
        np.float32,
        np.float64,
        np.uint8,
        np.int16,
        np.int32,
        np.int64,
        np.bool,
    ):
        pixels = np.zeros(12, dtype=cur_dtype)
        curname = str(cur_dtype)
        filename = tmp_path / f"{str(cur_dtype)}.fits"
        lbs.write_healpix_map_to_file(filename, pixels, dtype=cur_dtype, name=curname)
