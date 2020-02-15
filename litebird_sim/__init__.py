# -*- encoding: utf-8 -*-

from .healpix import (
    nside_to_npix,
    npix_to_nside,
    is_npix_ok,
    map_type,
    get_pixel_format,
    write_healpix_map_to_hdu,
    write_healpix_map_to_file,
)
from .simulations import Simulation

__author__ = "The LiteBIRD simulation team"
__version__ = "0.1.0"
__all__ = [
    "nside_to_npix",
    "npix_to_nside",
    "is_npix_ok",
    "map_type",
    "get_pixel_format",
    "write_healpix_map_to_hdu",
    "write_healpix_map_to_file",
    "Simulation",
]
