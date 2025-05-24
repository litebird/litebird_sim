# -*- encoding: utf-8 -*-

# WARNING: this file is a placeholder, meant to be used while we find
# a way to make existing solutions (healpy, pyHealpix, etc.) run under
# Windows. It is mostly a bunch of stuff copied-and-pasted from other
# projects, and it does not represent the standards of other codes in
# the litebird_sim framework!

import numpy as np
from astropy.io import fits

STANDARD_COLUMN_NAMES = {
    1: "TEMPERATURE",
    2: ["Q_POLARISATION", "U_POLARISATION"],
    3: ["TEMPERATURE", "Q_POLARISATION", "U_POLARISATION"],
    6: ["II", "IQ", "IU", "QQ", "QU", "UU"],
}


def nside_to_npix(nside):
    """Return the number of pixels in a Healpix map with the specified NSIDE.

    If the value of `nside` is not valid (power of two), an
    `AssertionError` exception is raised.

    .. doctest::

        >>> nside_to_npix(1)
        12

    """
    assert 2 ** np.log2(nside) == nside, f"Invalid value for NSIDE: {nside}"
    return 12 * nside * nside


def npix_to_nside(num_of_pixels):
    """Return NSIDE for a Healpix map containing `num_of_pixels` pixels.

    If the number of pixels does not conform to the Healpix standard,
    an `AssertionError` exception is raised.

    .. doctest::

        >>> npix_to_nside(48)
        2

    """

    assert is_npix_ok(num_of_pixels), f"Invalid number of pixels: {num_of_pixels}"
    return int(np.sqrt(num_of_pixels / 12))


def is_npix_ok(num_of_pixels):
    """Return True or False whenever num_of_pixels is a valid number.

    The function checks if the number of pixels provided as an
    argument conforms to the Healpix standard, which means that the
    number must be in the form 12NSIDE^2.

    .. doctest::

        >>> is_npix_ok(48)
        True
        >>> is_npix_ok(49)
        False

    """
    nside = np.sqrt(np.asarray(num_of_pixels) / 12.0)
    return nside == np.floor(nside)


def map_type(pixels):
    """Check the type of an Healpix map.

    This function returns an integer number that classifies the kind
    of map passed as the parameter:

    - -1: the argument is not a valid map
    - 0: the argument is a single map
    - `n > 0`: the argument is a sequence of `n` maps.

    .. doctest::

       >>> map_type(np.zeros(12))
       0
       >>> map_type([np.zeros(12)])
       1
       >>> map_type([np.zeros(12), np.zeros(12)])
       2
       >>> map_type(np.zeros(11))
       -1
    """

    try:
        npix = len(pixels[0])
    except TypeError:
        npix = None

    if npix is not None:
        for p in pixels[1:]:
            if len(p) != npix:
                return -1
        if is_npix_ok(len(pixels[0])):
            return len(pixels)
        else:
            return -1
    else:
        if is_npix_ok(len(pixels)):
            return 0
        else:
            return -1


def get_pixel_format(t):
    """Get the FITSIO format string for data type t.

    This function returns a string containing the value to be used
    with the `format` keyword in a `astropy.io.fits.Column`
    constructor.

    If the data type cannot be represented exactly in a FITS column, a
    `ValueError` exception is thrown. Beware that this happens even
    with some well-defined numerical types provided by NumPy, like
    `np.int8`, `np.uint16`, `np.uint32`, `np.uint64`. This differs
    from the convention used by HealPy, which returns `None` in this
    case.

    .. doctest::

        >>> import numpy
        >>> get_pixel_format(numpy.uint8)
        'B'

    """
    conv = {
        np.dtype(bool): "L",
        np.dtype(np.uint8): "B",
        np.dtype(np.int16): "I",
        np.dtype(np.int32): "J",
        np.dtype(np.int64): "K",
        np.dtype(np.float32): "E",
        np.dtype(np.float64): "D",
        np.dtype(np.complex64): "C",
        np.dtype(np.complex128): "M",
    }
    try:
        if t in conv:
            return conv[t]
    except Exception:
        pass
    try:
        if np.dtype(t) in conv:
            return conv[np.dtype(t)]
    except Exception:
        pass
    try:
        if np.dtype(type(t)) in conv:
            return conv[np.dtype(type(t))]
    except Exception:
        pass
    try:
        if np.dtype(type(t[0])) in conv:
            return conv[np.dtype(type(t[0]))]
    except Exception:
        pass
    try:
        if t is str:
            return "A"
    except Exception:
        pass
    try:
        if isinstance(t, str):
            return "A%d" % (len(t))
    except Exception:
        pass
    try:
        if isinstance(t[0], str):
            length = max(len(s) for s in t)
            return "A%d" % (length)
    except Exception:
        pass

    raise ValueError(f"Unable to convert type {t} into a CFITSIO data type")


# This is a simplified version of `healpy.save_map`, with more
# sensible defaults (e.g., overwrite is True by default) and a few
# functionalities dropped (e.g., partial maps). This permits to drop
# an explicit dependency on healpy here
def write_healpix_map_to_hdu(
    pixels,
    nest=False,
    dtype=None,
    coord=None,
    column_names=None,
    column_units=None,
    name=None,
    extra_header=(),
) -> fits.BinTableHDU:
    """Write a Healpix map into a FITS HDU.

    This function is a stripped-down implementation of
    healpy.write_map, which discards some not-widely-used keywords and
    adopts saner conventions. It does *not* write the Healpix map into
    a file, but it constructs a FITS HDU; in this way, you can save
    the map alongside other tables or maps in the same FITS file.

    Parameters:

    :param pixels: Array containing the pixels, or list of arrays if
    you want to save several maps into the same FITS table (e.g., I,
    Q, U components)

    :param nest: A Boolean value specifying if the ordering of the
    pixels follows the NEST or RING convention. This only affects the
    value of the ``ORDERING`` keyword in the HDU header.

    :param dtype: The NumPy data type to be used when saving data in
    the HDU. You can pass `numpy.float32` here if you want to save
    your array as 32-bit floating point numbers, regardless of the
    precision used by `pixels`.

    :param coord: A string identifying the coordinate system used for
    the map.

    :param column_names: A string or a list of strings specifying the
    name for each column in the FITS file. Each column is associated
    to a map in `pixels`, so if `pixels` contains *one* map, this
    parameter should be a string; otherwise, it should be a list of
    `N` strings, where `N` is the number of arrays in `pixels` (e.g.,
    3 for an IQU map).

    :param column_units: A string or a list of strings containing the
    measurement units for each column. The way this parameter is used
    is similar to `column_names`.

    :param name: A string to be assigned to the ``EXTNAME`` field in
    the FITS HDU header. This parameter is useful if you plan to save
    many HDUs in the same FITS file.

    :param extra_header: A list of pairs of the form ``(NAME, VALUE)``
    or ``(NAME, VALUE, COMMENT)``. These pairs are used to save
    additional metadata in the header of the FITS HDU.

    Here is an example::

        import numpy
        hdu = write_healpix_map_to_hdu(
           [
               numpy.zeros(12),   # I map
               numpy.zeros(12),   # Q map
               numpy.zeros(12),   # U map
           ],
           column_names=["I", "Q", "U"],
           column_units=["K", "mK", "mK"],
           dtype=[np.float64, np.float32, np.float32],
           name="MYMAP")
    """

    if len(pixels.shape) == 2:
        # "pixels" is a 2D matrix, convert it into a list of 1D matrices
        pixels = [pixels[i, :] for i in range(pixels.shape[0])]

    if not hasattr(pixels, "__len__"):
        raise TypeError("The map must be a sequence")

    try:
        pixels = pixels.filled()
    except AttributeError:
        try:
            pixels = np.array([p.filled() for p in pixels])
        except AttributeError:
            pass

    this_map_type = map_type(pixels)
    assert this_map_type != -1, "Invalid map"
    if this_map_type == 0:  # a single map is converted to a list
        pixels = [pixels]

    # check the dtype and convert it
    if dtype is None:
        dtype = [x.dtype for x in pixels]
    try:
        fitsformat = []
        for curr_dtype in dtype:
            fitsformat.append(get_pixel_format(curr_dtype))
    except TypeError:
        # dtype is not iterable
        fitsformat = [get_pixel_format(dtype)] * len(pixels)

    if column_names is None:
        column_names = STANDARD_COLUMN_NAMES.get(
            len(pixels), ["COLUMN_%d" % n for n in range(1, len(pixels) + 1)]
        )
    else:
        assert len(column_names) == len(pixels), (
            f"{len(column_names)=} != {len(pixels)=}"
        )

    if column_units is None or isinstance(column_units, str):
        column_units = [column_units] * len(pixels)

    assert len(set(map(len, pixels))) == 1, "Maps must have the same length"
    nside = npix_to_nside(len(pixels[0]))

    if nside < 0:
        raise ValueError("Invalid healpix map : wrong number of pixel")

    cols = []
    for cn, cu, mm, curr_fitsformat in zip(
        column_names, column_units, pixels, fitsformat
    ):
        cols.append(
            fits.Column(name=cn, format=str(curr_fitsformat), array=mm, unit=cu)
        )

    tbhdu = fits.BinTableHDU.from_columns(cols)
    # add needed keywords
    tbhdu.header["PIXTYPE"] = ("HEALPIX", "HEALPIX pixelisation")
    if nest:
        ordering = "NESTED"
    else:
        ordering = "RING"
    tbhdu.header["ORDERING"] = (
        ordering,
        "Pixel ordering scheme, either RING or NESTED",
    )
    if coord:
        tbhdu.header["COORDSYS"] = (
            coord,
            "Ecliptic, Galactic or Celestial (equatorial)",
        )
    tbhdu.header["EXTNAME"] = (
        "xtension" if not name else name,
        "name of this binary table extension",
    )
    tbhdu.header["NSIDE"] = (nside, "Resolution parameter of HEALPIX")
    tbhdu.header["FIRSTPIX"] = (0, "First pixel # (0 based)")
    tbhdu.header["LASTPIX"] = (nside_to_npix(nside) - 1, "Last pixel # (0 based)")
    tbhdu.header["INDXSCHM"] = ("IMPLICIT", "Indexing: IMPLICIT or EXPLICIT")
    tbhdu.header["OBJECT"] = ("FULLSKY", "Sky coverage, either FULLSKY or PARTIAL")

    for args in extra_header:
        tbhdu.header[args[0]] = args[1:]

    return tbhdu


def write_healpix_map_to_file(
    filename,
    pixels,
    nest=False,
    dtype=None,
    coord=None,
    column_names=None,
    column_units=None,
    name=None,
    extra_header=(),
    overwrite=True,
):
    """This method is a wrapper for :proc:`write_healpix_map_to_hdu`,
    saving the HDU in a FITS file whose name is `filename`.

    The only extra keyword is `overwrite` (default is ``True``), which
    tells the procedure whether an existing file with the same name
    should be overwritten or not.


    .. doctest::

        import numpy
        write_healpix_map_to_file(
           filename="out.fits.gz",
           pixels=[
               numpy.zeros(12),   # I map
               numpy.zeros(12),   # Q map
               numpy.zeros(12),   # U map
           ],
           column_names=["I", "Q", "U"],
           column_units=["K", "mK", "mK"],
           dtype=[np.float64, np.float32, np.float32],
           name="MYMAP")
    """
    hdu = write_healpix_map_to_hdu(
        pixels=pixels,
        nest=nest,
        dtype=dtype,
        coord=coord,
        column_names=column_names,
        column_units=column_units,
        name=name,
        extra_header=extra_header,
    )
    hdu.writeto(filename, overwrite=overwrite)
