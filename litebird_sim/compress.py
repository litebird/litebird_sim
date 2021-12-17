# -*- encoding: utf-8 -*-

import numpy as np
from numba import njit


def rle_compress(arr):
    """Perform a Run-Length encoding of the input array

    Returns a NumPy matrix of shape 2×N, where N is an integer; the first
    row contains the *lengths* of each run, and the second row contains
    the *values*."""

    # This code was adapted from https://stackoverflow.com/a/32681075/3967151

    ia = np.asarray(arr)  # force numpy
    nsamples = len(ia)
    if nsamples == 0:
        return None, None
    else:
        y = ia[1:] != ia[:-1]
        indexes = np.append(np.where(y), nsamples - 1)
        runs = np.diff(np.append(-1, indexes))
        return np.array([runs, ia[indexes]])


@njit
def _rle_decompress(compressed_arr, result):
    out_idx = 0
    for in_idx in range(compressed_arr.shape[1]):
        length = compressed_arr[0, in_idx]
        val = compressed_arr[1, in_idx]
        for i in range(length):
            result[out_idx] = val
            out_idx += 1


def rle_decompress(compressed_arr):
    result = np.empty(np.sum(compressed_arr[0, :]))
    _rle_decompress(compressed_arr, result)
    return result
