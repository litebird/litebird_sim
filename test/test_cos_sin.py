import numpy as np
from numba import njit

from litebird_sim.utilities import cos_sin


@njit
def _cos_sin_array(x):
    c = np.empty_like(x)
    s = np.empty_like(x)
    for i in range(x.size):
        c[i], s[i] = cos_sin(x[i])
    return c, s


def test_cos_sin():
    x = np.concatenate(
        [
            np.linspace(-20.0, 20.0, 100_001),
            [0.0, np.pi, 2 * np.pi, -2 * np.pi, np.nextafter(2 * np.pi, 0), -1e-300],
        ]
    )
    c, s = _cos_sin_array(x)
    np.testing.assert_allclose(c, np.cos(x), rtol=0, atol=2e-15)
    np.testing.assert_allclose(s, np.sin(x), rtol=0, atol=2e-15)

    # large angles (e.g. 2ψ + 4·HWP angle over a long run): error from the
    # range reduction grows as |x|·eps
    x = np.linspace(-1e4, 1e4, 100_001)
    c, s = _cos_sin_array(x)
    np.testing.assert_allclose(c, np.cos(x), rtol=0, atol=1e-11)
    np.testing.assert_allclose(s, np.sin(x), rtol=0, atol=1e-11)
