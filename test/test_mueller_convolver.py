import ducc0
import numpy as np
import pytest
import healpy
from litebird_sim import MuellerConvolver


pmp = pytest.mark.parametrize


def nalm(lmax, mmax):
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def random_alm(lmax, mmax, ncomp, rng):
    res = rng.uniform(-1.0, 1.0, (ncomp, nalm(lmax, mmax))) + 1j * rng.uniform(
        -1.0, 1.0, (ncomp, nalm(lmax, mmax))
    )
    # make a_lm with m==0 real-valued
    res[:, 0 : lmax + 1].imag = 0.0
    return res


# This compares Mueller matrices which are simply scaled identity matrices
# against the standard convolver
@pmp("lkmax", [(13, 9), (5, 1), (30, 15), (35, 2), (58, 0)])
@pmp("ncomp", [1, 3, 4])
@pmp("fct", [1.0, -1.0, np.pi])
def test_trivial_mueller_matrix(fct, lkmax, ncomp):
    rng = np.random.default_rng(41)

    lmax, kmax = lkmax
    nptg = 100
    epsilon = 1e-4
    ofactor = 1.5
    nthreads = 0

    slm = random_alm(lmax, lmax, ncomp, rng)
    blm = random_alm(lmax, kmax, ncomp, rng)

    ptg = np.empty((nptg, 3))
    ptg[:, 0] = rng.random(nptg) * np.pi
    ptg[:, 1] = rng.random(nptg) * 2 * np.pi
    ptg[:, 2] = rng.random(nptg) * 2 * np.pi
    alpha = rng.random(nptg) * 2 * np.pi

    mueller = np.identity(4) * fct

    fullconv = MuellerConvolver(
        lmax,
        kmax,
        slm,
        blm,
        mueller,
        single_precision=False,
        epsilon=epsilon,
        ofactor=ofactor,
        nthreads=nthreads,
    )
    sig = fullconv.signal(ptg, alpha)
    ref_conv = ducc0.totalconvolve.Interpolator(
        slm, blm, False, lmax, kmax, epsilon=epsilon, ofactor=ofactor, nthreads=nthreads
    )
    ref_sig = ref_conv.interpol(ptg)[0] * fct
    np.testing.assert_allclose(sig, ref_sig, atol=1e-3)


# This test case fails for reasons I don't yet understand
# We are using a polarized beam, which is observing a uniform, completely
# unpolarized sky through a rotating polarizer.
# I would expect a signal modulated with exp(i*2*alpha), but the result is
# actually constant. Any insights are welcome!


@pmp("lmax", [100])
def test_polarized(lmax):
    rng = np.random.default_rng(41)

    ncomp = 3
    kmax = 2
    nptg = 100
    epsilon = 1e-4
    ofactor = 1.5
    nthreads = 0

    # completely dark sky
    slm = random_alm(lmax, lmax, ncomp, rng) * 0
    # add uniform unpolarized emission
    slm[0, 0] = 1
    # generate a Gaussian beam using healpy
    blm = healpy.blm_gauss(1.0 * np.pi / 180.0, lmax=lmax, pol=True)

    ptg = np.empty((nptg, 3))
    ptg[:, 0] = 0.5 * np.pi
    ptg[:, 1] = 0.0
    ptg[:, 2] = 0.0
    alpha = rng.random(nptg) * 2 * np.pi

    # Linear polarizer (see last page of https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture17_0.pdf)
    mueller = np.zeros((4, 4))
    mueller[:2, :2] = 1

    fullconv = MuellerConvolver(
        lmax,
        kmax,
        slm,
        blm,
        mueller,
        single_precision=False,
        epsilon=epsilon,
        ofactor=ofactor,
        nthreads=nthreads,
    )
    sig = fullconv.signal(ptg, alpha)

    # I'm testing for near-constness for now, to detect that I'm not getting the
    # result I expect. This has to improve once we have found the bug.
    np.testing.assert_array_less(1e-4, np.max(sig) - np.min(sig))
