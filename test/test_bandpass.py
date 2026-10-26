from litebird_sim import BandPassInfo
import unittest
import numpy as np


def test_tophat_bpass():
    test = unittest.TestCase()

    bcenter = 140
    bwidth = 0.2
    Nsamp = 128
    bwidth_ghz = bcenter * bwidth

    B0 = BandPassInfo(
        bandcenter_ghz=bcenter,
        bandwidth_ghz=bwidth_ghz,
        nsamples_inband=Nsamp,
        name="Top-Hat ",
    )
    f0, f1 = B0.get_edges()
    bandrange = bcenter - bwidth_ghz, bcenter + bwidth_ghz
    freqs_ghz = np.linspace(bandrange[0], bandrange[1], Nsamp)

    assert np.allclose(B0.freqs_ghz, freqs_ghz)

    weights = np.zeros_like(freqs_ghz)
    weights[np.ma.masked_inside(freqs_ghz, f0, f1).mask] = 1.0
    assert np.allclose(B0.weights, weights)

    test.assertAlmostEqual(B0.get_normalization(), np.trapezoid(weights, freqs_ghz))


def test_bpass_apodization():
    test = unittest.TestCase()

    bcenter = 140
    bwidth = 0.2
    Nsamp = 128
    norm = True
    bwidth_ghz = bcenter * bwidth
    Bcos = BandPassInfo(
        bandcenter_ghz=bcenter,
        bandwidth_ghz=bwidth_ghz,
        nsamples_inband=Nsamp,
        bandtype="top-hat-cosine",
        normalize=norm,
    )
    Bexp = BandPassInfo(
        bandcenter_ghz=bcenter,
        bandwidth_ghz=bwidth_ghz,
        nsamples_inband=Nsamp,
        bandtype="top-hat-exp",
        normalize=norm,
    )
    Bcheby = BandPassInfo(
        bandcenter_ghz=bcenter,
        bandwidth_ghz=bwidth_ghz,
        nsamples_inband=Nsamp,
        bandtype="cheby",
        normalize=norm,
    )
    test.assertAlmostEqual(Bcos.get_normalization(), 1)
    test.assertAlmostEqual(Bexp.get_normalization(), 1)
    test.assertAlmostEqual(Bcheby.get_normalization(), 1)


def test_bpass_cosine_tails():
    # The tails start at the edges and are bandwidth / cosine_apo_length long
    B = BandPassInfo(
        bandcenter_ghz=140.0,
        bandwidth_ghz=42.0,
        nsamples_inband=1000,
        bandtype="top-hat-cosine",
    )
    apolength = B.bandwidth_ghz / B.cosine_apo_length
    f, w = B.freqs_ghz, B.weights

    assert np.all(w[(f < B.f0 - apolength) | (f > B.f1 + apolength)] == 0.0)
    assert np.all(w[(f >= B.f0) & (f <= B.f1)] == 1.0)

    tail = (f > B.f1) & (f < B.f1 + apolength)
    assert np.allclose(w[tail], (1 + np.cos((f[tail] - B.f1) / apolength * np.pi)) / 2)


def test_bpass_resampling():
    test = unittest.TestCase()

    bcenter = 140
    bwidth = 0.2
    Nsamp = 128
    norm = True
    bwidth_ghz = 40 * bwidth

    Bcos = BandPassInfo(
        bandcenter_ghz=bcenter,
        bandwidth_ghz=bwidth_ghz,
        nsamples_inband=Nsamp,
        bandtype="top-hat-cosine",
        normalize=norm,
        name="cosine band",
    )
    Bcos._interpolate_band()
    bpass_resampled = Bcos.bandpass_resampling(
        nresample=48, bstrap_size=5000, model=Bcos.model
    )

    test.assertAlmostEqual(
        np.trapezoid(bpass_resampled, Bcos.freqs_ghz), Bcos.get_normalization()
    )


test_tophat_bpass()
test_bpass_apodization()
test_bpass_resampling()


def test_cheby_default_ripple():
    # The default ripple of Chebyshev bandpasses must be the same in
    # `BandPassInfo` and in `bandpass_template_module.bandpass_profile`
    from litebird_sim.bandpass_template_module import bandpass_profile

    band = BandPassInfo(bandcenter_ghz=140.0, bandwidth_ghz=42.0, bandtype="cheby")
    _, profile = bandpass_profile(
        freqs=band.freqs_ghz,
        bandpass={
            "band_type": "cheby",
            "bandcenter_ghz": 140.0,
            "band_low_edge": 119.0,
            "band_high_edge": 161.0,
        },
    )
    np.testing.assert_allclose(profile, band.weights)
