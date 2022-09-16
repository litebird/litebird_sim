import litebird_sim as lbs
from litebird_sim import BandPassInfo
import unittest
import numpy as np

def test_tophat_bpass():
    test = unittest.TestCase()

    bcenter=140
    bwidth=0.2
    Nsamp=128
    norm= False
    bwidth_ghz = bcenter *bwidth

    B0 = BandPassInfo(bandcenter_ghz= bcenter,bandwidth_ghz=bwidth_ghz ,
                    nsamples_inband=Nsamp ,  name='Top-Hat ' )
    f0,f1 =B0.get_edges()
    bandrange = f0  -  bwidth_ghz  , f1  + bwidth_ghz
    freqs_ghz =np.linspace(bandrange[0], bandrange[1],Nsamp )

    assert np.allclose(B0.freqs_ghz, freqs_ghz )

    weights = np.zeros_like(freqs_ghz)
    weights[np.ma.masked_inside(freqs_ghz, f0, f1).mask] = 1.0
    assert np.allclose(B0.weights, weights )

    test.assertAlmostEqual (B0.get_normalization () ,
            np.trapz(weights, freqs_ghz ))

def test_bpass_apodization():
    test = unittest.TestCase()

    bcenter=140
    bwidth=0.2
    Nsamp=128
    norm= True
    bwidth_ghz = bcenter *bwidth
    Bcos  = BandPassInfo(bandcenter_ghz= bcenter,
                            bandwidth_ghz=bwidth_ghz ,
                            nsamples_inband=Nsamp  ,
                            bandtype='top-hat-cosine', normalize=norm )
    Bexp  = BandPassInfo(bandcenter_ghz= bcenter,
                            bandwidth_ghz=bwidth_ghz ,
                            nsamples_inband=Nsamp  ,
                            bandtype='top-hat-exp', normalize=norm )
    Bcheby  = BandPassInfo(bandcenter_ghz= bcenter,
                            bandwidth_ghz=bwidth_ghz ,
                            nsamples_inband=Nsamp  ,
                            bandtype='cheby', normalize=norm )
    test.assertAlmostEqual (Bcos.get_normalization (),1  )
    test.assertAlmostEqual (Bexp.get_normalization(),1 )
    test.assertAlmostEqual (Bcheby.get_normalization(),1)


def test_bpass_resampling():
    test = unittest.TestCase()

    bcenter=140
    bwidth=0.2
    Nsamp=128
    norm= True
    bwidth_ghz = 40 *bwidth

    Bcos = BandPassInfo(bandcenter_ghz= bcenter,
                        bandwidth_ghz=bwidth_ghz ,
                        nsamples_inband=Nsamp  ,
                        bandtype='top-hat-cosine',
                        normalize=norm, name='cosine band' )
    Bcos._interpolate_band()
    bpass_resampled = Bcos.bandpass_resampling(   nresample=48 ,
                        bstrap_size=5000 , model=Bcos)

    test.assertAlmostEqual (np.trapz(bpass_resampled, Bcos.freqs_ghz ), Bcos.get_normalization() )


test_tophat_bpass()
test_bpass_apodization()
test_bpass_resampling()
