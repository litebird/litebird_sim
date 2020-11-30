import litebird_sim as lbs
import healpy as hp
import numpy as np
from pathlib import Path


def test_Mbs():
    curpath = Path(__file__).parent
    sim = lbs.Simulation(parameter_file=curpath / "config_test.toml")
    myinst = {}
    myinst['mock'] = {
        'freq': 140.,
        'freq_band': 42.,
        'beam': 30.8,
        'P_sens': 6.39}
    mbs = lbs.Mbs(sim, instrument=myinst)
    maps = mbs.run_all()
    map_ref = hp.read_map(curpath / "reference_mbs.fits", (0,1,2))
    assert np.allclose(maps['mock'], map_ref, atol=1e-6)
