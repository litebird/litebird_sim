# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs


def test_simulation(tmp_path):
    sim = lbs.Simulation(base_path=tmp_path / "simulation_dir")
    sim.write_healpix_map(
        filename="test.fits.gz", pixels=np.zeros(12),
    )
