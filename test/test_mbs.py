# -*- encoding: utf-8 -*-

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import warnings

import litebird_sim as lbs
import healpy as hp
import numpy as np

PARAMETER_FILE = """
[map_based_sims]

  [map_based_sims.general]
  save = false
  nside = 16
  gaussian_smooth = true
  bandpass_int = true
  parallel_mc = false
  coadd = true

  [map_based_sims.noise]
  make_noise = true
  nmc_noise = 1
  seed_noise = 6437
  N_split = 1

  [map_based_sims.cmb]
  make_cmb = true
  cmb_ps_file = false
  cmb_r = 0
  nmc_cmb = 1
  seed_cmb = 38198

  [map_based_sims.fg]
  make_fg = true
  dust = 'pysm_dust_0'
  synch = 'pysm_synch_0'
  freefree =  'pysm_freefree_1'
  ame =  'pysm_ame_1'

  [map_based_sims.output]
  output_directory = '{output_directory}'
  output_string = 'test'
"""


def test_mbs():
    try:
        # PySM3 produces a lot of warnings when it downloads stuff from Internet
        warnings.filterwarnings("ignore")

        with NamedTemporaryFile(mode="w+t", suffix=".toml") as simfile:
            with TemporaryDirectory() as outdir:
                simfile.write(PARAMETER_FILE.format(output_directory=outdir))
            simfile.flush()

            sim = lbs.Simulation(base_path=outdir, parameter_file=simfile.name)

            myinst = {}
            myinst["mock"] = {
                "bandcenter_ghz": 140.0,
                "bandwidth_ghz": 42.0,
                "fwhm_arcmin": 30.8,
                "p_sens_ukarcmin": 6.39,
            }
            mbs = lbs.Mbs(sim, sim.parameters["map_based_sims"], instrument=myinst)
            (maps, saved_maps) = mbs.run_all()

            curpath = Path(__file__).parent
            map_ref = hp.read_map(curpath / "reference_mbs.fits", (0, 1, 2))
            assert np.allclose(maps["mock"], map_ref, atol=1e-6)
    finally:
        # Reset the standard warnings filter
        warnings.resetwarnings()
