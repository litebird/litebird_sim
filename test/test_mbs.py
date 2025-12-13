# -*- encoding: utf-8 -*-

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import healpy as hp
import numpy as np
import numpy.testing as npt

import litebird_sim as lbs

PARAMETER_FILE = """
[map_based_sims]

  [map_based_sims.general]
  save = false
  nside = 16
  gaussian_smooth = true
  bandpass_int = true
  parallel_mc = false
  coadd = true
  units = "uK_CMB"
  maps_in_ecliptic = false
  lmax_alms = 47

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
    with NamedTemporaryFile(mode="w+t", suffix=".toml") as simfile:
        with TemporaryDirectory() as outdir:
            simfile.write(PARAMETER_FILE.format(output_directory=outdir))
            simfile.flush()

            sim = lbs.Simulation(
                base_path=outdir, parameter_file=simfile.name, random_seed=12345
            )

            myinst = {}
            myinst["mock"] = {
                "bandcenter_ghz": 140.0,
                "bandwidth_ghz": 42.0,
                "fwhm_arcmin": 30.8,
                "p_sens_ukarcmin": 6.39,
                "band": None,
            }

            mbs = lbs.Mbs(sim, sim.parameters["map_based_sims"], instrument=myinst)
            (maps, saved_maps) = mbs.run_all()

            curpath = Path(__file__).parent
            map_ref = hp.read_map(curpath / "reference_mbs.fits", (0, 1, 2))
            npt.assert_allclose(maps["mock"].values, map_ref, atol=1e-3)


def test_map_rotation():
    telescope = "LFT"
    channel = "L4-140"
    detlist = ["000_001_017_QB_140_T"]
    start_time = 51
    mission_time_days = 30
    detector_sampling_freq = 1

    nside = 32
    random_seed = 42
    imo_version = "vPTEP"
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
    dtype_float = np.float64

    sim = lbs.Simulation(
        random_seed=random_seed,
        name="rotation_example",
        start_time=start_time,
        duration_s=mission_time_days * 24 * 60 * 60.0,
        imo=imo,
    )

    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
        )
    )

    detector_list = []
    for n_det in detlist:
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/{imo_version}/satellite/{telescope}/{channel}/{n_det}/detector_info",
            imo=imo,
        )
        det.sampling_rate_hz = detector_sampling_freq
        detector_list.append(det)

    sim.set_scanning_strategy(
        imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/"
    )

    sim.create_observations(
        detectors=detector_list,
        n_blocks_det=1,
        n_blocks_time=1,
    )

    sim.set_hwp(
        lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60),
    )
    sim.prepare_pointings()
    sim.precompute_pointings(pointings_dtype=dtype_float)

    ch_info = []
    n_ch_info = lbs.FreqChannelInfo.from_imo(
        imo,
        f"/releases/{imo_version}/satellite/{telescope}/{channel}/channel_info",
    )
    ch_info.append(n_ch_info)

    mbs_params = lbs.MbsParameters(
        make_cmb=True,  # True to check its rotation
        make_fg=True,  # True to check its rotation
        seed_cmb=1234,
        make_noise=True,  # True to check its rotation
        seed_noise=5678,
        make_dipole=True,  # True to check its rotation
        fg_models=[
            "pysm_synch_0",
            "pysm_freefree_1",
            "pysm_dust_0",
        ],
        gaussian_smooth=True,
        bandpass_int=True,
        nside=nside,
        units="K_CMB",
        maps_in_ecliptic=False,
    )

    mbs = lbs.Mbs(
        simulation=sim,
        parameters=mbs_params,
        channel_list=ch_info,
    )
    maps = mbs.run_all()[0]

    rot_mbs_params = lbs.MbsParameters(
        make_cmb=True,  # True to check its rotation
        make_fg=True,  # True to check its rotation
        seed_cmb=1234,
        make_noise=True,  # True to check its rotation
        seed_noise=5678,
        make_dipole=True,  # True to check its rotation
        fg_models=[
            "pysm_synch_0",
            "pysm_freefree_1",
            "pysm_dust_0",
        ],
        gaussian_smooth=True,
        bandpass_int=True,
        nside=nside,
        units="K_CMB",
        maps_in_ecliptic=True,  # Rotation requested
    )

    rot_mbs = lbs.Mbs(
        simulation=sim,
        parameters=rot_mbs_params,
        channel_list=ch_info,
    )
    rot_maps = rot_mbs.run_all()[0]

    ref_maps = rot_mbs.rotator.rotate_map_alms(
        maps["L4-140"].values, mbs_params.lmax_alms
    )

    np.testing.assert_allclose(ref_maps, rot_maps["L4-140"].values, atol=1e-3)
