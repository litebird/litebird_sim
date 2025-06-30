"""Test GLS mapmaking with BrahMap"""

import tempfile

import numpy as np
import pytest

import litebird_sim as lbs

brahmap = pytest.importorskip(
    modname="brahmap", reason="Couldn't import 'brahmap' module"
)
import brahmap  # noqa: E402 F811


def test_GLS_mapmaking():
    telescope = "MFT"
    channel = "M1-195"
    detectors = [
        "001_002_030_00A_195_B",
        "001_002_029_45B_195_B",
        "001_002_015_15A_195_T",
        "001_002_047_00A_195_B",
    ]
    start_time = 51
    mission_time_days = 30
    detector_sampling_freq = 1

    nside = 128
    random_seed = 45
    imo_version = "vPTEP"
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
    dtype_float = np.float64
    tmp_dir = tempfile.TemporaryDirectory()

    sim = lbs.Simulation(
        random_seed=random_seed,
        base_path=tmp_dir.name,
        name="brahmap_example",
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
    for n_det in detectors:
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
        num_of_obs_per_detector=3,
        n_blocks_det=1,
        n_blocks_time=1,
        split_list_over_processes=False,
        tod_dtype=dtype_float,
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
        make_cmb=True,
        make_fg=False,
        seed_cmb=1,
        gaussian_smooth=True,
        bandpass_int=False,
        nside=nside,
        units="uK_CMB",
        maps_in_ecliptic=False,
        output_string="mbs_cmb_lens",
    )

    mbs_obj = lbs.Mbs(
        simulation=sim,
        parameters=mbs_params,
        channel_list=ch_info,
    )

    input_maps = mbs_obj.run_all()

    lbs.scan_map_in_observations(
        sim.observations,
        maps=input_maps[0][channel],
    )

    inv_cov = brahmap.LBSim_InvNoiseCovLO_UnCorr(sim.observations)

    gls_params = brahmap.LBSimGLSParameters(solver_type=brahmap.SolverType.IQU)

    gls_results = brahmap.LBSim_compute_GLS_maps(
        nside=nside,
        observations=sim.observations,
        components="tod",
        inv_noise_cov_operator=inv_cov,
        dtype_float=dtype_float,
        LBSim_gls_parameters=gls_params,
    )

    interface_gls_results = sim.make_brahmap_gls_map(nside=nside)

    assert np.allclose(
        gls_results.GLS_maps[0],
        interface_gls_results.GLS_maps[0],
    )

    assert np.allclose(
        gls_results.GLS_maps[1],
        interface_gls_results.GLS_maps[1],
    )

    assert np.allclose(
        gls_results.GLS_maps[2],
        interface_gls_results.GLS_maps[2],
    )
