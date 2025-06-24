"""Test GLS mapmaking with BrahMap"""

import tempfile
from unittest import mock

import numpy as np
import pytest

import litebird_sim as lbs

brahmap = pytest.importorskip(
    modname="brahmap", reason="Couldn't import 'brahmap' module"
)
import brahmap  # noqa: E402 F811


def test_import_error():
    msg = "Could not import `BrahMap`. Make sure that the package "
    "`BrahMap` is installed in the same environment "
    "as `litebird_sim`. Refer to "
    "https://anand-avinash.github.io/BrahMap/overview/installation/ "
    "for the installation instruction"
    with mock.patch.dict("sys.modules", {"brahmap": None}):
        with pytest.raises(ImportError, match=msg):
            lbs.mapmaking.make_brahmap_gls_map(
                nside=1, observations=[], pointings_dtype=np.float64
            )


def prepare_sim():
    telescope = "MFT"
    channel = "M1-195"
    detectors = [
        "001_002_030_00A_195_B",
        "001_002_029_45B_195_B",
        "001_002_015_15A_195_T",
        "001_002_047_00A_195_B",
    ]
    start_time = 51
    # mission_time_days = 30
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
        duration_s=70,  # TODO! This breaks the tests
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

    return sim, nside, dtype_float


sim, nside, dtype_float = prepare_sim()


@pytest.mark.parametrize(
    "sim, nside, dtype_float",
    [(sim, nside, dtype_float)],
)
def test_high_level_interface(sim, nside, dtype_float):
    inv_cov = brahmap.LBSim_InvNoiseCovLO_UnCorr(sim.observations)

    gls_params = brahmap.LBSimGLSParameters(solver_type=brahmap.SolverType.IQU)

    gls_results = brahmap.LBSim_compute_GLS_maps(
        nside=nside,
        observations=sim.observations,
        component="tod",
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


@pytest.mark.parametrize(
    "sim, nside, dtype_float",
    [(sim, nside, dtype_float)],
)
def test_low_level_interface(sim, nside, dtype_float):
    inv_cov = brahmap.LBSim_InvNoiseCovLO_UnCorr(sim.observations)

    gls_params = brahmap.LBSimGLSParameters(solver_type=brahmap.SolverType.IQU)

    gls_results = brahmap.LBSim_compute_GLS_maps(
        nside=nside,
        observations=sim.observations,
        component="tod",
        inv_noise_cov_operator=inv_cov,
        dtype_float=dtype_float,
        LBSim_gls_parameters=gls_params,
    )

    lowlev_interface_gls_results = lbs.make_brahmap_gls_map(
        nside=nside,
        observations=sim.observations,
        inv_noise_cov_operator=inv_cov,
        gls_params=gls_params,
    )

    assert np.allclose(
        gls_results.GLS_maps[0],
        lowlev_interface_gls_results.GLS_maps[0],
    )

    assert np.allclose(
        gls_results.GLS_maps[1],
        lowlev_interface_gls_results.GLS_maps[1],
    )

    assert np.allclose(
        gls_results.GLS_maps[2],
        lowlev_interface_gls_results.GLS_maps[2],
    )


@pytest.mark.parametrize(
    "sim, nside, dtype_float",
    [(sim, nside, dtype_float)],
)
def test_circulant_operator(sim: lbs.Simulation, nside, dtype_float):
    covariance = sim.dets_random[0].random(size=sim.observations[0].n_samples)
    power_spec = np.fft.fft(covariance).real

    covariance = np.fft.ifft(power_spec).real
    power_spec = np.fft.fft(covariance).real

    inv_cov_1 = brahmap.LBSim_InvNoiseCovLO_Circulant(
        obs=sim.observations,
        input=covariance,
        input_type="covariance",
    )

    inv_cov_2 = brahmap.LBSim_InvNoiseCovLO_Circulant(
        obs=sim.observations,
        input=power_spec,
        input_type="power_spectrum",
    )

    gls_params = brahmap.LBSimGLSParameters(solver_type=brahmap.SolverType.IQU)

    gls_results_1 = brahmap.LBSim_compute_GLS_maps(
        nside=nside,
        observations=sim.observations,
        component="tod",
        inv_noise_cov_operator=inv_cov_1,
        dtype_float=dtype_float,
        LBSim_gls_parameters=gls_params,
    )

    gls_results_2 = lbs.make_brahmap_gls_map(
        nside=nside,
        observations=sim.observations,
        inv_noise_cov_operator=inv_cov_2,
        gls_params=gls_params,
    )

    assert np.allclose(
        gls_results_1.GLS_maps[0],
        gls_results_2.GLS_maps[0],
    )

    assert np.allclose(
        gls_results_1.GLS_maps[1],
        gls_results_2.GLS_maps[1],
    )

    assert np.allclose(
        gls_results_1.GLS_maps[2],
        gls_results_2.GLS_maps[2],
    )


@pytest.mark.parametrize(
    "sim, nside, dtype_float",
    [(sim, nside, dtype_float)],
)
def test_teoplitz_operator(sim: lbs.Simulation, nside, dtype_float):
    covariance = sim.dets_random[0].random(size=sim.observations[0].n_samples)

    # Making the operator positive definite for faster convergence
    covariance[0] = 10

    extended_covariance = np.concatenate([covariance, covariance[1:-1][::-1]])
    power_spec = np.fft.fft(extended_covariance).real

    inv_cov_1 = brahmap.LBSim_InvNoiseCovLO_Toeplitz(
        obs=sim.observations,
        input=covariance,
        input_type="covariance",
    )

    inv_cov_2 = brahmap.LBSim_InvNoiseCovLO_Toeplitz(
        obs=sim.observations,
        input=power_spec,
        input_type="power_spectrum",
    )

    gls_params = brahmap.LBSimGLSParameters(solver_type=brahmap.SolverType.IQU)

    gls_results_1 = brahmap.LBSim_compute_GLS_maps(
        nside=nside,
        observations=sim.observations,
        component="tod",
        inv_noise_cov_operator=inv_cov_1,
        dtype_float=dtype_float,
        LBSim_gls_parameters=gls_params,
    )

    gls_results_2 = lbs.make_brahmap_gls_map(
        nside=nside,
        observations=sim.observations,
        inv_noise_cov_operator=inv_cov_2,
        gls_params=gls_params,
    )

    assert np.allclose(
        gls_results_1.GLS_maps[0],
        gls_results_2.GLS_maps[0],
    )

    assert np.allclose(
        gls_results_1.GLS_maps[1],
        gls_results_2.GLS_maps[1],
    )

    assert np.allclose(
        gls_results_1.GLS_maps[2],
        gls_results_2.GLS_maps[2],
    )
