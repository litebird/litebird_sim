import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim import mpi
from litebird_sim.hwp_sys.hwp_sys import compute_orientation_from_detquat
from litebird_sim.scan_map import scan_map_in_observations


@pytest.mark.parametrize(
    "interpolation,nside_out",
    [
        ("", None),
        ("", 32),
        ("linear", None),
    ],
)
def test_hwp_sys(interpolation, nside_out):
    start_time = 0
    time_span_s = 1000
    nside = 64
    sampling = 1
    hwp_radpsec = lbs.IdealHWP(
        46 * 2 * np.pi / 60,
    ).ang_speed_radpsec

    list_of_obs = []
    for i in range(3):
        sim = lbs.Simulation(
            start_time=start_time, duration_s=time_span_s, random_seed=0
        )

        sim.set_hwp(lbs.IdealHWP(hwp_radpsec))

        comm = sim.mpi_comm
        rank = comm.rank

        channelinfo = lbs.FreqChannelInfo(
            bandcenter_ghz=140.0,
            channel="L4-140",
            bandwidth_ghz=42.0,
            net_detector_ukrts=38.44,
            net_channel_ukrts=3.581435543962163,
            pol_sensitivity_channel_ukarcmin=7.24525963532118,
        )

        det = lbs.DetectorInfo.from_dict(
            {
                "channel": channelinfo,
                "bandcenter_ghz": 140.0,
                "sampling_rate_hz": sampling,
                "quat": [
                    0.03967584136504414,
                    0.03725809501267564,
                    0.0,
                    0.9985177324254199,
                ],
                "pointing_theta_phi_psi_deg": [0, 0, 0],
            }
        )

        scan_strat = lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            precession_rate_hz=1.0 / (60.0 * 192.348),
            spin_rate_hz=0.05 / 60.0,
        )

        sim.set_scanning_strategy(append_to_report=False, scanning_strategy=scan_strat)

        instr = lbs.InstrumentInfo(
            name="LFT",
            boresight_rotangle_rad=0.0,
            spin_boresight_angle_rad=0.8726646259971648,
            spin_rotangle_rad=0.0,
            hwp_rpm=46.0,
            number_of_channels=1,
        )

        sim.set_instrument(instr)

        (obs,) = sim.create_observations(
            detectors=[det],
            n_blocks_det=comm.size,
            split_list_over_processes=False,
        )

        for idet in range(obs.n_detectors):
            sim.detectors[idet].pol_angle_rad = compute_orientation_from_detquat(
                obs.quat[idet].quats[0]
            ) % (2 * np.pi)

        sim.prepare_pointings(append_to_report=False)

        mbs_params = lbs.MbsParameters(
            make_cmb=True,
            seed_cmb=1234,
            make_noise=False,
            make_dipole=True,
            make_fg=True,
            fg_models=["pysm_synch_0", "pysm_dust_0", "pysm_freefree_1"],
            gaussian_smooth=True,
            bandpass_int=False,
            maps_in_ecliptic=True,
            nside=nside,
            units="K_CMB",
        )

        if rank == 0:
            mbs = lbs.Mbs(
                simulation=sim, parameters=mbs_params, channel_list=[channelinfo]
            )

            input_maps = mbs.run_all()[0]["L4-140"]
        else:
            input_maps = None

        if mpi.MPI_ENABLED:
            input_maps = comm.bcast(input_maps, root=0)

        hwp_sys = lbs.HwpSys(sim)

        hwp_sys.set_parameters(
            nside=nside,
            nside_out=nside_out,
            maps=input_maps,
            channel=channelinfo,
            interpolation=interpolation,
            mbs_params=mbs_params,
            build_map_on_the_fly=True,
            comm=comm,
        )

        list_of_obs.append(obs)

    # we have two similar observations, now we will compute the TOD
    # using both scan_map_in_observations and hwp_sys.fill_tod
    # and check that they are the same

    scan_map_in_observations(
        observations=list_of_obs[0],
        input_map_in_galactic=False,
        maps=input_maps,
        interpolation=interpolation,
    )

    hwp_sys.fill_tod(
        observations=list_of_obs[1],
        input_map_in_galactic=False,
        save_tod=True,
    )

    # the call below is equivalent to
    # hwp_sys.fill_tod
    scan_map_in_observations(
        observations=list_of_obs[2],
        input_map_in_galactic=False,
        hwp_type="non_ideal_harmonics",
        hwp_harmonics=hwp_sys,
        maps=input_maps,
        interpolation=interpolation,
    )

    # Check that we are using 64-bit floating-point numbers for pointings. See
    # https://github.com/litebird/litebird_sim/pull/429
    pointings, _ = list_of_obs[1].get_pointings()
    assert pointings.dtype == np.float64

    # The decimal=3 in here has a reason, explained in PR 395.
    # This should be changed in the future
    np.testing.assert_almost_equal(
        list_of_obs[0].tod, list_of_obs[1].tod, decimal=3, verbose=True
    )

    # testing using scan map with "non_ideal_harmonics" parameter instead of
    # hwp_sys.fill_tod directly
    np.testing.assert_almost_equal(
        list_of_obs[1].tod, list_of_obs[2].tod, decimal=3, verbose=True
    )
