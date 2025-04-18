import litebird_sim as lbs
import numpy as np
from litebird_sim.hwp_sys.hwp_sys import compute_orientation_from_detquat
from litebird_sim import mpi
from litebird_sim.scan_map import scan_map_in_observations


def test_hwp_sys():
    start_time = 0
    time_span_s = 365 * 24 * 3600
    nside = 64
    sampling = 1
    hwp_radpsec = lbs.IdealHWP(
        46 * 2 * np.pi / 60,
    ).ang_speed_radpsec

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s, random_seed=0)

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

    dets = []

    quats = [
        [0.03967584136504414, 0.03725809501267564, 0.0, 0.9985177324254199],
        [
            0.05440050811606006,
            -0.001709604840948807,
            0.706058659733029,
            0.7060586597330291,
        ],
    ]

    for d in range(2):
        det = lbs.DetectorInfo.from_dict(
            {
                "channel": channelinfo,
                "bandcenter_ghz": 140.0,
                "sampling_rate_hz": sampling,
                "quat": quats[d],
            }
        )
        det.phi = 133.2
        dets.append(det)

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
        detectors=dets,
        n_blocks_det=comm.size,
        split_list_over_processes=False,
    )

    for idet in range(obs.n_detectors):
        sim.detectors[idet].pol_angle_rad = compute_orientation_from_detquat(
            obs.quat[idet].quats[0]
        ) % (2 * np.pi)

    sim.prepare_pointings(append_to_report=False)

    Mbsparams = lbs.MbsParameters(
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
        mbs = lbs.Mbs(simulation=sim, parameters=Mbsparams, channel_list=[channelinfo])

        input_maps = mbs.run_all()[0]["L4-140"]
    else:
        input_maps = None

    if mpi.MPI_ENABLED:
        input_maps = comm.bcast(input_maps, root=0)

    hwp_sys = lbs.HwpSys(sim)

    hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        Mbsparams=Mbsparams,
        integrate_in_band=False,
        integrate_in_band_solver=False,
        build_map_on_the_fly=True,
        comm=comm,
    )

    hwp_sys.fill_tod(
        observations=[obs],
        input_map_in_galactic=False,
    )

    output_maps = hwp_sys.make_map([obs])

    np.testing.assert_almost_equal(input_maps, output_maps, decimal=9, verbose=True)

    # testing if code works also when passing list of observations, pointings and hwp_angle to fill_tod

    dets2 = []
    i = 0
    quats = [
        [0.06740000004400000, 0.0256776000009992898, 0.0, 0.987687266626111],
        [
            0.04540050811606006,
            -0.002109604840948807,
            0.809058659733029,
            0.990586597330291,
        ],
    ]

    for i in range(2):
        det = lbs.DetectorInfo.from_dict(
            {
                "channel": channelinfo,
                "bandcenter_ghz": 140.0,
                "sampling_rate_hz": sampling,
                "quat": quats[i],
            }
        )

        det.phi = 45
        dets2.append(det)

    (new_obs,) = sim.create_observations(detectors=dets2)
    for idet in range(new_obs.n_detectors):
        sim.detectors[idet].pol_angle_rad = compute_orientation_from_detquat(
            new_obs.quat[idet].quats[0]
        ) % (2 * np.pi)

    lbs.prepare_pointings(
        observations=[obs, new_obs],
        instrument=sim.instrument,
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        hwp=sim.hwp,
    )

    point_0, hwp_angle_0 = obs.get_pointings("all")
    point_45, hwp_angle_45 = new_obs.get_pointings("all")

    del hwp_sys
    del output_maps
    hwp_sys = lbs.HwpSys(sim)

    hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        Mbsparams=Mbsparams,
        integrate_in_band=False,
        integrate_in_band_solver=False,
        build_map_on_the_fly=True,
        comm=comm,
    )

    hwp_sys.fill_tod(
        observations=[obs, new_obs],
        input_map_in_galactic=False,
        pointings=[point_0, point_45],
        hwp_angle=[hwp_angle_0, hwp_angle_45],
    )

    output_maps = hwp_sys.make_map([obs])
    np.testing.assert_almost_equal(input_maps, output_maps, decimal=9, verbose=True)


def test_hwp_sys_angles():
    # testing if the angles are well defined (the tod computed with hwp_sys and the one
    # computed with scan_map must be the same)

    start_time = 0
    time_span_s = 1000
    nside = 64
    sampling = 1
    hwp_radpsec = lbs.IdealHWP(
        46 * 2 * np.pi / 60,
    ).ang_speed_radpsec

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s, random_seed=0)

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
            "quat": [0.03967584136504414, 0.03725809501267564, 0.0, 0.9985177324254199],
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

    (obs_scan,) = sim.create_observations(
        detectors=[det],
        n_blocks_det=comm.size,
        split_list_over_processes=False,
    )

    for idet in range(obs_scan.n_detectors):
        sim.detectors[idet].pol_angle_rad = compute_orientation_from_detquat(
            obs_scan.quat[idet].quats[0]
        ) % (2 * np.pi)
        sim.detectors[idet].pointing_theta_phi_psi_deg = [0, 0, 0]

    sim.prepare_pointings(append_to_report=False)

    Mbsparams = lbs.MbsParameters(
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
        mbs = lbs.Mbs(simulation=sim, parameters=Mbsparams, channel_list=[channelinfo])

        input_maps = mbs.run_all()[0]["L4-140"]
    else:
        input_maps = None

    if mpi.MPI_ENABLED:
        input_maps = comm.bcast(input_maps, root=0)

    hwp_sys = lbs.HwpSys(sim)

    hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        Mbsparams=Mbsparams,
        integrate_in_band=False,
        integrate_in_band_solver=False,
        build_map_on_the_fly=True,
        comm=comm,
    )

    obs_hwpsys = obs_scan

    scan_map_in_observations(
        observations=obs_scan,
        maps=input_maps,
    )

    hwp_sys.fill_tod(
        observations=[obs_hwpsys],
        input_map_in_galactic=False,
    )

    np.testing.assert_equal(obs_scan.tod, obs_hwpsys.tod, verbose=True)
