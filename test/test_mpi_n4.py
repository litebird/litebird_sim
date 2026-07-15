import numpy as np

import litebird_sim as lbs
from litebird_sim import mpi


def test_hwp_sys_mpi():
    start_time = 0
    time_span_s = 5
    nside = 128
    sampling = 2
    hwp_radpsec = lbs.IdealHWP(
        46 * 2 * np.pi / 60,
    ).ang_speed_radpsec
    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s, random_seed=0)

    comm = sim.mpi_comm

    channelinfo = lbs.FreqChannelInfo(
        bandcenter_ghz=140.0,
        channel="MF1_140",
        bandwidth_ghz=42.0,
        net_detector_ukrts=38.44,
        net_channel_ukrts=3.581435543962163,
        pol_sensitivity_channel_ukarcmin=7.24525963532118,
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

    dets = []

    random_angles = [[40, 30, 45], [45, 34, 135], [33, 5, 45], [48, 23, 135]]
    random_quats = [
        [0.2991869, 0.63963454, 0.51590709, 0.48496878],
        [0.36879414, 0.67473222, 0.51070988, 0.38458125],
        [0.36879414, 0.67473222, 0.51070988, 0.38458125],
        [0.26470446, 0.44080048, 0.47613514, 0.71338756],
    ]

    for d in range(4):
        det = lbs.DetectorInfo.from_dict(
            {
                "channel": channelinfo,
                "bandcenter_ghz": 140.0,
                "sampling_rate_hz": sampling,
                "quat": random_quats[d],
                "pointing_theta_phi_psi_deg": random_angles[d],
            }
        )

        det.mueller_hwp = {
            "0f": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
            "2f": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
            "4f": np.array(
                [
                    [0, 0, 0],
                    [3e-5, 1, 1],
                    [3e-5, 1, 1],
                ],
                dtype=np.float64,
            ),
        }

        sim.set_hwp(
            lbs.NonIdealHWP(
                ang_speed_radpsec=hwp_radpsec,
                harmonic_expansion=True,
                calculus=lbs.HWPFormalism.MUELLER,
            )
        )

        det.sampling_rate_hz = sampling

        dets.append(det)

    mueller_phases = {
        "2f": np.array(
            [[-2.32, -0.49, -2.06], [2.86, -0.25, -2.00], [1.29, -2.01, 2.54]],
            dtype=np.float64,
        ),
        "4f": np.array(
            [
                [-0.84, -0.04, -1.61],
                [0.14, -0.00061, -0.00056 - np.pi / 2],
                [-1.43, -0.00070 - np.pi / 2, np.pi - 0.00065],
            ],
            dtype=np.float64,
        ),
    }

    if comm.size == 1:
        n_blocks_det = 1
        n_blocks_time = 1
    else:
        n_blocks_det = 2
        n_blocks_time = 2

    (obs,) = sim.create_observations(
        detectors=dets,
        n_blocks_det=n_blocks_det,
        n_blocks_time=n_blocks_time,
        split_list_over_processes=False,
    )

    sim.prepare_pointings(append_to_report=False)

    sky_params = lbs.SkyGenerationParams(
        make_cmb=True,
        seed_cmb=1234,
        make_dipole=True,
        make_fg=True,
        fg_models=["s0", "d0", "f1"],
        apply_beam=True,
        bandpass_integration=False,
        nside=nside,
        units="K_CMB",
    )

    if comm.rank == 0:
        gen_sky = lbs.SkyGenerator(parameters=sky_params, channels=channelinfo)

        input_maps = gen_sky.execute()["MF1_140"]

    else:
        input_maps = None

    if mpi.MPI_ENABLED:
        input_maps = comm.bcast(input_maps, root=0)

    lbs.scan_map_in_observations(
        maps=input_maps,
        observations=[obs],
        save_tod=True,
        mueller_phases=mueller_phases,
    )

    # tod calculated for the same simulation using only 1 cpu
    expected_tod = np.array(
        [
            [
                0.00147525,
                0.00143843,
                0.00143786,
                0.00143678,
                0.00143947,
                0.00147244,
                0.00147919,
                0.00147183,
                0.00147948,
                0.001614,
            ],
            [
                0.00221989,
                0.00222688,
                0.00210987,
                0.00211222,
                0.0021095,
                0.00211247,
                0.00210937,
                0.00225522,
                0.00225325,
                0.00225616,
            ],
            [
                0.00221993,
                0.00222686,
                0.00210987,
                0.00211222,
                0.00210948,
                0.0021125,
                0.00210933,
                0.00225527,
                0.00225319,
                0.00225623,
            ],
            [
                -0.00037446,
                -0.00037189,
                -0.0003756,
                -0.00038376,
                -0.00038003,
                -0.0003829,
                -0.00030053,
                -0.00039171,
                -0.00038854,
                -0.00039345,
            ],
        ]
    )

    if comm.size == 4:
        tmp_arr = None
        if comm.rank == 0:
            tmp_arr = np.empty([comm.size, 2, len(obs.tod[0])], dtype="f")

        comm.Gather(obs.tod, tmp_arr, root=0)

        if comm.rank == 0:
            final_tod = np.empty([len(dets), time_span_s * sampling], dtype="f")

            # manually reorganize the gathered tod into the final tod array
            final_tod[0, :5] = tmp_arr[0, 0, :]
            final_tod[0, 5:] = tmp_arr[1, 0, :]
            final_tod[1, :5] = tmp_arr[0, 1, :]
            final_tod[1, 5:] = tmp_arr[1, 1, :]
            final_tod[2, :5] = tmp_arr[2, 0, :]
            final_tod[2, 5:] = tmp_arr[3, 0, :]
            final_tod[3, :5] = tmp_arr[2, 1, :]
            final_tod[3, 5:] = tmp_arr[3, 1, :]

            print(final_tod[0, :])
            print(final_tod[1, :])
            print(final_tod[2, :])
            print(final_tod[3, :])

            np.testing.assert_almost_equal(final_tod, expected_tod, decimal=8)

    else:
        np.testing.assert_almost_equal(obs.tod, expected_tod, decimal=8)
