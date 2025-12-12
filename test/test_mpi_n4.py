import numpy as np

import litebird_sim as lbs
from litebird_sim import mpi
from litebird_sim.hwp_harmonics import mueller_interpolation


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

        theta = det.pointing_theta_phi_psi_deg[0]

        det.mueller_hwp = {
            "0f": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
            "2f": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
            "4f": np.array(
                [
                    [0, 0, 0],
                    [mueller_interpolation(theta, "4f", 1, 0), 1, 1],
                    [mueller_interpolation(theta, "4f", 2, 0), 1, 1],
                ],
                dtype=np.float64,
            ),
        }

        sim.set_hwp(
            lbs.NonIdealHWP(
                ang_speed_radpsec=hwp_radpsec,
                harmonic_expansion=True,
                calculus=lbs.Calc.MUELLER,
            )
        )

        det.sampling_rate_hz = sampling

        dets.append(det)

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

    if comm.rank == 0:
        mbs = lbs.Mbs(simulation=sim, parameters=mbs_params, channel_list=[channelinfo])

        input_maps = mbs.run_all()[0]["MF1_140"]

    else:
        input_maps = None

    if mpi.MPI_ENABLED:
        input_maps = comm.bcast(input_maps, root=0)

    lbs.scan_map_in_observations(
        maps=input_maps,
        observations=[obs],
        save_tod=True,
    )

    # tod calculated for the same simulation using only 1 cpu
    expected_tod = np.array(
        [
            [
                1.53299456e-03,
                1.52484816e-03,
                1.53325195e-03,
                1.49508251e-03,
                1.49978010e-03,
                1.49765855e-03,
                1.49710907e-03,
                1.52729300e-03,
                1.52358599e-03,
                1.52939605e-03,
            ],
            [
                2.24427693e-03,
                2.24184804e-03,
                2.23166589e-03,
                2.22934526e-03,
                2.23292410e-03,
                2.22825119e-03,
                2.23380793e-03,
                2.22761487e-03,
                2.26642378e-03,
                2.29074061e-03,
            ],
            [
                2.24442524e-03,
                2.24241056e-03,
                2.23042397e-03,
                2.23121536e-03,
                2.23050755e-03,
                2.23110872e-03,
                2.23063445e-03,
                2.23096577e-03,
                2.26299674e-03,
                2.29409453e-03,
            ],
            [
                -2.14275962e-04,
                -2.12599334e-04,
                -2.14288637e-04,
                1.16508354e-05,
                1.49919360e-05,
                1.46891825e-05,
                1.57095856e-05,
                -1.29254782e-04,
                -1.28528220e-04,
                -1.71196109e-04,
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

            np.testing.assert_almost_equal(final_tod, expected_tod, decimal=10)

    else:
        np.testing.assert_almost_equal(obs.tod, expected_tod, decimal=10)
