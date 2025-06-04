import litebird_sim as lbs
import numpy as np
from litebird_sim import mpi
from litebird_sim.hwp_sys.hwp_sys import mueller_interpolation


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

    sim.set_hwp(lbs.IdealHWP(hwp_radpsec))

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

    if comm.rank == 0:
        mbs = lbs.Mbs(simulation=sim, parameters=Mbsparams, channel_list=[channelinfo])

        input_maps = mbs.run_all()[0]["MF1_140"]

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
        build_map_on_the_fly=False,
        comm=comm,
    )

    hwp_sys.fill_tod(
        observations=[obs],
        input_map_in_galactic=False,
        save_tod=True,
    )
    # tod calculated for the same simulation using only 1 cpu
    expected_tod = np.array(
        [
            [
                1.5313686e-03,
                1.5240961e-03,
                1.5315217e-03,
                1.4930577e-03,
                1.4962899e-03,
                1.4954093e-03,
                1.4938965e-03,
                1.5316117e-03,
                1.5286847e-03,
                1.5333812e-03,
            ],
            [
                2.2478492e-03,
                2.2439796e-03,
                2.2345495e-03,
                2.2312715e-03,
                2.2357476e-03,
                2.2302750e-03,
                2.2365001e-03,
                2.2297977e-03,
                2.2673765e-03,
                2.2908563e-03,
            ],
            [
                2.2479978e-03,
                2.2445428e-03,
                2.2333062e-03,
                2.2331434e-03,
                2.2333285e-03,
                2.2331355e-03,
                2.2333232e-03,
                2.2331520e-03,
                2.2639481e-03,
                2.2942105e-03,
            ],
            [
                -2.1046442e-04,
                -2.1007526e-04,
                -2.1058515e-04,
                1.2266909e-05,
                8.3528857e-06,
                8.4155436e-06,
                9.3025519e-06,
                -1.2861186e-04,
                -1.2815163e-04,
                -1.6338057e-04,
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
