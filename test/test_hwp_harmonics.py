from pathlib import Path

import litebird_sim as lbs
import numpy as np
import pytest
from litebird_sim.hwp_harmonics import jones_methods, mueller_methods
from litebird_sim.hwp_harmonics.hwp_harmonics import compute_orientation_from_detquat
from litebird_sim.scan_map import scan_map_in_observations


@pytest.mark.parametrize(
    "calculus",
    [
        lbs.HWPFormalism.MUELLER,
        lbs.HWPFormalism.JONES,
    ],
)
def test_hwp_harmonics(calculus):
    start_time = 0
    time_span_s = 1000
    nside = 64
    sampling = 1
    hwp_radpsec = lbs.IdealHWP(
        46 * 2 * np.pi / 60,
    ).ang_speed_radpsec

    list_of_sims = []
    for i in range(2):
        sim = lbs.Simulation(
            start_time=start_time, duration_s=time_span_s, random_seed=0
        )

        comm = sim.mpi_comm
        _ = comm.rank

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

        sky_params = lbs.SkyGenerationParams(
            make_cmb=True,
            seed_cmb=1234,
            output_type="map",
            make_dipole=True,
            make_fg=True,
            fg_models=["s0", "d0", "f1"],
            apply_beam=True,
            bandpass_integration=False,
            nside=nside,
            units="K_CMB",
        )

        gen_sky = lbs.SkyGenerator(parameters=sky_params, channels=channelinfo)

        input_maps = gen_sky.execute()["L4-140"]

        list_of_sims.append(sim)

    nonideal_hwp = lbs.NonIdealHWP(
        ang_speed_radpsec=hwp_radpsec,
        harmonic_expansion=True,
        calculus=calculus,
    )
    if calculus == lbs.HWPFormalism.MUELLER:
        list_of_sims[1].observations[0].mueller_hwp[0] = {
            "0f": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
            "2f": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
            "4f": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float64),
        }
    else:
        list_of_sims[1].observations[0].jones_hwp[0] = {
            "0f": np.array([[1, 0], [0, -1]], dtype=np.float64),
            "2f": np.array([[0, 0], [0, 0]], dtype=np.float64),
        }

    list_of_sims[1].set_hwp(nonideal_hwp)

    ideal_hwp = lbs.IdealHWP(
        ang_speed_radpsec=hwp_radpsec,
    )
    list_of_sims[0].set_hwp(ideal_hwp)

    list_of_sims[0].prepare_pointings()
    list_of_sims[1].prepare_pointings()

    # we have two similar observations, now we will compute the TOD
    # using both scan_map_in_observations and  fill_tod_with_hwp_harmonics
    # and check that they are the same

    scan_map_in_observations(
        observations=list_of_sims[0].observations[0],
        maps=input_maps,
    )

    mueller_phases = {
        "2f": np.zeros((3, 3)),
        "4f": np.array(
            [
                [0, 0, 0],
                [0, 0, -np.pi / 2],
                [0, -np.pi / 2, np.pi],
            ],
            dtype=np.float64,
        ),
    }

    scan_map_in_observations(
        observations=list_of_sims[1].observations[0],
        maps=input_maps,
        mueller_phases=mueller_phases,
    )

    # Check that we are using 64-bit floating-point numbers for pointings. See
    # https://github.com/litebird/litebird_sim/pull/429
    pointings, _ = list_of_sims[1].observations[0].get_pointings()
    assert pointings.dtype == np.float64
    pointings, _ = list_of_sims[0].observations[0].get_pointings()
    assert pointings.dtype == np.float64

    np.testing.assert_almost_equal(
        list_of_sims[0].observations[0].tod,
        list_of_sims[1].observations[0].tod,
        decimal=12,
        verbose=True,
    )


def test_hwp_harmonics_kernels_regression():
    # Generic (non-ideal) Mueller and Jones parameters, random phases, non-uniform
    # band weights, 2f HWPSS and non-linearity on. The reference TODs were computed
    # with the kernels of litebird_sim a13a1696, before they were rewritten in
    # PR #581: this is a regression test, not a validation of the physics
    ref = np.load(Path(__file__).parent / "hwp_harmonics_kernels_reference.npz")

    phi, xi = float(ref["phi"]), float(ref["xi"])
    common = dict(
        cos2Xi2Phi=np.cos(2 * xi - 2 * phi),
        sin2Xi2Phi=np.sin(2 * xi - 2 * phi),
        phi=phi,
        xi=xi,
        rho=ref["rho"],
        psi=ref["psi"],
        apply_non_linearity=True,
        g_one_over_k=float(ref["g_one_over_k"]),
        add_2f_hwpss=True,
        amplitude_2f_k=float(ref["amplitude_2f_k"]),
    )

    tod = np.zeros_like(ref["rho"])
    mueller_methods.compute_signal_for_one_detector(
        tod_det=tod,
        m0f=ref["m0f"],
        m2f=ref["m2f"],
        m4f=ref["m4f"],
        mapT=ref["T"],
        mapQ=ref["Q"],
        mapU=ref["U"],
        phases_2f=ref["phases_2f"],
        phases_4f=ref["phases_4f"],
        **common,
    )
    np.testing.assert_allclose(tod, ref["tod_mueller"], rtol=1e-12, atol=0)

    tod = np.zeros_like(ref["rho"])
    jones_methods.compute_signal_for_one_detector(
        tod_det=tod,
        deltas_j0f=ref["j0f"],
        deltas_j2f=ref["j2f"],
        mapT=ref["T"],
        mapQ=ref["Q"],
        mapU=ref["U"],
        **common,
    )
    np.testing.assert_allclose(tod, ref["tod_jones"], rtol=1e-12, atol=0)

    tod = np.zeros_like(ref["rho"])
    jones_methods.integrate_inband_signal_for_one_detector(
        tod_det=tod,
        freqs=ref["freqs"],
        band=ref["band"],
        deltas_j0f=ref["j0f_band"],
        deltas_j2f=ref["j2f_band"],
        mapT=ref["TB"],
        mapQ=ref["QB"],
        mapU=ref["UB"],
        **common,
    )
    np.testing.assert_allclose(tod, ref["tod_jones_band"], rtol=1e-12, atol=0)
