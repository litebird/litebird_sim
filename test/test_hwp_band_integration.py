import tempfile
from pathlib import Path

import litebird_sim as lbs
import numpy as np
from litebird_sim.input_sky import SkyGenerator
from litebird_sim.scan_map import scan_map_in_observations


def test_hwp_band_integration():
    # testing that band integration with a single frequency yields the same TOD
    # as the usual tod filling without integration
    start_time = 0
    time_span_s = 1 * 24 * 3600
    nside = 64
    fwhm_arcmin = 37.805193
    hwp_radpsec = 46 * 2 * np.pi / 60

    list_of_sims = []
    for i in range(2):
        sim = lbs.Simulation(
            start_time=start_time, duration_s=time_span_s, random_seed=0
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

        list_of_sims.append(sim)

    channelinfo = lbs.FreqChannelInfo(
        bandcenter_ghz=140.0,
        channel="L4-140",
        bandwidth_ghz=1.0,
        fwhm_arcmin=fwhm_arcmin,
    )

    det = lbs.DetectorInfo.from_dict(
        {
            "channel": channelinfo,
            "bandcenter_ghz": 140.0,
            "bandwidth_ghz": 1.0,
            "sampling_rate_hz": 1,
            "fwhm_arcmin": fwhm_arcmin,
            "quat": [
                0.03967584136504414,
                0.03725809501267564,
                0.0,
                0.9985177324254199,
            ],
            "pointing_theta_phi_psi_deg": [0, 0, 0],
            "jones_hwp": {
                "0f": np.array(
                    [
                        [
                            9.89e-01 * np.exp(1j * np.deg2rad(-1.38e02)),
                            7.43e-02 * np.exp(1j * np.deg2rad(1.45e02)),
                        ],
                        [
                            7.41e-02 * np.exp(1j * np.deg2rad(1.49e02)),
                            9.72e-01 * np.exp(1j * np.deg2rad(7.71e01)),
                        ],
                    ],
                    dtype=np.complex128,
                ),
                "2f": np.array([[0, 0], [0, 0]], dtype=np.complex128),
            },
        }
    )

    sky_params = lbs.SkyGenerationParams(
        make_cmb=True,
        seed_cmb=1234,
        output_type="map",
        make_dipole=False,
        make_fg=True,
        fg_models=["s0", "d0", "f1"],
        apply_beam=True,
        bandpass_integration=False,
        nside=nside,
        units="K_CMB",
    )

    gen_sky_0 = lbs.SkyGenerator(
        parameters=sky_params,
        channels=channelinfo,
    )
    input_maps_0 = gen_sky_0.execute()["L4-140"]

    gen_sky_1 = SkyGenerator(
        parameters=sky_params,
        frequencies_ghz=np.array([140.0]),
        fwhm_rad=np.radians(fwhm_arcmin / 60.0),
    )
    input_maps_1 = gen_sky_1.execute()

    list_of_sims[0].set_hwp(
        lbs.NonIdealHWP(
            hwp_radpsec,
            harmonic_expansion=True,
            calculus=lbs.Calc.JONES,
        )
    )

    text = """freq,Jxx_0f,Phxx_0f,Jxy_0f,Phxy_0f,Jyx_0f,Phyx_0f,Jyy_0f,Phyy_0f,Jxx_2f,Phxx_2f,Jxy_2f,Phxy_2f,Jyx_2f,Phyx_2f,Jyy_2f,Phyy_2f
    138.0,0.906,-88.5,0.0477,145.0,0.0458,172.0,0.969,125,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    140.0,0.989,-138.0,0.0743,145.0,0.0741,149.0,0.972,77.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    142.0,0.965,171.0,0.106,145.0,0.106,103.0,0.973,33.8,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8", newline="\n"
    ) as tmp:
        path = Path(tmp.name)
        tmp.write(text)

    list_of_sims[1].set_hwp(
        lbs.NonIdealHWP(
            hwp_radpsec,
            harmonic_expansion=True,
            calculus=lbs.Calc.JONES,
            jones_per_freq_csv_path=path,
        )
    )

    for i in range(len(list_of_sims)):
        list_of_sims[i].create_observations(
            detectors=[det],
            split_list_over_processes=False,
        )
        list_of_sims[i].prepare_pointings()

    scan_map_in_observations(
        observations=list_of_sims[0].observations[0],
        maps=input_maps_0,
    )

    scan_map_in_observations(
        observations=list_of_sims[1].observations[0],
        maps=input_maps_1,
        integrate_in_band=True,
    )

    np.testing.assert_almost_equal(
        list_of_sims[0].observations[0].tod,
        list_of_sims[1].observations[0].tod,
        decimal=12,
        verbose=True,
    )
