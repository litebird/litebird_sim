# -*- encoding: utf-8 -*-
import astropy

import litebird_sim as lbs
import numpy as np
from ducc0.healpix import Healpix_Base
import healpy as hp


def test_scan_map():

    # The purpose of this test is to simulate the motion of the spacecraft
    # for one year (see `time_span_s`) and produce *two* maps: the first
    # is associated with the Observation `obs1` and is built using
    # `scan_map_in_observations` and `make_bin_map`, the second is associated
    # the Observation `obs2` and is built directly filling in the test
    # `tod`, `psi` and `pixind` and then using `make_bin_map`
    # In the second test `out_map1` is compared with both `out_map2` and the
    # input map. Both simulations use two orthogonal detectors at the boresight
    # and input maps generated with `np.random.normal`.
    # The final test verifies that scan_map with the option `input_map_in_galactic`
    # activated correctly handles internaly the coordinate rotation

    start_time = 0
    time_span_s = 365 * 24 * 3600
    nside = 256
    sampling_hz = 1
    net = 50.0
    hwp_radpsec = 4.084_070_449_666_731

    hpx = Healpix_Base(nside, "RING")

    npix = lbs.nside_to_npix(nside)

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.set_scanning_strategy(
        start_time, time_span_s, delta_time_s=60
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    detT = lbs.DetectorInfo(
        name="Boresight_detector_T",
        sampling_rate_hz=sampling_hz,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    detB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
    )

    np.random.seed(seed=123_456_789)
    maps = np.random.normal(0, 1, (3, npix))

    # This part tests the ecliptic coordinates
    in_map = {"Boresight_detector_T": maps, "Boresight_detector_B": maps}

    (obs1,) = sim.create_observations(detectors=[detT, detB])
    (obs2,) = sim.create_observations(detectors=[detT, detB])

    pointings = lbs.get_pointings(
        obs1,
        spin2ecliptic_quats=spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
        detector_quats=[detT.quat, detB.quat],
        hwp=lbs.IdealHWP(ang_speed_radpsec=hwp_radpsec),
    )

    lbs.scan_map_in_observations(
        obs=obs1,
        pointings=pointings,
        maps=in_map,
        input_map_in_galactic=False,
    )
    out_map1 = lbs.make_bin_map(obs1, nside, output_map_in_galactic=False)

    obs2.pointings = pointings[:, :, 0:2]
    obs2.psi = pointings[:, :, 2]

    for idet in range(obs2.n_detectors):
        pixind = hpx.ang2pix(obs2.pointings[idet])
        obs2.tod[idet, :] = (
            maps[0, pixind]
            + np.cos(2 * obs2.psi[idet, :]) * maps[1, pixind]
            + np.sin(2 * obs2.psi[idet, :]) * maps[2, pixind]
        )

    out_map2 = lbs.make_bin_map(obs2, nside, output_map_in_galactic=False)

    np.testing.assert_allclose(
        out_map1, in_map["Boresight_detector_T"], rtol=1e-6, atol=1e-6
    )

    np.testing.assert_allclose(out_map1, out_map2, rtol=1e-6, atol=1e-6)

    # This part tests the galactic coordinates
    r = hp.Rotator(coord=["E", "G"])
    maps = r.rotate_map_alms(maps, use_pixel_weights=False)

    in_map_G = {"Boresight_detector_T": maps, "Boresight_detector_B": maps}

    (obs1,) = sim.create_observations(detectors=[detT, detB])

    pointings = lbs.get_pointings(
        obs1,
        spin2ecliptic_quats=spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
        detector_quats=[detT.quat, detB.quat],
    )

    lbs.scan_map_in_observations(
        obs1,
        in_map_G,
        pointings=pointings,
        input_map_in_galactic=True,
    )
    out_map1 = lbs.make_bin_map(obs1, nside)

    np.testing.assert_allclose(
        out_map1, in_map_G["Boresight_detector_T"], rtol=1e-6, atol=1e-6
    )


def test_scanning_list_of_obs(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=astropy.time.Time("2020-01-01T00:00:00"),
        duration_s=100.0,
    )
    dets = [
        lbs.DetectorInfo(name="A", sampling_rate_hz=1),
        lbs.DetectorInfo(name="B", sampling_rate_hz=1),
    ]

    sim.create_observations(
        detectors=dets,
        num_of_obs_per_detector=2,
    )

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=sim.start_time,
    )

    spin2ecliptic_quats = scanning.set_scanning_strategy(
        sim.start_time,
        sim.duration_s,
        delta_time_s=60,
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    pointings = lbs.get_pointings_for_observations(
        sim.observations,
        spin2ecliptic_quats=spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
    )

    np.random.seed(seed=123_456_789)
    base_map = np.zeros((3, lbs.nside_to_npix(128)))

    # This part tests the ecliptic coordinates
    maps = {"A": base_map, "B": base_map}

    # Just call the function and check that it does not raise any of the
    # "assert" that are placed at the beginning to check the consistency
    # of observations and pointings
    lbs.scan_map_in_observations(
        obs=sim.observations,
        maps=maps,
        pointings=pointings,
        input_map_in_galactic=True,
    )
