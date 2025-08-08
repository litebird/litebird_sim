# -*- encoding: utf-8 -*-
import astropy
import healpy

import litebird_sim as lbs
import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base
import healpy as hp


def get_map_mask(pixels: npt.ArrayLike) -> npt.ArrayLike:
    return np.isfinite(pixels) & (pixels != healpy.UNSEEN)


def test_scan_map_no_interpolation():
    # The purpose of this test is to simulate the motion of the spacecraft
    # for one month (see `time_span_s`) and produce *two* maps: the first
    # is associated with the Observation `obs1` and is built using
    # `scan_map_in_observations` and `make_binned_map`, the second is associated
    # the Observation `obs2` and is built directly filling in the test
    # `tod`, `psi` and `pixind` and then using `make_binned_map`
    # In the second test `out_map1` is compared with both `out_map2` and the
    # input map. Both simulations use two orthogonal detectors at the boresight
    # and input maps generated with `np.random.normal`.
    # The final test verifies that scan_map with the option `input_map_in_galactic`
    # activated correctly handles internaly the coordinate rotation

    start_time = 0
    time_span_s = 30 * 24 * 3600
    nside = 256
    sampling_hz = 1
    net = 50.0
    hwp_radpsec = 4.084_070_449_666_731
    tolerance = 1e-5

    hpx = Healpix_Base(nside, "RING")

    npix = lbs.nside_to_npix(nside)

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=12345
    )

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
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
        pol_angle_rad=0.0,
    )

    detB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        pol_angle_rad=np.pi / 2.0,
    )

    np.random.seed(seed=123_456_789)
    maps = np.random.normal(0, 1, (3, npix))

    # This part tests the ecliptic coordinates
    in_map = {
        "Boresight_detector_T": maps,
        "Boresight_detector_B": maps,
        "Coordinates": lbs.CoordinateSystem.Ecliptic,
    }

    (obs1,) = sim.create_observations(detectors=[detT, detB])
    (obs2,) = sim.create_observations(detectors=[detT, detB])

    hwp = lbs.IdealHWP(ang_speed_radpsec=hwp_radpsec)

    lbs.prepare_pointings(
        obs1,
        instr,
        spin2ecliptic_quats,
        hwp=hwp,
    )

    pointings, hwp_angle = obs1.get_pointings("all")

    lbs.scan_map_in_observations(
        observations=obs1,
        maps=in_map,
        input_map_in_galactic=False,
        interpolation=None,
    )

    out_map1 = lbs.make_binned_map(
        nside=nside,
        observations=obs1,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
    )

    for idet in range(obs2.n_detectors):
        pixind = hpx.ang2pix(pointings[idet, :, 0:2])
        angle = 2 * pointings[idet, :, 2] - 2 * obs2.pol_angle_rad[idet] + 4 * hwp_angle
        obs2.tod[idet, :] = (
            maps[0, pixind]
            + np.cos(angle) * maps[1, pixind]
            + np.sin(angle) * maps[2, pixind]
        )

    out_map2 = lbs.make_binned_map(
        nside=nside,
        observations=obs2,
        pointings=pointings,
        hwp=hwp,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
    )

    mask1 = get_map_mask(out_map1.binned_map)
    mask2 = get_map_mask(out_map2.binned_map)
    np.testing.assert_array_equal(mask1, mask2)

    np.testing.assert_allclose(
        out_map1.binned_map[mask1],
        in_map["Boresight_detector_T"][mask1],
        rtol=tolerance,
        atol=0.1,
    )

    np.testing.assert_allclose(
        out_map1.binned_map[mask1], out_map2.binned_map[mask2], rtol=tolerance, atol=0.1
    )

    # This part tests the galactic coordinates
    r = hp.Rotator(coord=["E", "G"])
    maps = r.rotate_map_alms(maps, use_pixel_weights=False)

    in_map_G = {
        "Boresight_detector_T": maps,
        "Boresight_detector_B": maps,
        "Coordinates": lbs.CoordinateSystem.Galactic,
    }

    (obs1,) = sim.create_observations(detectors=[detT, detB])

    lbs.prepare_pointings(
        obs1,
        instr,
        spin2ecliptic_quats,
        hwp=hwp,
    )

    lbs.scan_map_in_observations(
        observations=obs1,
        maps=in_map_G,
        input_map_in_galactic=True,
    )

    out_map1 = lbs.make_binned_map(nside=nside, observations=obs1)
    mask1 = get_map_mask(out_map1.binned_map)

    np.testing.assert_allclose(
        out_map1.binned_map[mask1],
        in_map_G["Boresight_detector_T"][mask1],
        rtol=tolerance,
        atol=0.1,
    )


def test_scan_map_linear_interpolation():
    # This test is the same as test_scan_map_no_interpolation, but here we just check
    # that the call to `scan_map` does not fail

    start_time = 0
    time_span_s = 24 * 3600  # The time is much shorter here!
    nside = 256
    sampling_hz = 1
    net = 50.0
    hwp_radpsec = 4.084_070_449_666_731

    npix = lbs.nside_to_npix(nside)

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=12345
    )

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
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

    in_map = {
        "Boresight_detector_T": maps,
        "Boresight_detector_B": maps,
        "Coordinates": lbs.CoordinateSystem.Ecliptic,
    }

    (obs1,) = sim.create_observations(detectors=[detT, detB])
    (obs2,) = sim.create_observations(detectors=[detT, detB])

    hwp = lbs.IdealHWP(ang_speed_radpsec=hwp_radpsec)

    lbs.prepare_pointings(
        obs1,
        instr,
        spin2ecliptic_quats,
        hwp=hwp,
    )

    pointings, hwp_angle = obs1.get_pointings("all")

    # Just check that the code does not crash
    lbs.scan_map_in_observations(
        observations=obs1,
        maps=in_map,
        input_map_in_galactic=False,
        interpolation="linear",
    )


def test_scanning_list_of_obs(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=astropy.time.Time("2020-01-01T00:00:00"),
        duration_s=100.0,
        random_seed=12345,
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

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        sim.start_time,
        sim.duration_s,
        delta_time_s=60,
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    lbs.prepare_pointings(
        sim.observations,
        instr,
        spin2ecliptic_quats,
    )

    np.random.seed(seed=123_456_789)
    base_map = np.zeros((3, lbs.nside_to_npix(128)))

    # This part tests the ecliptic coordinates
    maps = {"A": base_map, "B": base_map, "Coordinates": lbs.CoordinateSystem.Ecliptic}

    # Just call the function and check that it does not raise any of the
    # "assert" that are placed at the beginning to check the consistency
    # of observations and pointings
    lbs.scan_map_in_observations(
        observations=sim.observations,
        maps=maps,
        input_map_in_galactic=True,
    )


def test_scanning_list_of_obs_in_other_component(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=astropy.time.Time("2020-01-01T00:00:00"),
        duration_s=120.0,
        random_seed=12345,
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

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        sim.start_time,
        sim.duration_s,
        delta_time_s=60,
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    lbs.prepare_pointings(
        sim.observations,
        instr,
        spin2ecliptic_quats,
    )

    # Create fake maps containing only nonzero pixels
    base_map = np.ones((3, lbs.nside_to_npix(128)))

    maps = {"A": base_map, "B": base_map, "Coordinates": lbs.CoordinateSystem.Galactic}

    for cur_obs in sim.observations:
        cur_obs.fg_tod = np.zeros_like(cur_obs.tod)

    lbs.scan_map_in_observations(
        observations=sim.observations,
        maps=maps,
        input_map_in_galactic=True,
        component="fg_tod",
    )

    # Check that the "tod" field has been left unchanged
    assert np.allclose(sim.observations[0].tod, 0.0)

    # Check that "fg_tod" has some non-zero data in it, as
    # all the pixels in the foreground map have nonzero pixels
    assert np.sum(np.abs(sim.observations[0].fg_tod)) > 0.0


def test_scan_map_algebras():
    start_time = 0
    time_span_s = 30 * 24 * 3600
    nside = 256
    sampling_hz = 1
    net = 50.0
    hwp_radpsec = 4.084_070_449_666_731
    tolerance = 1e-5

    hpx = Healpix_Base(nside, "RING")

    npix = lbs.nside_to_npix(nside)

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=12345
    )

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    sim.set_scanning_strategy(scanning)

    sim.set_instrument(instr)

    detT = lbs.DetectorInfo(
        name="Boresight_detector_T",
        sampling_rate_hz=sampling_hz,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        pol_angle_rad=0.0,
    )

    detB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        pol_angle_rad=np.pi / 2.0,
    )

    np.random.seed(seed=123_456_789)
    maps = np.random.normal(0, 1, (3, npix))

    # This part tests the ecliptic coordinates
    in_map = {
        "Boresight_detector_T": maps,
        "Boresight_detector_B": maps,
        "Coordinates": lbs.CoordinateSystem.Ecliptic,
    }

    (obs,) = sim.create_observations(detectors=[detT, detB])
    tod = np.empty_like(obs.tod)

    # standard computation with HWP:
    # - set the HWP in the simulation
    # - then fill_tods
    hwp = lbs.IdealHWP(ang_speed_radpsec=hwp_radpsec)

    sim.set_hwp(hwp)

    sim.prepare_pointings()

    sim.fill_tods(in_map, input_map_in_galactic=False)

    for idet in range(obs.n_detectors):
        pointings, hwp_angle = obs.get_pointings(idet)
        pixind = hpx.ang2pix(pointings[:, 0:2])
        angle = 2 * pointings[:, 2] - 2 * obs.pol_angle_rad[idet] + 4 * hwp_angle
        tod[idet, :] = (
            maps[0, pixind]
            + np.cos(angle) * maps[1, pixind]
            + np.sin(angle) * maps[2, pixind]
        )

    np.testing.assert_allclose(
        obs.tod,
        tod,
        rtol=tolerance,
        atol=0.1,
    )

    sim.nullify_tod()

    # here we use the low level function scan_map_in_observations
    # passing an external HWP
    # - so we initialize a new instance of the class HWP
    # - then we call scan_map_in_observations
    hwp_new = lbs.IdealHWP(ang_speed_radpsec=hwp_radpsec * 0.5)

    lbs.scan_map_in_observations(
        observations=obs,
        maps=in_map,
        hwp=hwp_new,
        input_map_in_galactic=False,
        interpolation=None,
    )

    for idet in range(obs.n_detectors):
        pointings, _ = obs.get_pointings(idet)
        pixind = hpx.ang2pix(pointings[:, 0:2])
        hwp_angle = lbs.pointings_in_obs._get_hwp_angle(obs, hwp_new)
        angle = 2 * pointings[:, 2] - 2 * obs.pol_angle_rad[idet] + 4 * hwp_angle
        tod[idet, :] = (
            maps[0, pixind]
            + np.cos(angle) * maps[1, pixind]
            + np.sin(angle) * maps[2, pixind]
        )

    np.testing.assert_allclose(
        obs.tod,
        tod,
        rtol=tolerance,
        atol=0.1,
    )

    sim.nullify_tod()

    # here same as before but precomputing the pointing information
    obs.precompute_pointings()

    lbs.scan_map_in_observations(
        observations=obs,
        maps=in_map,
        hwp=hwp_new,
        input_map_in_galactic=False,
        interpolation=None,
    )

    for idet in range(obs.n_detectors):
        pointings = obs.pointing_matrix[idet]
        pixind = hpx.ang2pix(pointings[:, 0:2])
        hwp_angle = lbs.pointings_in_obs._get_hwp_angle(obs, hwp_new)
        angle = 2 * pointings[:, 2] - 2 * obs.pol_angle_rad[idet] + 4 * hwp_angle
        tod[idet, :] = (
            maps[0, pixind]
            + np.cos(angle) * maps[1, pixind]
            + np.sin(angle) * maps[2, pixind]
        )

    np.testing.assert_allclose(
        obs.tod,
        tod,
        rtol=tolerance,
        atol=0.1,
    )

    # here we test the algebra in absence of HWP
    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=12345
    )
    sim.set_scanning_strategy(scanning)
    sim.set_instrument(instr)

    (obs,) = sim.create_observations(detectors=[detT, detB])

    sim.prepare_pointings()

    sim.fill_tods(in_map, input_map_in_galactic=False)

    for idet in range(obs.n_detectors):
        pointings, _ = obs.get_pointings(idet)
        pixind = hpx.ang2pix(pointings[:, 0:2])
        angle = 2 * pointings[:, 2] + 2 * obs.pol_angle_rad[idet]
        tod[idet, :] = (
            maps[0, pixind]
            + np.cos(angle) * maps[1, pixind]
            + np.sin(angle) * maps[2, pixind]
        )

    np.testing.assert_allclose(
        obs.tod,
        tod,
        rtol=tolerance,
        atol=0.1,
    )
