"""Test mapping routines"""

import numpy as np
import healpy as hp
import astropy.units as u
import litebird_sim as lbs
import litebird_sim.mapmaking.binner as mapping
from litebird_sim import CoordinateSystem


def test_accumulate_map_and_info():
    # Parameters
    res_map = np.arange(6).reshape(2, 3) + 1
    n_samples = 10
    psi = np.linspace(0, np.pi, n_samples)
    pix = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1])

    # Explicitly compute the dense pointing matrix
    pointing_matrix = np.zeros((n_samples, 2, 3), dtype=np.float32)
    for i in range(2):
        mask = pix == i
        pointing_matrix[mask, i, 0] = 1
        pointing_matrix[mask, i, 1] = np.cos(2 * psi[mask])
        pointing_matrix[mask, i, 2] = np.sin(2 * psi[mask])

    # Create the TOD and the target result
    tod = pointing_matrix.reshape(n_samples, -1).dot(res_map.reshape(-1))
    res_info = np.einsum("tpi,tpj->pij", pointing_matrix, pointing_matrix)
    res_info[:, 1, 0] = np.bincount(pix, tod)
    res_info[:, 2, 0] = np.bincount(pix, tod * np.cos(2 * psi))
    res_info[:, 2, 1] = np.bincount(pix, tod * np.sin(2 * psi))

    info = np.zeros((2, 3, 3))
    weights = np.ones(1)

    # Simulate the presence of *two* components in the TOD
    # (e.g., the CMB and the Galaxy)
    first_tod = np.expand_dims(tod, axis=0) * 0.25
    second_tod = np.expand_dims(tod, axis=0) * 0.75
    psi = np.expand_dims(psi, axis=0)
    pix = np.expand_dims(pix, axis=0)

    d_mask = np.ones(1)
    t_mask = np.ones(n_samples)

    # Now add both components to the TOD
    mapping._accumulate_samples_and_build_nobs_matrix(
        first_tod, pix, psi, weights, d_mask, t_mask, info, additional_component=False
    )
    mapping._accumulate_samples_and_build_nobs_matrix(
        second_tod, pix, psi, weights, d_mask, t_mask, info, additional_component=True
    )

    assert np.allclose(res_info, info)

    rhs = mapping._extract_map_and_fill_info(info)
    assert np.allclose(np.linalg.solve(info, rhs), res_map)


def test_make_binned_map_api_simulation(tmp_path):
    # We should add a more meaningful observation:
    # Currently this test just shows the interface
    sim = lbs.Simulation(
        base_path=tmp_path / "tut04",
        start_time=0.0,
        duration_s=86400.0,
        random_seed=12345,
    )

    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.radians(30),  # CORE-specific parameter
            spin_rate_hz=0.5 / 60,  # Ditto
            # We use astropy to convert the period (4 days) in
            # minutes, the unit expected for the precession period
            precession_rate_hz=1 / (4 * u.day).to("s").value,
        )
    )
    instr = lbs.InstrumentInfo(name="core", spin_boresight_angle_rad=np.deg2rad(65))
    det = lbs.DetectorInfo(name="foo", sampling_rate_hz=10, net_ukrts=1.0)
    obss = sim.create_observations(detectors=[det])

    lbs.prepare_pointings(
        obss,
        instr,
        sim.spin2ecliptic_quats,
    )

    pointings, _ = obss[0].get_pointings("all")

    nside = 64
    #    obss[0].pixind = hp.ang2pix(nside, pointings[..., 0], pointings[..., 1])
    #    obss[0].psi = pointings[..., 2]
    mapping.make_binned_map(nside=nside, observations=obss, pointings=[pointings])


def test_make_binned_map_basic_mpi():
    if lbs.MPI_COMM_WORLD.size > 2:
        return

    # Parameters
    res_map = np.arange(36, dtype=float).reshape(3, 12) + 1.0
    psi = np.tile([0, np.pi / 4.0, np.pi / 2.0], 12)
    pix = np.repeat(np.arange(12), 3)
    pointings = hp.pix2ang(1, pix)

    ndets = 2

    tod = np.empty(36)
    for i in range(len(tod)):
        tod[i] = (
            res_map[0, pix[i]]
            + np.cos(2 * psi[i]) * res_map[1, pix[i]]
            + np.sin(2 * psi[i]) * res_map[2, pix[i]]
        )

    # Craft the observation with the attributes needed for map-making
    obs = lbs.Observation(
        detectors=ndets,
        n_samples_global=18,
        start_time_global=0.0,
        sampling_rate_hz=1.0,
        comm=lbs.MPI_COMM_WORLD,
    )
    if obs.comm.rank == 0:
        obs.tod[:] = tod.reshape(ndets, 18)
        pointings = np.concatenate(
            [np.array(pointings).T.reshape((ndets, 18, 2)), psi.reshape(ndets, 18, 1)],
            axis=2,
        )

    obs.set_n_blocks(n_blocks_time=obs.comm.size, n_blocks_det=1)

    obs.pol_angle_rad = np.zeros(ndets)

    res = mapping.make_binned_map(
        nside=1,
        observations=[obs],
        pointings=[pointings],
        output_coordinate_system=CoordinateSystem.Ecliptic,
    )
    assert np.allclose(res.binned_map, res_map)

    obs.set_n_blocks(n_blocks_time=1, n_blocks_det=obs.comm.size)
    res = mapping.make_binned_map(
        nside=1,
        observations=[obs],
        pointings=[pointings],
        output_coordinate_system=CoordinateSystem.Ecliptic,
    )
    assert np.allclose(res.binned_map, res_map)


def test_hwp_offset_angle():
    # In this test the mapmaking uses an external HWP with an offset angle
    # the comparison is done at the power spectrum level computing analitically
    # the rotate power spectrum
    start_time = 0
    time_span_s = 200 * 24 * 3600
    nside = 64
    sampling_hz = 1
    net = 50.0
    fwhm_arcmin = 120
    hwp_radpsec = 4.084_070_449_666_731

    offset_angle = np.pi / 20

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
        fwhm_arcmin=fwhm_arcmin,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        pol_angle_rad=0.0,
    )

    detB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        fwhm_arcmin=fwhm_arcmin,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        pol_angle_rad=np.pi / 2.0,
    )

    hwp = lbs.IdealHWP(ang_speed_radpsec=hwp_radpsec)
    sim.set_hwp(hwp)

    sim.create_observations(detectors=[detT, detB])

    sky_params = lbs.SkyGenerationParams(
        make_cmb=True,
        make_fg=False,
        output_type="map",
        seed_cmb=1,
        apply_beam=True,
        bandpass_integration=False,
        nside=nside,
        units="uK_CMB",
    )

    sky = sim.get_sky(parameters=sky_params, store_in_observation=True)

    sim.prepare_pointings()

    sim.fill_tods()

    hwp_offset = lbs.IdealHWP(
        ang_speed_radpsec=hwp_radpsec, start_angle_rad=offset_angle
    )

    results = lbs.make_binned_map(nside, sim.observations, hwp=hwp_offset)

    cl_in = hp.anafast(sky["Boresight_detector_T"].values, lmax=2 * nside)

    cl_rot = hp.anafast(results.binned_map, lmax=2 * nside)

    # form Eq. 1 of https://arxiv.org/pdf/0905.1651
    cl_mod = (
        cl_in[1] * np.cos(4 * offset_angle) ** 2
        + cl_in[2] * np.sin(4 * offset_angle) ** 2
    )

    assert np.allclose(cl_mod, cl_rot[1], rtol=1e-8, atol=1e-4)
