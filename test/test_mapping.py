"""Test mapping routines
"""
import numpy as np
import healpy as hp
import astropy.units as u
import litebird_sim as lbs
import litebird_sim.mapping as mapping


def test_accumulate_map_and_info():
    # Parameters
    res_map = np.arange(6).reshape(2, 3) + 1
    n_samples = 10
    psi = np.linspace(0, np.pi, n_samples)
    pix = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1])

    # Explicitely compute the dense pointing matrix
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
    mapping._accumulate_map_and_info(tod, pix, psi, info)

    assert np.allclose(res_info, info)

    rhs = mapping._extract_map_and_fill_info(info)
    assert np.allclose(np.linalg.solve(info, rhs), res_map)


def test_make_bin_map_api_simulation(tmp_path):
    # We should add a more meaningful observation:
    # Currently this test just shows the interface
    sim = lbs.Simulation(
        base_path=tmp_path / "tut04",
        start_time=0.0,
        duration_s=86400.0,
    )

    sim.generate_spin2ecl_quaternions(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.radians(30),  # CORE-specific parameter
            spin_rate_hz=0.5 / 60,  # Ditto
            # We use astropy to convert the period (4 days) in
            # minutes, the unit expected for the precession period
            precession_rate_hz=1 / (4 * u.day).to("s").value,
        )
    )
    instr = lbs.InstrumentInfo(name="core", spin_boresight_angle_rad=np.radians(65))
    det = lbs.DetectorInfo(name="foo", sampling_rate_hz=10)
    obss = sim.create_observations(detectors=[det])
    pointings = lbs.get_pointings(
        obss[0],
        sim.spin2ecliptic_quats,
        detector_quats=[det.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    nside = 64
    obss[0].pixel = hp.ang2pix(nside, pointings[..., 0], pointings[..., 1])
    obss[0].psi = pointings[..., 2]
    res = mapping.make_bin_map(obss, nside)


def test_make_bin_map_basic_mpi():
    if lbs.MPI_COMM_WORLD.size > 2:
        return

    # Parameters
    res_map = np.arange(9).reshape(3, 3) + 1
    n_samples = 10
    psi = np.array([1, 2, 1, 4, 4, 1, 4, 0, 0, 0]) * np.pi / 5
    pix = np.array([0, 0, 1, 0, 1, 2, 2, 0, 2, 1])

    # Explicitely compute the dense pointing matrix and hence the TOD
    pointing_matrix = np.zeros((n_samples,) + res_map.shape, dtype=np.float32)
    for i in range(len(res_map)):
        mask = pix == i
        pointing_matrix[mask, i, 0] = 1
        pointing_matrix[mask, i, 1] = np.cos(2 * psi[mask])
        pointing_matrix[mask, i, 2] = np.sin(2 * psi[mask])

    tod = pointing_matrix.reshape(n_samples, -1).dot(res_map.reshape(-1))

    # Craft the observation with the attributes needed for map-making
    obs = lbs.Observation(
        detectors=2,
        n_samples_global=5,
        start_time_global=0.0,
        sampling_rate_hz=1.0,
        comm=lbs.MPI_COMM_WORLD,
    )
    if obs.comm.rank == 0:
        obs.tod[:] = tod.reshape(2, 5)
        obs.pixel = pix.reshape(2, 5)
        obs.psi = psi.reshape(2, 5)

    obs.set_n_blocks(n_blocks_time=obs.comm.size, n_blocks_det=1)
    res = mapping.make_bin_map([obs], 1)[: len(res_map)]
    assert np.allclose(res, res_map)

    obs.set_n_blocks(n_blocks_time=1, n_blocks_det=obs.comm.size)
    res = mapping.make_bin_map([obs], 1)[: len(res_map)]
    assert np.allclose(res, res_map)
