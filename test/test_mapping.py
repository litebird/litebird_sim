import numpy as np
import litebird_sim as lbs
import litebird_sim.mapping as mapping
import healpy as hp
import astropy.units as u

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
    res_info = np.einsum('tpi,tpj->pij', pointing_matrix, pointing_matrix)
    res_info[:, 1, 0] = np.bincount(pix, tod)
    res_info[:, 2, 0] = np.bincount(pix, tod * np.cos(2 * psi))
    res_info[:, 2, 1] = np.bincount(pix, tod * np.sin(2 * psi))
    
    info = np.zeros((2, 3, 3))
    mapping._accumulate_map_and_info(tod, pix, psi, info)

    assert np.allclose(res_info, info)

    rhs = mapping._extract_map_and_fill_info(info)
    assert np.allclose(np.linalg.solve(info, rhs), res_map)


def test_make_bin_map():
    # We should add a more meaningful observation:
    # Currently this test just shows the interface
    sim = lbs.Simulation(
	base_path="./tut04",
	start_time=0,
	duration_s=86400.,
    )

    sim.generate_spin2ecl_quaternions(
	scanning_strategy=lbs.SpinningScanningStrategy(
	    spin_sun_angle_deg=30, # CORE-specific parameter
	    spin_rate_rpm=0.5,     # Ditto
	    # We use astropy to convert the period (4 days) in
	    # minutes, the unit expected for the precession period
	    precession_period_min=(4 * u.day).to("min").value,
	)
    )
    instr = lbs.Instrument(name="core", spin_boresight_angle_deg=65)
    det = lbs.Detector(name="foo", sampling_rate_hz=10)
    obss = sim.create_observations(detectors=[det])
    pointings = obss[0].get_pointings(
	sim.spin2ecliptic_quats,
	detector_quats=[det.quat],
	bore2spin_quat=instr.bore2spin_quat,
    )

    nside = 64
    obss[0].pixidx = hp.ang2pix(nside, pointings[..., 0], pointings[..., 1])
    obss[0].psi = pointings[..., 2]
    res = mapping.make_bin_map(obss, nside)
