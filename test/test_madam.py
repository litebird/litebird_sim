# -*- encoding: utf-8 -*-

import numpy as np
from numpy.random import SeedSequence, MT19937, RandomState
import astropy.units as u

import litebird_sim as lbs


def test_madam(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "destriper_output", start_time=0, duration_s=86400.0
    )

    sim.generate_spin2ecl_quaternions(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(30),  # CORE-specific parameter
            spin_rate_hz=0.5 / 60,  # Ditto
            # We use astropy to convert the period (4 days) in
            # seconds
            precession_rate_hz=1.0 / (4 * u.day).to("s").value,
        )
    )
    instr = lbs.InstrumentInfo(name="core", spin_boresight_angle_rad=np.deg2rad(65))
    detectors = [
        lbs.DetectorInfo(name="0A", sampling_rate_hz=10),
        lbs.DetectorInfo(
            name="0B", sampling_rate_hz=10, quat=lbs.quat_rotation_z(np.pi / 2)
        ),
    ]
    sim.create_observations(
        detectors=detectors,
        # num_of_obs_per_detector=lbs.MPI_COMM_WORLD.size,
        dtype_tod=np.float64,
        n_blocks_time=lbs.MPI_COMM_WORLD.size,
        split_list_over_processes=False,
    )

    lbs.get_pointings_for_observations(
        sim.observations,
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
    )

    # Generate some white noise
    rs = RandomState(MT19937(SeedSequence(123456789)))
    for cur_obs in sim.observations:
        cur_obs.tod *= 0.0
        cur_obs.tod += rs.randn(*cur_obs.tod.shape)

    params = lbs.DestriperParameters(
        nside=16,
        nnz=3,
        baseline_length_s=100,
        iter_max=10,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
        return_npp=True,
        return_invnpp=True,
        return_rcond=True,
    )

    # Just check that all the files are saved without errors/exceptions:
    # to verify that the input files are ok, we should download and install
    # Madam…
    lbs.save_simulation_for_madam(sim, detectors=detectors, params=params)
