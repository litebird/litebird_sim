# -*- encoding: utf-8 -*-

import numpy as np
import astropy.units as u

import litebird_sim as lbs


def _num_of_obs_per_detector(descr: lbs.MpiDistributionDescr, det_name: str) -> int:
    return sum(
        [
            1
            for mpi_proc in descr.mpi_processes
            for obs in mpi_proc.observations
            if det_name in obs.det_names
        ]
    )


def run_test_on_madam(tmp_path, n_blocks_det: int, n_blocks_time: int):
    sim = lbs.Simulation(
        base_path=tmp_path / "destriper_output",
        start_time=0,
        duration_s=86400.0,
        mpi_comm=lbs.MPI_COMM_WORLD,
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
        dtype_tod=np.float64,
        split_list_over_processes=False,
        num_of_obs_per_detector=2,
        n_blocks_det=n_blocks_det,
        n_blocks_time=n_blocks_time,
    )

    distribution = sim.describe_mpi_distribution()
    assert distribution is not None
    if lbs.MPI_COMM_WORLD.rank == 0:
        print(distribution)

    lbs.get_pointings_for_observations(
        sim.observations,
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
    )

    for cur_obs in sim.observations:
        cur_obs.tod[:] = float(lbs.MPI_COMM_WORLD.rank)

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
    result = lbs.save_simulation_for_madam(sim, detectors=detectors, params=params)

    if lbs.MPI_COMM_WORLD.rank != 0:
        # The value of "result" is meaningful only for MPI process #0, so there
        # is no point in continuing
        return

    assert len(result["detectors"]) == 2

    for det_name in ("0A", "0B"):
        obs_per_detector = _num_of_obs_per_detector(
            descr=distribution, det_name=det_name
        )

        tod_files = [x for x in result["tod_files"] if x["det_name"] == det_name]
        assert len(tod_files) == obs_per_detector

        pointing_files = [
            x for x in result["pointing_files"] if x["det_name"] == det_name
        ]
        assert len(pointing_files) == obs_per_detector


def test_madam(tmp_path):
    run_test_on_madam(tmp_path, n_blocks_det=lbs.MPI_COMM_WORLD.size, n_blocks_time=1)
    run_test_on_madam(tmp_path, n_blocks_det=1, n_blocks_time=lbs.MPI_COMM_WORLD.size)
