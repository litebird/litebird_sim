# -*- encoding: utf-8 -*-

import litebird_sim as lbs


def test_distribution():
    comm_world = lbs.MPI_COMM_WORLD

    if comm_world.rank == 0:
        print(f"MPI configuration: {lbs.MPI_CONFIGURATION}")

    sim = lbs.Simulation(mpi_comm=comm_world)
    det1 = lbs.Detector(
        name="pol01",
        wafer="mywafer",
        pixel=1,
        pixel_type="A",
        channel=30,
        sampling_frequency_hz=5,
        fwhm_arcmin=30,
        ellipticity=1.0,
        net_ukrts=1.0,
        fknee_mhz=10,
        fmin_hz=1e-6,
        alpha=1.0,
        pol="Q",
        orientation="A",
        quaternion=[0, 0, 0, 0],
    )
    det2 = lbs.Detector(
        name="pol02",
        wafer="mywafer",
        pixel=2,
        pixel_type="B",
        channel=44,
        sampling_frequency_hz=50,
        fwhm_arcmin=30,
        ellipticity=1.0,
        net_ukrts=1.0,
        fknee_mhz=10,
        fmin_hz=1e-6,
        alpha=1.0,
        pol="Q",
        orientation="A",
        quaternion=[0, 0, 0, 0],
    )
    sim.create_observations(
        [det1, det2],
        num_of_obs_per_detector=10,
        start_time=0.0,
        duration_s=86400.0 * 10,
    )

    assert (
        sim.observations
    ), "No observations have been defined for process {lbs.MPI_RANK + 1}/{lbs.MPI_SIZE}"

    if comm_world.size > 1:
        allobs = comm_world.allreduce([x.detector.name for x in sim.observations])
    else:
        allobs = sim.observations

    assert len(allobs) == 20


def main():
    test_distribution()


main()
