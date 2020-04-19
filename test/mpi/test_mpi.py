# -*- encoding: utf-8 -*-

import litebird_sim as lbs


def test_distribution():
    assert isinstance(lbs.HAVE_MPI4PY, bool)

    if lbs.MPI_RANK == 0:
        print(f"MPI_CONFIGURATION = {lbs.MPI_CONFIGURATION}")

    sim = lbs.Simulation(use_mpi=lbs.HAVE_MPI4PY)
    det1 = lbs.Detector(name="pol01", beam_z=[0, 0, 1], sampfreq_hz=5, simulation=sim)
    det2 = lbs.Detector(name="pol01", beam_z=[0, 0, 1], sampfreq_hz=50, simulation=sim)
    sim.create_observations(
        [det1, det2],
        num_of_obs_per_detector=10,
        start_time=0.0,
        duration_s=86400.0 * 10,
    )

    assert (
        sim.observations
    ), "No observations have been defined for process {lbs.MPI_RANK + 1}/{lbs.MPI_SIZE}"

    if lbs.MPI_SIZE > 1:
        allobs = lbs.MPI_COMM_WORLD.allreduce(sim.observations)
    else:
        allobs = sim.observations

    assert len(allobs) == 20


def main():
    test_distribution()


main()
