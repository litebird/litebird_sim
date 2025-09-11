import tempfile
import numpy as np
import pytest
import litebird_sim as lbs

# data for testing detector blocks and MPI sub-communicators
sampling_freq_Hz = 1
dets = [
    lbs.DetectorInfo(
        name="channel1_w9_detA",
        wafer="wafer_9",
        channel="channel1",
        sampling_rate_hz=sampling_freq_Hz,
    ),
    lbs.DetectorInfo(
        name="channel1_w3_detB",
        wafer="wafer_3",
        channel="channel1",
        sampling_rate_hz=sampling_freq_Hz,
    ),
    lbs.DetectorInfo(
        name="channel1_w1_detC",
        wafer="wafer_1",
        channel="channel1",
        sampling_rate_hz=sampling_freq_Hz,
    ),
    lbs.DetectorInfo(
        name="channel1_w1_detD",
        wafer="wafer_1",
        channel="channel1",
        sampling_rate_hz=sampling_freq_Hz,
    ),
    lbs.DetectorInfo(
        name="channel2_w4_detA",
        wafer="wafer_4",
        channel="channel2",
        sampling_rate_hz=sampling_freq_Hz,
    ),
    lbs.DetectorInfo(
        name="channel2_w4_detB",
        wafer="wafer_4",
        channel="channel2",
        sampling_rate_hz=sampling_freq_Hz,
    ),
]


def test_detector_blocks(dets=dets, sampling_freq_Hz=sampling_freq_Hz):
    comm = lbs.MPI_COMM_WORLD

    start_time = 456
    duration_s = 100
    nobs_per_det = 3

    if comm.size > 4:
        det_blocks_attribute = ["channel", "wafer"]
    else:
        det_blocks_attribute = ["channel"]

    sim = lbs.Simulation(
        start_time=start_time,
        duration_s=duration_s,
        random_seed=12345,
        mpi_comm=comm,
    )

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
        num_of_obs_per_detector=nobs_per_det,
        det_blocks_attributes=det_blocks_attribute,
    )

    tod_len_per_det_per_proc = 0
    for obs in sim.observations:
        tod_shape = obs.tod.shape

        n_blocks_det = obs.n_blocks_det
        n_blocks_time = obs.n_blocks_time
        tod_len_per_det_per_proc += obs.tod.shape[1]

        # No testing required if the proc doesn't owns a detector
        if not any(idx is None for idx in obs.det_idx):
            det_names_per_obs = [
                obs.detectors_global[idx]["name"] for idx in obs.det_idx
            ]

            # Testing if the mapping between the obs.name and
            # obs.det_idx is consistent with obs.detectors_global
            np.testing.assert_equal(obs.name, det_names_per_obs)

            # Testing the distribution of the number of detectors per
            # detector block
            np.testing.assert_equal(obs.name.shape[0], tod_shape[0])

    # Testing if the distribution of samples along the time axis is consistent
    if comm.rank < n_blocks_det * n_blocks_time:
        arr = [
            span.num_of_elements
            for span in lbs.distribute.distribute_evenly(
                duration_s * sampling_freq_Hz, n_blocks_time * nobs_per_det
            )
        ]

        start_idx = (comm.rank % n_blocks_time) * nobs_per_det
        stop_idx = start_idx + nobs_per_det
        np.testing.assert_equal(sum(arr[start_idx:stop_idx]), tod_len_per_det_per_proc)


def test_rng_generators_with_detector_blocks():
    comm = lbs.MPI_COMM_WORLD
    size = comm.size

    ### Mission parameters
    telescope = "MFT"
    channel = "M1-195"
    detectors = [
        "001_002_030_00A_195_B",
        "001_002_029_45B_195_B",
        "001_002_015_15A_195_T",
        "001_002_047_00A_195_B",
    ]

    if size == 1:
        detectors = detectors[0:1]
        n_chunks = 1
    elif size == 2:
        detectors = detectors[0:2]
        n_chunks = 1
    elif size == 4:
        detectors = detectors[0:2]
        n_chunks = 2

    n_detectors = len(detectors)

    start_time = 51

    n_samples_pow = 2**13
    detector_sampling_freq = 4.1
    mission_time_days = n_samples_pow / detector_sampling_freq / 3600 / 24

    ### Simulation parameters
    random_seed = 45
    imo_version = "vPTEP"
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
    dtype_float = np.float64
    tmp_dir = tempfile.TemporaryDirectory()

    ### Detector list
    detector_list = []
    for n_det in detectors:
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/{imo_version}/satellite/{telescope}/{channel}/{n_det}/detector_info",
            imo=imo,
        )
        det.sampling_rate_hz = detector_sampling_freq
        det.fknee_mhz = 1e2
        det.fmin_hz = 1e-7
        det.alpha = 1.5
        detector_list.append(det)

    ### Initializing the simulation
    sim = lbs.Simulation(
        random_seed=random_seed,
        base_path=tmp_dir.name,
        name="brahmap_example",
        mpi_comm=comm,
        start_time=start_time,
        duration_s=mission_time_days * 24 * 60 * 60.0,
        imo=imo,
    )

    ### Create observations
    sim.create_observations(
        detectors=detector_list,
        num_of_obs_per_detector=1,
        n_blocks_det=n_detectors,  # FIXME: I want to split the dets too,
        n_blocks_time=n_chunks,  # Non-zero number of time blocks for example, should be 16
        split_list_over_processes=False,
        tod_dtype=dtype_float,
    )

    ### Adding 1/f noise
    lbs.noise.add_noise_to_observations(
        sim.observations, "one_over_f", dets_random=sim.dets_random
    )

    sim.add_noise(noise_type="one_over_f")


def test_mpi_subcommunicators(dets=dets):
    comm = lbs.MPI_COMM_WORLD

    start_time = 456
    duration_s = 100
    nobs_per_det = 3

    if comm.size > 4:
        det_blocks_attribute = ["channel", "wafer"]
    else:
        det_blocks_attribute = ["channel"]

    sim = lbs.Simulation(
        start_time=start_time,
        duration_s=duration_s,
        random_seed=12345,
        mpi_comm=comm,
    )

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
        num_of_obs_per_detector=nobs_per_det,
        det_blocks_attributes=det_blocks_attribute,
    )

    if lbs.MPI_COMM_GRID.COMM_OBS_GRID != lbs.MPI_COMM_GRID.COMM_NULL:
        # since unused MPI processes stay at the end of global,
        # communicator, the rank of the used processes in
        # `MPI_COMM_GRID.COMM_OBS_GRID` must be same as their rank in
        # global communicator
        np.testing.assert_equal(lbs.MPI_COMM_GRID.COMM_OBS_GRID.rank, comm.rank)

        for obs in sim.observations:
            # comm_det_block.rank + comm_time_block.rank * n_block_time
            # must be equal to the global communicator rank for the
            # used processes. It follows from the way split colors
            # were defined.
            np.testing.assert_equal(
                obs.comm_det_block.rank + obs.comm_time_block.rank * obs.n_blocks_time,
                comm.rank,
            )
    else:
        for obs in sim.observations:
            # the global rank of the unused MPI processes must be larger than the number of used processes.
            assert comm.rank > (obs.n_blocks_det * obs.n_blocks_time - 1)

            # The block communicators on the unused MPI processes must
            # be the NULL communicators
            np.testing.assert_equal(obs.comm_det_block, lbs.MPI_COMM_GRID.COMM_NULL)
            np.testing.assert_equal(obs.comm_time_block, lbs.MPI_COMM_GRID.COMM_NULL)


if __name__ == "__main__":
    pytest.main([f"{__file__}"])
