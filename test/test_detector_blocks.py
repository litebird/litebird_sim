import numpy as np
import litebird_sim as lbs


def test_detector_blocks():
    comm = lbs.MPI_COMM_WORLD

    start_time = 456
    duration_s = 100
    sampling_freq_Hz = 1
    nobs_per_det = 3

    # Creating a list of detectors.
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
        det_blocks_attributes=["channel", "wafer"],
    )

    tod_len_per_det_per_proc = 0
    for obs in sim.observations:
        tod_shape = obs.tod.shape

        n_blocks_det = obs.n_blocks_det
        n_blocks_time = obs.n_blocks_time
        tod_len_per_det_per_proc += obs.tod.shape[1]

        # No testing required if the proc doesn't owns a detector
        if obs.det_idx is not None:
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