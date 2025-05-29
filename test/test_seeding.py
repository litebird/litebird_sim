from copy import deepcopy
import numpy as np
import pytest
import litebird_sim as lbs
from astropy.time import Time


def test_mpi_generators():
    master_seed = 12345
    RNG_hierarchy = lbs.RNGHierarchy(master_seed)

    num_fake_mpi_tasks = 4
    RNG_hierarchy.build_mpi_layer(num_fake_mpi_tasks)

    generators = [RNG_hierarchy.get_generator(idx) for idx in range(num_fake_mpi_tasks)]

    assert len(generators) == num_fake_mpi_tasks

    fake_realizations = np.array(
        [gen.normal(loc=0, scale=1, size=1000) for gen in generators]
    )

    for i in range(num_fake_mpi_tasks):
        for j in range(i + 1, num_fake_mpi_tasks):
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    fake_realizations[i], fake_realizations[j]
                )


def test_detector_generators():
    master_seed = 12345
    RNG_hierarchy = lbs.RNGHierarchy(master_seed)

    num_fake_mpi_tasks = 2
    RNG_hierarchy.build_mpi_layer(num_fake_mpi_tasks)

    num_fake_detectors = 3
    RNG_hierarchy.build_detector_layer(num_fake_detectors)

    generators = [
        RNG_hierarchy.get_detector_level_generators_on_rank(idx)
        for idx in range(num_fake_mpi_tasks)
    ]

    assert np.shape(generators) == (num_fake_mpi_tasks, num_fake_detectors)

    fake_realizations = np.empty((num_fake_mpi_tasks, num_fake_detectors, 1000))
    for mpi_idx in range(num_fake_mpi_tasks):
        for detector_idx in range(num_fake_detectors):
            real = generators[mpi_idx][detector_idx].normal(loc=0, scale=1, size=1000)
            fake_realizations[mpi_idx, detector_idx] = real

    for mpi_idx_i, mpi_idx_j, detector_idx_i, detector_idx_j in np.ndindex(
        num_fake_mpi_tasks, num_fake_mpi_tasks, num_fake_detectors, num_fake_detectors
    ):
        if detector_idx_j > detector_idx_i:
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    fake_realizations[mpi_idx_i, detector_idx_i],
                    fake_realizations[mpi_idx_j, detector_idx_j],
                )


def test_hierarchies_with_same_seed():
    master_seed = 12345

    RNG_hierarchy = lbs.RNGHierarchy(master_seed)
    num_fake_mpi_tasks = 2
    RNG_hierarchy.build_mpi_layer(num_fake_mpi_tasks)
    num_fake_detectors = 3
    RNG_hierarchy.build_detector_layer(num_fake_detectors)

    identical_RNG_hierarchy = lbs.RNGHierarchy(master_seed)
    num_fake_mpi_tasks = 2
    identical_RNG_hierarchy.build_mpi_layer(num_fake_mpi_tasks)
    num_fake_detectors = 3
    identical_RNG_hierarchy.build_detector_layer(num_fake_detectors)

    single_rng = deepcopy(RNG_hierarchy.get_generator(0, 2))
    identical_single_rng = deepcopy(RNG_hierarchy.get_generator(0, 2))
    assert single_rng.bit_generator.state == identical_single_rng.bit_generator.state

    generators = [
        RNG_hierarchy.get_detector_level_generators_on_rank(idx)
        for idx in range(num_fake_mpi_tasks)
    ]
    ref_realizations = np.empty((num_fake_mpi_tasks, num_fake_detectors, 1000))
    for mpi_idx in range(num_fake_mpi_tasks):
        for detector_idx in range(num_fake_detectors):
            real = generators[mpi_idx][detector_idx].normal(loc=0, scale=1, size=1000)
            ref_realizations[mpi_idx, detector_idx] = real

    generators = [
        identical_RNG_hierarchy.get_detector_level_generators_on_rank(idx)
        for idx in range(num_fake_mpi_tasks)
    ]
    same_realizations = np.empty((num_fake_mpi_tasks, num_fake_detectors, 1000))
    for mpi_idx in range(num_fake_mpi_tasks):
        for detector_idx in range(num_fake_detectors):
            real = generators[mpi_idx][detector_idx].normal(loc=0, scale=1, size=1000)
            same_realizations[mpi_idx, detector_idx] = real

    np.testing.assert_array_equal(
        ref_realizations,
        same_realizations,
    )


def test_save_load_hierarchy(tmp_path):
    master_seed = 12345
    RNG_hierarchy = lbs.RNGHierarchy(master_seed)

    num_fake_mpi_tasks = 2
    RNG_hierarchy.build_mpi_layer(num_fake_mpi_tasks)

    num_fake_detectors = 3
    RNG_hierarchy.build_detector_layer(num_fake_detectors)

    RNG_hierarchy.save(tmp_path / "hierarchy.pkl")  # Save immediately

    loaded_RNG_hierarchy: lbs.RNGHierarchy = lbs.RNGHierarchy.load(
        tmp_path / "hierarchy.pkl"
    )

    assert RNG_hierarchy == loaded_RNG_hierarchy, (
        "The loaded hierarchy is not the same as the original one"
    )

    ref_generators = [
        RNG_hierarchy.get_detector_level_generators_on_rank(idx)
        for idx in range(num_fake_mpi_tasks)
    ]

    assert np.shape(ref_generators) == (num_fake_mpi_tasks, num_fake_detectors)

    ref_realizations = np.empty((num_fake_mpi_tasks, num_fake_detectors, 1000))
    for mpi_idx in range(num_fake_mpi_tasks):
        for detector_idx in range(num_fake_detectors):
            real = ref_generators[mpi_idx][detector_idx].normal(
                loc=0, scale=1, size=1000
            )
            ref_realizations[mpi_idx, detector_idx] = real

    loaded_generators = [
        loaded_RNG_hierarchy.get_detector_level_generators_on_rank(idx)
        for idx in range(num_fake_mpi_tasks)
    ]

    assert np.shape(loaded_generators) == (num_fake_mpi_tasks, num_fake_detectors)

    loaded_realizations = np.empty((num_fake_mpi_tasks, num_fake_detectors, 1000))
    for mpi_idx in range(num_fake_mpi_tasks):
        for detector_idx in range(num_fake_detectors):
            real = loaded_generators[mpi_idx][detector_idx].normal(
                loc=0, scale=1, size=1000
            )
            loaded_realizations[mpi_idx, detector_idx] = real

    np.testing.assert_array_equal(
        ref_realizations,
        loaded_realizations,
        err_msg="Realizations from original and loaded hierarchies do not match.",
    )


def test_compatibility_error(tmp_path):
    master_seed = 12345
    RNG_hierarchy = lbs.RNGHierarchy(master_seed)

    num_fake_mpi_tasks = 2
    RNG_hierarchy.build_mpi_layer(num_fake_mpi_tasks)

    num_fake_detectors = 3
    RNG_hierarchy.build_detector_layer(num_fake_detectors)

    # Fake breaking-change in implementation
    RNG_hierarchy.SAVE_FORMAT_VERSION = 2

    RNG_hierarchy.save(tmp_path / "hierarchy.pkl")  # Save immediately

    with pytest.raises(ValueError):
        _ = lbs.RNGHierarchy.load(tmp_path / "hierarchy.pkl")


def test_detector_generators_regeneration(tmp_path):
    start_time = Time("2034-05-02")
    duration_s = 2 * 24 * 3600
    sampling_freq_Hz = 3

    dets = [
        lbs.DetectorInfo(
            name="det_A_wafer_1", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_1"
        ),
        lbs.DetectorInfo(
            name="det_B_wafer_1", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_1"
        ),
        lbs.DetectorInfo(
            name="det_C_wafer_2", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_2"
        ),
    ]

    sim = lbs.Simulation(
        base_path=tmp_path / "gd_lineargain_test",
        start_time=start_time,
        duration_s=duration_s,
        random_seed=12345,
    )

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
        num_of_obs_per_detector=1,
    )

    with pytest.raises(ValueError):
        _ = lbs.regenerate_or_check_detector_generators(
            observations=sim.observations,
            user_seed=None,
            dets_random=None,
        )
    with pytest.raises(AssertionError):
        _ = lbs.regenerate_or_check_detector_generators(
            observations=sim.observations,
            user_seed=None,
            dets_random=[sim.dets_random[0], sim.dets_random[1]],
        )
    # `user_seed` takes priority over the generators of the `Simulation``
    regenerated_dets_random = lbs.regenerate_or_check_detector_generators(
        observations=sim.observations,
        user_seed=987654321,
        dets_random=sim.dets_random,
    )

    rng_hierarchy = lbs.RNGHierarchy(
        987654321, num_ranks=1, num_detectors_per_rank=sim.observations[0].n_detectors
    )
    fresh_dets_random = rng_hierarchy.get_detector_level_generators_on_rank(0)

    for idx in range(len(dets)):
        assert (
            fresh_dets_random[idx].bit_generator.state
            == regenerated_dets_random[idx].bit_generator.state
        ), f"Generators at index {idx} are not the same."
