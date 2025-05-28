import numpy as np
import pytest
import litebird_sim as lbs


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


def test_save_load_hierarchy_and_reproducibility(tmp_path):
    master_seed = 12345
    RNG_hierarchy = lbs.RNGHierarchy(master_seed)

    num_fake_mpi_tasks = 2
    RNG_hierarchy.build_mpi_layer(num_fake_mpi_tasks)

    num_fake_detectors = 3
    RNG_hierarchy.build_detector_layer(num_fake_detectors)

    RNG_hierarchy.save(tmp_path / "hierarchy.pkl")  # Save immediately

    loaded_RNG_hierarchy: lbs.RNGHierarchy = lbs.RNGHierarchy.from_saved_hierarchy(
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
        _ = lbs.RNGHierarchy.from_saved_hierarchy(tmp_path / "hierarchy.pkl")
