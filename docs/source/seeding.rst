RNG hierarchy
=============

The RNG hierarchy system in ``litebird_sim`` provides a reproducible and flexible
structure for managing random number generators across multiple levels of a simulation.
This is especially useful in MPI-parallel contexts where independent and deterministic
streams of randomness are required at different granularity levels—such as per-task or
per-detector.

The key classes and methods provided by this module are:

1. :class:`.RNGHierarchy`: A container managing a nested RNG tree from a single base seed.
2. :meth:`.RNGHierarchy.build_mpi_layer`: Method to build the first level of the RNG hierarchy for MPI tasks.
3. :meth:`.RNGHierarchy.build_detector_layer`: Method to build the second level of the RNG hierarchy for detectors.
4. :meth:`.RNGHierarchy.get_generator`: Method to get a speficic RNG generator from the hierarchy.
5. :meth:`.RNGHierarchy.get_detector_level_generators_on_rank`: Method to extract the full list of generators for a specific MPI rank.
6. :meth:`.RNGHierarchy.regenerate_or_check_detector_generators`: Method to check validity of a set of detector-level generators and eventually regenerate them from a new base seed.

The following example demonstrates how to create a layered RNG hierarchy:

.. code-block:: python

    import litebird_sim as lbs

    base_seed = 12345
    num_mpi_tasks = 2
    num_detectors = 3

    # Create a hierarchy of RNGs with two levels: task and detector
    rng_hierarchy = lbs.RNGHierarchy(base_seed)
    rng_hierarchy.build_mpi_layer(num_mpi_tasks)
    rng_hierarchy.build_detector_layer(num_detectors)

    # or directly with
    rng = lbs.RNGHierarchy(base_seed, num_mpi_tasks, num_detectors)

Once the :class:`.RNGHierarchy` is instanciated and RNGs are created, it is sufficient to use :meth:`.RNGHierarchy.get_detector_level_generators_on_rank` to obtain all the generators of a specific MPI rank. If needed, :meth:`.RNGHierarchy.get_generator` allows instead to retrieve a specific generator from the hierarchy.

.. code-block:: python

    rank = 0
    generators_of_first_rank = rng_hierarchy.get_detector_level_generators_on_rank(rank)

    indeces = (1, 2)
    generetor_second_rank_third_det = rng_hierarchy.get_generator(indeces)

If needed, :meth:`.RNGHierarchy.add_extra_layer` allows one to define a new level of the hierarchy from the current leaf layer (the detector layer in our case).

Finally, the :meth:`.RNGHierarchy.regenerate_or_check_detector_generators` allows the user to quickly regenerate a set of new detector-level generators, or to check the validity of the existing one. This is useful if the user passes a random seed specific for each systematics. This needs a list of :class:`.Observation` to work.

Saving and loading
------------------

RNG hierarchies can be serialized and deserialized via:

- :meth:`.RNGHierarchy.save`
- :meth:`.RNGHierarchy.load`

This is useful for ensuring reproducibility even across different simulation runs.

.. code-block:: python

    rng_hierarchy.save("my_rng_state.pkl")

    restored_rng_hierarchy = lbs.RNGHierarchy.load("my_rng_state.pkl")

    assert rng == restored_rng

API reference
-------------

.. automodule:: litebird_sim.rng_hierarchy
    :members:
    :undoc-members:
    :show-inheritance: