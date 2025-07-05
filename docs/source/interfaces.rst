Interface Hierarchy
====================

The ``litebird_sim`` framework provides a tiered interface architecture, designed to support workflows ranging from simple simulations to advanced custom pipelines. This modular design ensures that new users can quickly build simulations, while advanced users can customize or optimize nearly every detail.

Core Classes
------------

At the heart of the framework are two primary classes:

**Simulation**

    Acts as the global orchestrator of the simulation:

    - Parses configuration files
    - Manages the instrument model (IMo)
    - Tracks provenance and metadata
    - Controls MPI parallelism
    - Handles random number generators (RNGs)
    - Creates and manages observations

**Observation**

    Represents a single observational unit:

    - Manages TODs and pointing
    - Stores detector-specific metadata
    - Controls MPI distribution and I/O
    - Acts as an interface to low-level computations

Interface Levels
----------------

::

    High-Level Interface
        ↓
    Simulation class
        ↓
    Observation class
        ↓
    TODs / Pointing Arrays
        ↓
    Numba Core (Low-Level)

High-Level Interface
~~~~~~~~~~~~~~~~~~~~

Ideal for most users. Operates through the ``Simulation`` object and requires minimal configuration.

.. code-block:: python

    sim = Simulation(config="config.toml")
    sim.create_observations(detectors=["LFT_1A", "LFT_1B"])
    sim.prepare_pointing()
    sim.add_dipole()

- Pointing and TODs are handled automatically.
- Modules such as dipole generation, noise injection, and map-making can be invoked directly on the simulation.

Intermediate-Level Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides finer control by operating on ``Observation`` instances directly.

.. code-block:: python

    (obs,) = sim.create_observations(detectors=["LFT_1A"])
    obs.prepare_pointings(instrument=my_inst, spin2ecliptic_quats=my_quats)
    pointing_matrix, _ = obs.get_pointings()
    lbs.add_dipole_to_observations(obs, pointing=pointing_matrix)

- Custom pointing and configuration possible.
- Enables partial or staged execution of the simulation pipeline.

Low-Level Interface
~~~~~~~~~~~~~~~~~~~

For expert users who need full control over TOD arrays and performance tuning.

.. code-block:: python

    pointing_matrix, _ = obs.get_pointings()
    lbs.add_dipole(obs.tod, pointing=pointing_matrix)

- Direct access to arrays and low-level operations.
- Useful for diagnostics, prototyping, or bypassing built-in abstractions.

Consistency Across Modules
--------------------------

The three interface levels are consistently implemented across all major simulation modules:

- Scanning of input maps, see here :ref:`mapscanning`,
- Beam simulation and convolution, see here :ref:`beamconvolution`,
- Dipole signal generation, see here :ref:`dipole-anisotropy`,
- Noise injection, see here :ref:`noise`,
- Map-making (binner, destriper, GLS), see here :ref:`mapmaking`,
- Input/output management, see here :ref:`simulations`.

This uniform interface hierarchy allows users to write generic tools while also enabling low-level extensions for performance or custom needs.
