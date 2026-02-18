.. _interface_hierarchy:

Interface Hierarchy
====================

The ``litebird_sim``` framework, usually called LBS, is designed to be flexible and easy to use.
You can start with simple simulations right away, or build more advanced and customized workflows as you gain experience.
Its modular structure makes it easy for beginners to get started, while still giving advanced users full control over the details if they need it.

Core Classes
------------

At the heart of LBS there are two fundamental data types:

- :class:`.Simulation` acts as the global orchestrator of the simulation:

  - Parses configuration files

  - Manages the instrument model (IMo)

  - Tracks provenance and metadata

  - Controls MPI parallelism

  - Handles random number generators (RNGs)

  - Creates and manages observations

- :class:`.Observation` represents a single observational unit:

  - Manages TOD (time-ordered data) and pointing matrices

  - Stores detector-specific metadata

  - Controls MPI distribution and I/O

  - Acts as an interface to low-level computations

Interface Levels
----------------

LBS provides thousands of functions and class methods to create and manipulate the data used in simulations. These functionalities can be grouped in a hierarchy:

1. Low-level functions are usually implemented in Numba. They are very fast but not versatile, as they often work on one single detector/observation and expect inputs to have specific types, and thus are rarely exported to the user.

2. LBS exports several array-oriented functions that wrap the low-level Numba routines. An example is :func:`.add_dipole`.

3. Array-oriented functions often require the same parameters to be passed over and over again. For instance, a TOD matrix is required both by the functions that simulate the signal of the dipole (:func:`.add_dipole`) and by noise-generation modules (:func:`.add_noise`). Functions that match the pattern ``*_to_observations``, like :func:`.add_dipole_to_observations`, apply the same array-oriented functions to a list of observations.

4. In an MPI environment, a :class:`.Simulation` class creates multiple :class:`.Observation` instances and wraps functions in methods that automatically pass most of the relevant parameters. For instance, :func:`.add_dipole_to_observations` is wrapped by :meth:`.Simulation.add_dipole`, which requires far fewer parameters.

   Additionally, :class:`Simulation` offers ``set_*`` methods (e.g., :meth:`.Simulation.set_instrument`) that define shared parameters once and propagate them to subsequent function calls.

The following diagram illustrates the hierarchy of interfaces from highest-level control to the underlying computational core:

.. code-block:: text

    ┌──────────────────────┐
    │ High-Level Interface |
    └──────────────────────┘
            ↓
    ┌───────────────────┐
    │  Simulation class │
    └───────────────────┘
            ↓
    ┌───────────────────┐
    │ Observation class │
    └───────────────────┘
            ↓
    ┌──────────────────────────┐
    │ Array-oriented functions │
    └──────────────────────────┘
            ↓
    ┌──────────────────────────────────────────────┐
    │ Numba Core (Low-Level, usually not exported) │
    └──────────────────────────────────────────────┘

Simulation methods
~~~~~~~~~~~~~~~~~~

The high-level methods in the :class:`.Simulation` are ideal for most users, as they require minimal configuration. Here is an example:

.. code-block:: python

    sim = Simulation(config="config.toml")
    sim.create_observations(detectors=["LFT_1A", "LFT_1B"])
    sim.prepare_pointing()
    sim.add_dipole()

The simulation automatically takes care of TODs and pointing information.
Common tasks such as dipole generation, noise injection, and map-making can be performed directly through the simulation object, making it easy to build complete workflows with just a few method calls.

Observation-oriented functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users who need finer control, functions that follow the pattern ``*_to_observations`` enable working directly with individual :class:`.Observation` instances.

.. code-block:: python

    (obs,) = sim.create_observations(detectors=["LFT_1A"])
    obs.prepare_pointings(instrument=my_inst, spin2ecliptic_quats=my_quats)
    pointing_matrix, _ = obs.get_pointings()
    lbs.add_dipole_to_observations(obs, pointing=pointing_matrix)

This approach allows you to provide custom pointing data and configurations.
It is particularly useful when you want to run only part of the simulation pipeline or when you need to execute different stages manually and independently.

Array-based functions
~~~~~~~~~~~~~~~~~~~~~

Expert users who require full control over TOD arrays and performance tuning but do not need the complication of handling :class:`.Observation` instances can work directly with array-based functions.

.. code-block:: python

    pointing_matrix, _ = obs.get_pointings()
    lbs.add_dipole(obs.tod, pointing=pointing_matrix)

This level of access provides full control over array data and internal operations. It is especially useful for diagnostics, prototyping new features, or bypassing the built-in abstractions when needed.

Consistency Across Modules
--------------------------

The interface levels presented in this section are consistently implemented across all major simulation modules:

- Scanning of input maps, see here :ref:`mapscanning`,
- Beam simulation and convolution, see here :ref:`beamconvolution`,
- Dipole signal generation, see here :ref:`dipole-anisotropy`,
- Noise injection, see here :ref:`noise`,
- Map-making (binner, destriper, GLS), see here :ref:`mapmaking`,
- Input/output management, see here :ref:`simulations`.

This uniform interface hierarchy allows users to write generic tools while also enabling low-level extensions for performance or custom needs.
Further explanations are given in :ref:`high-level-vs-low-level-interfaces`
