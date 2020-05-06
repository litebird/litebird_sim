Simulations
===========

The LiteBIRD Simulation Framework is built on the :class:`.Simulation`
class, which should be instantiated in any pipeline built using this
framework. The class acts as a container for the many analysis modules
available to the user, and it offers the following features:

1. Provenance model;
2. Interface with the instrument database;
3. System abstractions;
4. Generation of reports.

Provenance model
----------------

A «provenance model» is, generally speaking, a way to track the
history and origin of a data set by recording the following
information:

1. Who or what created the dataset?
2. Which algorithm or instrumentation was used to produce it?
3. Which steps were undertaken to process the raw data?
4. How can one get access to the raw samples used to produce the
   dataset?

The LiteBIRD Simulation Framework tracks these information using
parameter files (in TOML format) and generating reports at the end of
a simulation.


Interface with the instrument database
--------------------------------------

To simulation LiteBIRD's data acquisition, the simulation code must be
aware of the characteristics of the instrument. These are specified in
the LiteBIRD Instrument Model (IMO) database, which can be accessed by
people with sufficient rights. This Simulation Framework has the
ability to access the database and take the input parameters necessary
for its analysis modules to produce the expected output.

   
System abstractions
-------------------

In some cases, simulations must be ran on HPC computers, distributing
the job on many processing units; in other cases, a simple laptop
might be enough. The LiteBIRD Simulation Framework uses MPI to
parallelize its codes, which is however an optional dependency: the
code can be ran serially.

When creating a :class:`.Simulation` object, the user can tell the
framework to use or not MPI using the flag `use_mpi`::

  import litebird_sim as lbs

  # This simulation must be ran using MPI
  sim = lbs.Simulation(use_mpi = True)

The framework sets a number of variables related to MPI; these
variables are *always* defined, even if MPI is not available, and they
can be used to make the code work in different situations. If your
code must be able to run both with and without MPI, you should
initialize a :class:`.Simulation` object using the variable
:class:`.HAVE_MPI4PY`::

  import litebird_sim as lbs

  # This simulation can take advantage of MPI, if present
  sim = lbs.Simulation(use_mpi = lbs.HAVE_MPI4PY)
         
See the page :ref:`using_mpi` for more information.


Generation of reports
---------------------

This section should explain how reports can be generated, first from
the perspective of a library user, and then describing how developers
can generate plots for their own modules.

Here is an example, showing several advanced topics like mathematical
formulae, plots, and value substitution::

    import litebird_sim as lbs
    import matplotlib.pylab as plt

    sim = lbs.Simulation(name="My simulation", base_path="output")
    data_points = [0, 1, 2, 3]

    plt.plot(data_points)
    fig = plt.gcf()

    sim.append_to_report('''
    Here is a formula for $`f(x)`$:

    ```math
    f(x) = \sin x
    ```

    And here is a completely unrelated plot:

    ![](myplot.png)

    The data points have the following values:
    {% for sample in data_points %}
    - {{ sample }}
    {% endfor %}
    ''', figures=[(fig, "myplot.png")],
         data_points=data_points)

    sim.flush()


API reference
-------------

.. automodule:: litebird_sim.simulations
    :members:
    :undoc-members:
    :show-inheritance:
