Integrating existing codes
==========================

In this section, we provide details about how to integrate existing
codes in the LiteBIRD Simulation Framework, be they general-purpose
libraries or modules specifically developed for LiteBIRD.

What kind of codes can be integrated
------------------------------------

The LiteBIRD simulation framework provides a set of tools to simulate
the stages of data acquisition of the instrument onboard the
spacecraft. Therefore, any code that helps in producing synthetic
sample output can be integrated in the framework, in principle.
Examples of codes are:

1. Simulation of ADC non-linearities;
2. Injection of gain drifts;
3. Convolution of the sky signal with beam functions;
4. Generation of noise;
5. Et cetera.

Data *analysis* codes (in opposition to *simulation* codes producing
synthetic data) can be integrated, provided that they help in
producing larger simulations for the kind of analysis done by the
Joint Study Groups (JSGs). This means that the following codes can be
integrated, even if the are analysis codes rather than simulation
codes:

1. Map-making;
2. Component separation;
3. Etc.

If you plan to import a general-purpose library, the code should be
available as Python modules that can be installed using ``pip``;
therefore, they should be available on the `Python Package Index
<https://pypi.org/>`_. On the other hand, if the code you want to
integrate is specific to LiteBIRD, it can probably be integrated
directly into the `litebird_sim` codebase.

How to integrate codes
----------------------

General-purpose libraries that are installable using ``pip`` should
not be copied in the `litebird_sim` repository, unless there is some
strong reason to do so; instead, the library should be added to the
list of dependencies using the command ``uv add LIBRARY_NAME``. If
the library is specific to LiteBIRD, it is probably better to import
it into the repository https://github.com/litebird/litebird_sim.

Once the code is available, either as a dependency or as some code in
the repository, the author must implement a class that wraps the
functionality of the library and accesses input/output parameters
through the :class:`.Simulation` class. Writing this wrapper ensures
the following properties:

1. Input parameters can be easily read from the IMO;
2. Output files are saved in the directory specified by the user for
   the simulation;
3. The wrapper can provide functions to write automatic reports.

If you are writing some code from scratch for `litebird_sim`, it is
advisable to implement this approach and split it into a low-level
part that performs all the calculations, and a high-level part that
takes its data using the :class:`.Simulation` class. In this way, the
low-level part can be called directly from the terminal without the
hassle to create a :class:`.Simulation` object, which is easier for
debugging.

Finally, you should add some automatic tests and put them in the
folder ``litebird_sim/test`` in files whose name matches the pattern
``test_*.py``: in this way, they will be called automatically after
every commit and will ensure that your code works as expected on the
target machine.


A practical example
-------------------
   
Imagine you have developed a robust noise generation module, which has
the following structure::

     # File mynoisegen.py

     class NoiseGenerator:
         def __init__(self, seed):
             self.seed = seed

         def generate_noise(self):
             # Generate some noise
             return ...

To integrate this module in the LiteBIRD Simulation Framework, you
might want to write a wrapper class to ``NoiseGenerator`` that has an
interface of this kind::

  import litebird_sim as lbs
  import numpy as np
  import mynoisegen   # This is the library we're integrating
  
  class MyWrapper:
      def __init__(self, sim: lbs.Simulation):
          self.sim = sim

          # To decide which seed to pass to "NoiseGenerator",
          # take advantage of the "sim" object. Here we
          # assume to load it using the "seed" key in a
          # section named "noise"
          self.seed = self.sim.parameters["noise"].get("seed", 1234)
          
          # Initialize your code
          self.generator = mynoisegen.NoiseGenerator(seed)

      def run(self):
          # Call the function doing the *real* work
          self.noise = self.generator.generate_noise()

          self.sim.append_to_report("""
          # Noise generation
          
          The noise generator produced {num_of_points} points,
          with an average {avg} and a standard deviation {sd}.
          """,
              num_of_points=len(self.noise),
              avg=np.mean(self.noise),
              sd=np.std(self.noise),
          )

          # Now use "self.noise" somehow!
          ...

The interface implements the following features, which were missing in
the class ``NoiseGenerator``:

- It loads the seed of the generator from the parameter file passed by
  the user; the noise generator is likely to be used in a wider
  pipeline, and this ensures that parameters to ``NoiseGenerator`` can
  be kept with any other input parameter. The TOML parameter file
  could be the following:

  .. code-block:: text
                
    [noise]
    seed = 6343
  
    [scanning_strategy]
    parameters = "/releases/v1.3/Satellite/scanning_parameters/"
  
    [map_maker]
    nside = 512

  The code above accesses the field ``sim.parameters``, which is a
  Python dictionary containing the parsed content of the TOML file;
  the call to the standard `get` method ensures that a default
  value (1234) is used if parameter ``seed`` is not found in the TOML
  file, but in the example above it would retrieve the number
  ``6343``. Note that the wrapper class does not need to deal with the
  other sections in the file (``scanning_strategy``, ``map_maker``):
  they are handled by other modules in the pipeline. See
  :ref:`parameter_files`.
  
- It produces a section in the report output by the framework, which
  contains some statistics about the generated noise (number of
  samples, average, standard deviation). See :ref:`report-generation`.

Finally, we must add some tests. If the ``NoiseGenerator`` class is
expected to produce zero-mean output, then you might check that this
is indeed the case::

  # Save this in file litebird_sim/test/test_noise_generator.py

  import numpy as np
  import litebird_sim as lbs

  def test_noise_generator():
      sim = lbs.Simulation(random_seed=12345)
      noisegen = MyWrapper(sim)
      noisegen.run()

      # Of course, in real-life codes you would implement a
      # much more robust check here…
      assert np.abs(np.mean(noisegen.noise)) < 1e-5

  
Checklist
---------

Here we list what any developer should check before integrating their
codes in the LiteBIRD Simulation Framework:

1. You must not leave sensitive information in the code (e.g.,
   hardcoded noise levels): anything related to a quantitative
   description of the instrument should be loaded from parameter files
   or from the Instrument Model database. The best way to do this is
   to delegate the loading of input parameters in a wrapper class that
   uses a :class:`.Simulation` object (see above).
             
2. All the *public* functions should be documented, either using
   docstrings or other tools. You can put most of your effort in
   documenting the wrapper class (in the example above,
   ``MyWrapper``), as this is the public interface most of the people
   will use. Prefer the 
   `numpy sphinx syntax
   <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

3. All the measurement units should be stated clearly, possibly in
   parameter/variable/function names. Consider the following
   function::

     def calc_sensitivity(t_ant):
         # Some very complex calculation comes here
         return f(t_ant, whatever...)

   The prototype does not help the user to understand what kind of
   measurement units should be used for ``t_ant``, nor what is the
   measurement unit of the value returned by the function. The
   following is much better::

     def calc_sensitivity_k_sqr_s(t_ant_k):
         # The same calculations as above
         return f(t_ant_k, whatever...)

   The second definition clarifies that the antenna temperature must
   be specified in Kelvin, and that the result is in K⋅√s.

4. If you want to produce logging message, rely on the `logging
   library <https://docs.python.org/3/library/logging.html>`_ in the
   Python standard library.

5. You **must** format your code using `ruff
   <https://github.com/astral-sh/ruff/>`_. If you fail to do so,
   your code cannot be merged in the framework, as we automatically
   check its conformance every time a new pull request is opened.

6. Similarly, your code must pass all the tests run by `ruff check`.

7. Always implement some tests!

8. If you are unsure about your python coding practices, the `Google
   style guide
   <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>`_
   is a good resource. See also our `CONTRIBUTING file
   <https://github.com/litebird/litebird_sim/blob/master/CONTRIBUTING.md>`_
