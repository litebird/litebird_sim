Integrating existing codes
==========================

In this section, we provide details about how to integrate existing
codes in the LiteBIRD Simulation Framework.

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

Data analysis codes can be integrated, provided that they help in
producing simulated measurements of the sky signal viable for the kind
of analysis done by the Joint Study Groups (JSGs). This means that the
following codes can be integrated, even if the are analysis codes
rather than simulation codes:

1. Map-making;
2. Component separation;
3. Etc.

The code should be available as Python modules that can be installed
using ``pip``; therefore, they should be available on the `Python
Package Index <https://pypi.org/>`_.

How to integrate them
---------------------

External libraries should not be copied in the `litebird_sim`
repository, unless there is some very strong reason to do so. Instead,
the library should be added to the list of dependencies using the
command ``poetry add LIBRARY_NAME``.

Once the library is available as a dependency, the author must
implement a class that wraps the functionality of the library and
accesses input/output parameters through the :class:`.Simulation`
class. Writing this wrapper ensures the following properties:

1. Input parameters can be easily read from the IMO;
2. Output files are saved in the directory specified by the user for
   the simulation;
3. The wrapper can provide functions to write automatic reports.

Imagine you have developed a robust noise
generation module, which has the following structure::

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
          # section named "noise_generation"
          self.seed = self.sim.query_input_parameter(
              "noise_generation",
              "seed",
              type=int,
              default=1234,
          )
          
          # Initialize your code
          self.generator = mynoisegen.NoiseGenerator(seed)

      def run(self):
          noise = self.generator.generate_noise()

          self.sim.append_to_report("""
          # Noise generation
          
          The noise generator produced {num_of_points} points,
          with an average {avg} and a standard deviation {sd}.
          """,
              num_of_points=len(noise),
              avg=np.mean(noise),
              sd=np.std(noise),
          )

          # Now use "noise" somehow!
          ...

The interface implements the following features, that were missing
in the class ``NoiseGenerator``:

- It loads the seed of the generator from the parameter file passed
  by the user; the noise generator is likely to be used in a wider
  pipeline, and this ensures that parameters to ``NoiseGenerator``
  can be kept with any other input parameter. The TOML parameter
  file could be the following:

  .. code-block:: text
                
    [noise_generation]
    seed = 6343
  
    [scanning_strategy]
    parameters = imo://v1.0/scanning_strategy
  
    [map_maker]
    nside = 512

  The call to `sim.query_input_parameter` in the example above would
  retrieve the number ``6343``. Note that the wrapper class does not
  need to deal with the other sections in the file
  (``scanning_strategy``, ``map_maker``): they are handled by other
  modules in the pipeline. See :ref:`parameter_files`.
  
- It produces a section in the report output by the framework, which
  contains some statistics about the generated noise (number of
  samples, average, standard deviation). See :ref:`report-generation`.
   
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
   measurement units should be used for `tant`, nor what is the
   measurement unit of the value returned by the function. The
   following is much better::

     def calc_sensitivity_k_sqr_s(t_ant_k):
         # The same calculations as above
         return f(t_ant, whatever...)

   The second definition clarifies that the antenna temperature must
   be specified in Kelvin, and that the result is in K⋅√s.

4. If you want to produce logging message, rely on the `logging
   library <https://docs.python.org/3/library/logging.html>`_ in the
   Python standard library.

5. If you are unsure about your python coding practices, the 
   `Google style guide
   <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>`_
   is a good resource. TL;DR. Then run a static analyzer on your code,
   like `Flake8 <https://pypi.org/project/flake8>`_.
   See also our `CONTRIBUTING file
   <https://github.com/litebird/litebird_sim/blob/master/CONTRIBUTING.md>`_
