.. _noise:

Instrumental noise
==================

The ability to add noise to your detector timestreams is supported through the
function :func:`.add_noise_to_observations` and the low-level versions
:func:`.add_noise`, :func:`.add_white_noise`, and
:func:`.add_one_over_f_noise`.

Here is a short example that shows how to add noise:

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   # Create a simulation lasting 100 seconds
   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
       random_seed=12345,
   )

   # Create a detector object
   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10
   )

   obs = sim.create_observations(detectors=[det])

   # Here we add white noise using the detector
   # noise parameters from the `det` object.
   # We use the random number generator provided
   # by `sim`, which is initialized with the
   # seed we passed to the Simulation constructor
   # to ensure repeatability.
   lbs.noise.add_noise_to_observations(obs, 'white', random=sim.random)

   for i in range(10):
       print(f"{obs[0].tod[0][i]:.5e}")

.. testoutput::

    5.65263e-04
    -1.54522e-04
    3.42276e-04
    -1.42274e-04
    -1.71110e-04
    8.72188e-05
    -1.23400e-04
    -6.99311e-05
    6.58389e-05
    5.51306e-04


Note that we pass ``sim.random`` as the number generator to use.
This is a member variable that is initialized by the constructor
of the class :class:`.Simulation`, and it is safe to be used with
multiple MPI processes as it ensures that each process has its
own random number generator with a different seed. You can also
pass another random number generator, as long as it has the
``normal`` method. More information on the generation of random
numbers can be found in :ref:`random-numbers`.

To add white noise using a custom white noise sigma, in µK, we can
call the low level function directly:

.. testcode::

   import litebird_sim as lbs

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
       random_seed=12345,
   )

   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10,
   )

   obs = sim.create_observations(detectors=[det])

   custom_sigma_uk = 1234
   lbs.noise.add_white_noise(obs[0].tod[0], custom_sigma_uk, random=sim.random)

We can also add 1/f noise using a very similar call to the above:

.. testcode::

   import litebird_sim as lbs

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
       random_seed=12345,
   )

   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10,
     alpha=1,
     fknee_mhz=10
   )

   obs = sim.create_observations(detectors=[det])

   # Here we add 1/f noise using the detector noise
   # parameters from the detector object
   lbs.noise.add_noise_to_observations(obs, 'one_over_f', random=sim.random)

Again, to generate noise with custom parameters, we can either use the low-level function or edit the :class:`.Observation` object to contain the desired noise parameters.

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
       random_seed=12345,
   )

   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10,
     alpha=1,
     fknee_mhz=10,
     fmin_hz=0.001,
   )

   obs = sim.create_observations(detectors=[det])

   custom_sigma_uk = 1234
   custom_fknee_mhz = 12.34
   custom_alpha = 1.234
   custom_fmin_hz = 0.0123

   # Option 1: we call the low-level function directly
   lbs.noise.add_one_over_f_noise(
       obs[0].tod[0],
       custom_fknee_mhz,
       custom_fmin_hz,
       custom_alpha,
       custom_sigma_uk,
       obs[0].sampling_rate_hz,
       sim.random,
   )

   # Option 2: we change the values in `obs`
   obs[0].fknee_mhz[0] = custom_fknee_mhz
   obs[0].fmin_hz[0] = custom_fmin_hz
   obs[0].alpha[0] = custom_alpha
   obs[0].net_ukrts[0] = (
       custom_sigma_uk / np.sqrt(obs[0].sampling_rate_hz)
   )

   lbs.noise.add_noise_to_observations(obs, 'one_over_f', random=sim.random)


.. warning::

    It's crucial to grasp the distinction between the noise level in a
    timestream and the noise level in a map. While the latter is
    dependent on the former, the conversion is influenced by several
    factors. This understanding will empower you in your data analysis
    tasks.

    A common mistake is to use the mission time divided by the number
    of pixels in the map in a call to func:`.add_white_noise`. This is
    **wrong**, as the noise level per pixel depends on the overall
    integration time, which is always less than the mission time
    because of cosmic ray loss, repointing maneuvers, etc. These
    effects reduce the number of samples in the timeline that can be
    used to estimate the map, but they do not affect the noise of the
    timeline.


Methods of the Simulation class
-------------------------------

The class :class:`.Simulation` provides the function
:func:`.Simulation.add_noise` adds noise to the timelines.

API reference
-------------

.. automodule:: litebird_sim.noise
   :members:
   :undoc-members:
   :show-inheritance:
