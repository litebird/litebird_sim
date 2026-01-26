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
   lbs.noise.add_noise_to_observations(obs, 'white', dets_random=sim.dets_random)

   for i in range(10):
       print(f"{obs[0].tod[0][i]:.5e}")

.. testoutput::

    -1.37982e-04
    3.65642e-04
    2.47778e-04
    1.78779e-04
    -5.03410e-05
    4.21404e-04
    5.90033e-04
    5.07347e-04
    -9.98478e-05
    -5.19765e-05


Note that we pass ``sim.dets_random`` as the detector-level random
number generators to use. This is a list member variable that is
initialized by the constructor of the class :class:`.Simulation`,
and it is safe to be used with multiple MPI processes as it ensures
that each detector has its own random number generator with a
different seed. You can also pass another list of random number
generators, as long as each has the ``normal`` method. More
information on the generation of random numbers can be found in
:ref:`random-numbers`.

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
   lbs.noise.add_white_noise(obs[0].tod[0], custom_sigma_uk, random=sim.dets_random[0])

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
   lbs.noise.add_noise_to_observations(obs, 'one_over_f', dets_random=sim.dets_random)

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
       sim.dets_random[0],
   )

   # Option 2: we change the values in `obs`
   obs[0].fknee_mhz[0] = custom_fknee_mhz
   obs[0].fmin_hz[0] = custom_fmin_hz
   obs[0].alpha[0] = custom_alpha
   obs[0].net_ukrts[0] = (
       custom_sigma_uk / np.sqrt(obs[0].sampling_rate_hz)
   )

   lbs.noise.add_noise_to_observations(obs, 'one_over_f', dets_random=sim.dets_random)


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
:func:`.Simulation.add_noise` which adds noise to the timelines.
All the details of the noise are provided in the class observation and
the interface is simplified. 

.. testcode::

  import litebird_sim as lbs
  from astropy.time import Time
  import numpy as np

  start_time = 0
  time_span_s = 1000.0
  sampling_hz = 10.0
  nside = 128

  sim = lbs.Simulation(
      start_time=start_time,
      duration_s=time_span_s,
      random_seed=12345,
  )

  # We pick a simple scanning strategy where the spin axis is aligned
  # with the Sun-Earth axis, and the spacecraft spins once every minute
  sim.set_scanning_strategy(
      lbs.SpinningScanningStrategy(
          spin_sun_angle_rad=np.deg2rad(0),
          precession_rate_hz=0,
          spin_rate_hz=1 / 60,
          start_time=start_time,
      ),
      delta_time_s=5.0,
   )

  # We simulate an instrument whose boresight is perpendicular to
  # the spin axis.
  sim.set_instrument(
      lbs.InstrumentInfo(
          boresight_rotangle_rad=0.0,
          spin_boresight_angle_rad=np.deg2rad(90),
          spin_rotangle_rad=np.deg2rad(75),
      )
  )

  # A simple detector looking along the boresight direction
  det = lbs.DetectorInfo(
      name="Boresight_detector",
      sampling_rate_hz=sampling_hz,
      bandcenter_ghz=100.0,
      net_ukrts=50.0,
  )

  sim.create_observations(detectors=det)

  sim.add_noise(noise_type='one_over_f')

  for i in range(5):
      print(f"{sim.observations[0].tod[0][i]:.5e}")

.. testoutput::

    -6.90763e-05
    1.82736e-04
    1.23804e-04
    8.93039e-05
    -2.52559e-05

API reference
-------------

.. automodule:: litebird_sim.noise
   :members:
   :undoc-members:
   :show-inheritance:
