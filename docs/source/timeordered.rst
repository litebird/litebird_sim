.. _timeordered:

Time Ordered Simulations
========================

The LiteBIRD simulation framework intends to offer the capability to
simulate many effects in detector timestreams. This page will detail
these simulated effects as they become available. The timestreams are
stored in the ``tod`` field of :class:`.Observation` objects. These
arrays have shape ``(n_detectors, n_samples)``, where detectors are
indexed in the same order as the ``Observation.detector_global_info``
array. All the time streams inside the framework use Kelvin as the
default unit.


Filling TOD with signal
-----------------------

The framework provides :func:`.scan_map`, a routine which scans an
input map accordingly to the scanning strategy and fills the detector
timestreams. You can fill with signal an existing TOD by using the
function :func:`.scan_map_in_observations`, as the following example
shows:

.. testcode::
      
   import litebird_sim as lbs
   import numpy as np
   
   hwp_radpsec = np.pi / 8
   start_time_s = 0
   time_span_s = 1
   
   nside = 256
   npix = 12 * nside * nside
   
   # Create a simulation
   sim = lbs.Simulation(
       base_path="./output",
       start_time=start_time_s,
       duration_s=time_span_s,
   )
   
   # Define the scanning strategy
   sim.set_scanning_strategy(
       lbs.SpinningScanningStrategy(
           spin_sun_angle_rad=0.785_398_163_397_448_3,
           precession_rate_hz=8.664_850_513_998_931e-05,
           spin_rate_hz=0.000_833_333_333_333_333_4,
           start_time=start_time_s,
       ),
       delta_time_s=7200,
   )
   
   sim.set_instrument(
       lbs.InstrumentInfo(
           boresight_rotangle_rad=0.0,
           spin_boresight_angle_rad=0.872_664_625_997_164_8,
           spin_rotangle_rad=3.141_592_653_589_793,
       ),
   )
    
   # Create a detector object
   det = lbs.DetectorInfo(
       name="Detector",
       sampling_rate_hz=10,
       quat=[0.0, 0.0, 0.0, 1.0],
   )
   
   # Initialize the observation
   (obs,) = sim.create_observations(detectors=[det])
   
   # Compute the pointing information
   sim.compute_pointings()
   
   # Create a map to scan (in realistic simulations,
   # use the MBS module provided by litebird_sim)
   maps = np.ones((3, npix))
   in_map = {"Detector": maps, "Coordinates": lbs.CoordinateSystem.Ecliptic}

   # Here scan the map and fill tod
   lbs.scan_map_in_observations(
       obs, 
       in_map,
       input_map_in_galactic = False,
   )
   
   for i in range(obs.n_samples):
       # + 0. removes leading minus from negative zero
       value = np.round(obs.tod[0][i], 5) + 0.
       print(f"{value:.5f}")

.. testoutput::

    0.00000
    -0.00075
    -0.00151
    -0.00226
    -0.00301
    -0.00376
    -0.00451
    -0.00526
    -0.00601
    -0.00676   


The input maps to scan can be either included in a dictionary with the name of
the channel or the name of the dectector as keyword (the routines described in 
:ref:`Mbs` already provied the inputs in the correct format), or a numpy array
with shape (3, n_pixels).
The pointing information can be included in the observation or passed through 
`pointings`. If both `obs` and `pointings` are provided, they must be coherent,
so either a single Observation and a single numpy array, or same lenght list of
Observations and numpy arrays.
If the input map is ecliptic coordinates set `input_map_in_galactic` to `False`.
The effect of a possible HWP is included in the pointing information, see 
:ref:`scanning-strategy`.

Adding Noise
------------

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
   # by `sim`, which is always initialized with the
   # same seed to ensure repeatability.
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
own random number generator with a different seed.

To add white noise using a custom white noise sigma, in µK, we can
call the low level function directly:

.. testcode::

   import litebird_sim as lbs

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
   )

   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10,
   )

   obs = sim.create_observations(detectors=[det])

   custom_sigma_uk = 1234
   lbs.noise.add_white_noise(obs[0].tod[0], custom_sigma_uk)

We can also add 1/f noise using a very similar call to the above:

.. testcode::
   
   import litebird_sim as lbs

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
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
   lbs.noise.add_noise_to_observations(obs, 'one_over_f')

Again, to generate noise with custom parameters, we can either use the low-level function or edit the :class:`.Observation` object to contain the desired noise parameters. 

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
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
   )

   # Option 2: we change the values in `obs`
   obs[0].fknee_mhz[0] = custom_fknee_mhz
   obs[0].fmin_hz[0] = custom_fmin_hz
   obs[0].alpha[0] = custom_alpha
   obs[0].net_ukrts[0] = (
       custom_sigma_uk / np.sqrt(obs[0].sampling_rate_hz)
   )

   lbs.noise.add_noise_to_observations(obs, 'one_over_f')


Methods of class simulation
---------------------------

The class :class:`.Simulation` provides two simple functions that fill
with sky signal and nosie all the observations of a given simulation.
The function :func:`.Simulation.fill_tods` takes a map and scans it, while
the function :func:`.Simulation.add_noise` adds noise to the timelines.
Thanks to these functions the generation of a simulation becomes quite
transparent:

.. testcode::

  import litebird_sim as lbs
  from astropy.time import Time
  import numpy as np

  start_time = 0
  time_span_s = 1000.0
  sampling_hz = 10.0
  nside = 128

  sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

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

  sim.compute_pointings()

  sky_signal = np.ones((3,12*nside*nside))*1e-4

  sim.fill_tods(sky_signal)

  sim.add_noise(noise_type='white')

  for i in range(5):
      print(f"{sim.observations[0].tod[0][i]:.5e}")

.. testoutput::

    4.14241e-04
    5.46700e-05
    3.03378e-04
    6.13975e-05
    4.72613e-05
    

API reference
-------------

.. automodule:: litebird_sim.scan_map
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: litebird_sim.noise
   :members:
   :undoc-members:
   :show-inheritance:
