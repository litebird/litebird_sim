.. _timeordered:

Time Ordered Simulations
========================

The LiteBIRD simulation framework offers or intends to offer the capability to 
simulate many effects in detector timestreams. This page will detail these 
simulated effects as they become available. The timestreams are stored in
`.Observation` objects in the `.Observation.tod` arrays. These arrays are of 
the shape (n_detectors, n_samples), where detectors are indexed in the same 
order as the `.Observation.detector_global_info` array. The timestreams inside 
the framework all use Kelvin as the default unit.



Filling TOD with signal
-----------------------

The framework provides a basic routine which scans an input map accordingly to
the scanning strategy and fills the detector timestreams. This is supported
through the :file: `.scan_map.py`. You can fill with signal an existing TOD
by using the function :func:`.scan_map_in_observations`, as the following example
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
       base_path="./output", start_time=start_time_s, duration_s=time_span_s
   )
   
   # Create a detector object
   det = lbs.DetectorInfo(name="Detector", sampling_rate_hz=10, quat=[0.0, 0.0, 0.0, 1.0])
   
   # Define the scanning strategy
   scanning = lbs.SpinningScanningStrategy(
       spin_sun_angle_rad=0.785_398_163_397_448_3,
       precession_rate_hz=8.664_850_513_998_931e-05,
       spin_rate_hz=0.000_833_333_333_333_333_4,
       start_time=start_time_s,
   )
   
   spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
       start_time_s, time_span_s, delta_time_s=7200
   )
   
   instr = lbs.InstrumentInfo(
       boresight_rotangle_rad=0.0,
       spin_boresight_angle_rad=0.872_664_625_997_164_8,
       spin_rotangle_rad=3.141_592_653_589_793,
   )
   
   # Initialize the observation
   (obs,) = sim.create_observations(detectors=[det])
   
   # Compute the pointing
   pointings = lbs.scanning.get_pointings(
       obs,
       spin2ecliptic_quats=spin2ecliptic_quats,
       detector_quats=[det.quat],
       bore2spin_quat=instr.bore2spin_quat,
   )
   
   # Create a map to scan
   # In a realistic simulation use Mbs
   maps = np.ones((3, npix))
   in_map = {"Detector": maps}

   # Here scan the map and fill tod
   lbs.scan_map_in_observations(obs, pointings, hwp_radpsec, in_map)
   
   for i in range(obs.n_samples):
       # + 0. removes leading minus from negative zero
       value = np.round(obs.tod[0][i], 5) + 0.
       print(f"{value:.5f}")

.. testoutput::

   0.00000
   -0.14475
   -0.26104
   -0.34598
   -0.39746
   -0.41420
   -0.39579
   -0.34267
   -0.25618
   -0.13846


The input maps to scan must be included in a dictionary with either the name of
the channel or the name of the dectector as keyword. The routines described in 
:ref:`Mbs` already provied the inputs in the correct format. 
When set `True` the option `fill_psi_and_pixind_in_obs` fills the polarization
angle `obs.psi` and the pixel index `obs.pixind` for each sample, allowing to 
quickly bin the Observations through the function :func:`.make_bin_map`.


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
   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)

   # Create a detector object
   det = lbs.DetectorInfo(
     fknee_mhz=1.0,
     net_ukrts=100,
     sampling_rate_hz=10
   )
     
   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   # Here we add white noise using the detector noise parameters from the Imo
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

   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)

   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10,
   )

   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   custom_sigma_uk = 1234
   lbs.noise.add_white_noise(obs[0].tod[0], custom_sigma_uk)

We can also add 1/f noise using a very similar call to the above:

.. testcode::
   
   import litebird_sim as lbs

   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)

   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10,
     alpha=1,
     fknee_mhz=10
   )

   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   # Here we add 1/f noise using the detector noise parameters from the
   # detector object
   lbs.noise.add_noise_to_observations(obs, 'one_over_f')

Again, to generate noise with custom parameters, we can either use the low level function directly, or edit the observation object to contain the desired noise parameters. 

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)

   det = lbs.DetectorInfo(
     net_ukrts=100,
     sampling_rate_hz=10,
     alpha=1,
     fknee_mhz=10
   )

   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   custom_sigma_uk = 1234
   custom_fknee_mhz = 12.34
   custom_alpha = 1.234

   # Option 1, where we call the low lever function directly
   lbs.noise.add_one_over_f_noise(
       obs[0].tod[0],
       custom_fknee_mhz,
       custom_alpha,
       custom_sigma_uk,
       obs[0].sampling_rate_hz,
   )

   # Option 2, where we change the values in the observation object
   obs[0].fknee_mhz[0] = custom_fknee_mhz
   obs[0].alpha[0] = custom_alpha
   obs[0].net_ukrts[0] = custom_sigma_uk / np.sqrt(obs[0].sampling_rate_hz)

   lbs.noise.add_noise_to_observations(obs, 'one_over_f')

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
