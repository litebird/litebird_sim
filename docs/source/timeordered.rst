.. _timeordered:

Time Ordered Simulations
========================

The LiteBIRD simulation framework offers or intends to offer the capability to 
simulate many effects in detector timestreams. This page will detail these 
simulated effects as they become available. The timestreams are stored in
`.Observation` objects in the `.Observation.tod` arrays. These arrays are of 
the shape (n_detectors, n_samples), where detectors are indexed in the same 
order as the `.Observation.detector_global_info` array. The timestreams inside the framework all use Kelvin as the default unit.


Adding Noise
------------

The ability to add noise to your detector timestreams is supported through the
:file: `.noise.py`. This file contains support for several different types of 
noise. Firstly, we can add white noise like this:

.. testcode::
   import litebird_sim as lbs

   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)
   #This will make us 100 seconds of noise

   det = lbs.DetectorInfo(
     fknee_mhz = 1.0,
     net_ukrts = 100,
     sampling_rate_hz = 10
   )#make a detector object, could also load from the imo
     
   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   lbs.noise.add_noise(obs, 'white')
   #here we add white noise using the detector noise parameters from the Imo

   print(obs[0].tod[0], len(obs[0].tod[0]))

.. testoutput::
   [-4.1348690e-06 -6.5450854e-06 -3.6649195e-05 ...  1.0698411e-05
 -3.7251013e-07  7.8516023e-06] 1900


To add white noise using a custom white noise sigma, in uK, we can call the low level
function directly:

.. testcode::

   import litebird_sim as lbs

   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)
   #This will make us 100 seconds of noise

   det = lbs.DetectorInfo(
     net_ukrts = 100,
     sampling_rate_hz = 10,
   )#in real code this would be read from the imo


   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   custom_sigma_uk = 1234

   lbs.noise.generate_white_noise(obs[0].tod[0], custom_sigma_uk)

We can also add 1/f noise using a very similar call to the above:

.. testcode::
   import litebird_sim as lbs

   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)
   #This will make us 100 seconds of noise

   det = lbs.DetectorInfo(
     net_ukrts = 100,
     sampling_rate_hz = 10,
     alpha = 1,
     fknee_mhz = 10
   )#in real code this would be read from the imo

   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   lbs.noise.add_noise(obs, 'one_over_f')
   #here we add 1/f noise using the detector noise parameters from the detector object

Again, to generate noise with custom parameters, we can either use the low level function directly, or edit the observation object to contain the desired noise parameters. 

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   sim = lbs.Simulation(base_path='./output', start_time=0, duration_s=100)
   #This will make us 100 seconds of noise

   det = lbs.DetectorInfo(
     net_ukrts = 100,
     sampling_rate_hz = 10,
     alpha = 1,
     fknee_mhz = 10
   )#in real code this would be read from the imo

   obs = sim.create_observations(detectors=[det], num_of_obs_per_detector=1)

   custom_sigma_uk = 1234
   custom_fknee_mhz = 12.34
   custom_alpha = 1.234

   #option 1, where we call the low lever function directly
   lbs.noise.generate_one_over_f_noise(obs[0].tod[0], custom_fknee_mhz, custom_alpha, custom_sigma_uk, obs[0].sampling_rate_hz)

   #option 2 where we change the values in the observation object
   obs[0].fknee_mhz[0] = custom_fknee_mhz
   obs[0].alpha[0] = custom_alpha
   obs[0].net_ukrthz[0] = custom_sigma_uk * np.sqrt(obs[0].sampling_rate_hz)

   lbs.noise.add_noise(obs, 'one_over_f')

Noise API reference
-------------------
.. automodule:: litebird_sim.noise
   :members:
   :undoc-members:
   :show-inheritance:
