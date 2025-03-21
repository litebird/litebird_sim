.. _mapscanning:

Scanning a map to fill a TOD
============================

The framework provides :func:`.scan_map`, a routine which scans an
input map accordingly to the scanning strategy and fills the detector
timestreams. It implements three possible algebras:

- No HWP:

.. math:: 
   d_t = T + \gamma \left(Q\cos(2\theta + 2\psi_t)+U\sin(2\theta + 2\psi_t)\right),
   :label: noHWP

where :math:`\theta` is the polarization angle of the detecotor, :math:`\psi_t` 
is the orientation of the telescope at the time :math:`t`, and :math:`\gamma`
is the polarization efficiency.

- Ideal HWP:

.. math::
   d_t = T + \gamma \left(Q\cos(4\alpha_t - 2\theta + 2\psi_t)+U\sin(4\alpha_t - 2\theta + 2\psi_t)\right),
   :label: idealHWP

where :math:`\alpha_t` is the HWP angle at the time :math:`t`.

- Generic HWP:

.. math::
   d_t = (1,0,0,0) \times M_{\rm pol} \times R(\theta) \times R^T(\alpha_t) \times M_{\rm HWP} \times R(\alpha_t) \times R(\psi_t) \times \vec{S}
   :label: genericHWP

where 
    * :math:`M_{\rm pol}` is mueller matrix of the polarimeter;
    * :math:`M_{\rm HWP}` is mueller matrix of the HWP;
    * :math:`R` is rotation matrix;
    * :math:`\vec{S}` is the Stokes vector.

You can fill with signal an existing TOD by using the
function :func:`.scan_map_in_observations`, as the following example
shows:

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   start_time_s = 0
   time_span_s = 1

   nside = 256
   npix = 12 * nside * nside

   # Create a simulation
   sim = lbs.Simulation(
       base_path="./output",
       start_time=start_time_s,
       duration_s=time_span_s,
       random_seed=12345,
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

   # Prepare the quaternions used to compute the pointings
   sim.prepare_pointings()

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

The code automatically selects the fastest algebra based on the provided HWP.

The input maps to scan can be either included in a dictionary with the name of
the channel or the name of the dectector as keyword (the routines described in
:ref:`Mbs` already provied the inputs in the correct format), or a numpy array
with shape (3, n_pixels).

The pointing information can be included in the observation or passed through
`pointings`. If both `observations` and `pointings` are provided, they must be 
coherent, so either a single Observation and a single numpy array, or same
lenght list of Observations and numpy arrays.
If the input map is ecliptic coordinates set `input_map_in_galactic` to `False`.
The effect of a possible HWP is included in the pointing information, see
:ref:`scanning-strategy`, the polarization angle of the detectors is taken from 
the corresponding attributes included in the observations. The same applies to 
the polarization efficiency. 

The routine provides an on-the-fly interpolation of the input maps. This option
is available through the argument `interpolation` which specifies the type of TOD
interpolation ("" for no interpolation, "linear" for linear interpolation).
Default: no interpolation.

The low level function, :func:`.scan_map`, allows a more refined handling of the
inputs. 

Methods of the Simulation class
-------------------------------

The class :class:`.Simulation` provides the function
:func:`.Simulation.fill_tods`, which takes a map and scans it. Using this with
:func:`.Simulation.add_noise`, the generation of a simulation becomes quite
transparent:

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

  sim.prepare_pointings()

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


The input sky can be generated using the function :func:`.Simulation.get_sky` 
which produces sky maps for all the detectors of a given observation based 
on an instance of :class:`.mbs.MbsParameters`. These methods are MPI-compatible, 
distributing inputs based on the job’s detector configuration without requiring 
broadcast operations.


API reference
-------------

.. automodule:: litebird_sim.scan_map
    :members:
    :undoc-members:
    :show-inheritance:
