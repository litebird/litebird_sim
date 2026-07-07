.. _noise:

Instrumental noise
==================

The ability to add noise to your detector timestreams is supported through the
function :func:`.add_noise_to_observations` and the low-level versions
:func:`.add_noise`, :func:`.add_white_noise`, and
:func:`.add_one_over_f_noise`.

White noise
-----------

Here is a short example that shows how to add white noise to timelines:

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


1/f Models and Engines
----------------------

The framework supports also the generatio of 1/f noise. Here you can choose the computational **engine** 
(how it is calculated) and the physical **model** (the shape of the power spectrum).

Engines
^^^^^^^

The engine is selected via the ``engine`` parameter:

1.  **"fft" (Default)**: Generates noise in the Fourier domain.
2.  **"ducc"**: Uses time-domain infinite impulse response (IIR) filtering provided by the `ducc0` library. It only supports the "keshner" model.

Models
^^^^^^

The physical shape of the Power Spectral Density (PSD) is selected via the ``model`` parameter:

1.  **"toast" (Default)**:
    The classic power-law ratio, also implemented in https://github.com/hpc4cmb/toast/blob/372fa7642bbe61a5f01d239e707c04b80ad4bf46/src/toast/tod/sim_noise.py#L74. The PSD is proportional to:

    .. math::

        P(f) \propto \frac{f^\alpha + f_{knee}^\alpha}{f^\alpha + f_{min}^\alpha}

2.  **"keshner"**:
    Corresponds to a sum of relaxation processes. This is the native model of the `ducc` engine. The PSD is proportional to:

    .. math::

        P(f) \propto \left( \frac{f^2 + f_{knee}^2}{f^2 + f_{min}^2} \right)^{\alpha/2}


This call allows to add 1/f noise:

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


Correlated noise
----------------

Real detector arrays are often affected by noise that is partially
correlated across detectors — for example through a common thermal
bath, shared readout electronics, or optical leakage.  The function
:func:`.add_correlated_noise` (and its high-level wrapper
:func:`.add_noise_to_observations`) supports two models for
inter-detector correlations.

Common-mode model
^^^^^^^^^^^^^^^^^

Each detector :math:`i` belonging to group :math:`g` receives:

.. math::

    n_i(t) = \sqrt{\rho_i}\,c_g(t) + \sqrt{1 - \rho_i}\,u_i(t)

where :math:`c_g(t)` is a shared noise stream for the whole group and
:math:`u_i(t)` is a detector-unique stream.  The parameter
:math:`\rho_i \in [0, 1]` controls the fraction of variance contributed
by the common mode: :math:`\rho_i = 0` gives fully independent detectors
while :math:`\rho_i = 1` makes all detectors in a group identical.

Detectors are assigned to groups via the ``group_by`` key (name of a
per-detector attribute, e.g. ``"wafer"``) or an explicit integer label
array passed as ``groups``.

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
       random_seed=12345,
   )

   # Two detectors with identical noise parameters
   dets = [
       lbs.DetectorInfo(name="det_A", net_ukrts=50, sampling_rate_hz=10,
                        alpha=1.0, fknee_mhz=10, fmin_hz=0.001),
       lbs.DetectorInfo(name="det_B", net_ukrts=50, sampling_rate_hz=10,
                        alpha=1.0, fknee_mhz=10, fmin_hz=0.001),
   ]

   obs = sim.create_observations(detectors=dets)

   # Both detectors share the same common-mode stream (rho=0.5 means
   # half of the variance is common, half is independent).
   lbs.noise.add_noise_to_observations(
       obs,
       noise_type='correlated',
       dets_random=sim.dets_random,
       correlation={
           "groups": [0, 0],   # both detectors in group 0
           "rho": 0.5,
       },
   )

   print("det_A sample:", f"{obs[0].tod[0][0]:.3e}")
   print("det_B sample:", f"{obs[0].tod[1][0]:.3e}")

.. testoutput::

   det_A sample: ...
   det_B sample: ...

When detectors should be grouped by a named attribute (e.g. focal-plane
wafer), pass ``group_by="wafer"`` instead of an explicit array.  All
detectors that share the same value of the attribute will receive the
same common-mode stream.

Cholesky model
^^^^^^^^^^^^^^

For a richer correlation structure, you can specify a full
:math:`n_{det} \times n_{det}` correlation matrix :math:`\mathbf{R}`.
The function performs a Cholesky decomposition
:math:`\mathbf{R} = \mathbf{L}\mathbf{L}^T`, generates :math:`n_{det}`
independent unit-variance noise streams :math:`z_j(t)` (one per detector,
each with the correct 1/f PSD), and mixes them:

.. math::

    n_i(t) = \sigma_i \sum_j L_{ij}\,z_j(t)

This allows arbitrary positive-semi-definite correlation structures,
including block-diagonal layouts and continuously varying off-diagonal
elements.

.. testcode::

   import litebird_sim as lbs
   import numpy as np

   sim = lbs.Simulation(
       base_path='./output',
       start_time=0,
       duration_s=100,
       random_seed=12345,
   )

   dets = [
       lbs.DetectorInfo(name=f"det_{i}", net_ukrts=50, sampling_rate_hz=10,
                        alpha=1.0, fknee_mhz=10, fmin_hz=0.001)
       for i in range(3)
   ]

   obs = sim.create_observations(detectors=dets)

   # Define a correlation matrix: strong correlation between det_0/det_1,
   # det_2 is weakly correlated with the others.
   R = np.array([
       [1.0, 0.9, 0.1],
       [0.9, 1.0, 0.1],
       [0.1, 0.1, 1.0],
   ])

   lbs.noise.add_noise_to_observations(
       obs,
       noise_type='correlated',
       dets_random=sim.dets_random,
       correlation={"corr_matrix": R},
   )

   print("det_0 sample:", f"{obs[0].tod[0][0]:.3e}")
   print("det_1 sample:", f"{obs[0].tod[1][0]:.3e}")
   print("det_2 sample:", f"{obs[0].tod[2][0]:.3e}")

.. testoutput::

   det_0 sample: ...
   det_1 sample: ...
   det_2 sample: ...

.. note::

    The matrix :math:`\mathbf{R}` must be symmetric and
    positive-semi-definite (PSD).  If the Cholesky decomposition fails
    because the matrix is numerically singular, add a small diagonal
    regularisation before passing it::

        R += 1e-10 * np.eye(n)

    The diagonal of :math:`\mathbf{R}` should be 1 (unit variance);
    per-detector scaling is handled automatically by the NET values
    stored in the observation.


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
