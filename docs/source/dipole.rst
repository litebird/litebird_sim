.. _dipole-anisotropy:

Dipole anisotropy
=================

The LiteBIRD Simulation Framework provides tools to simulate the
signal associated with the relative velocity between the spacecraft's
rest frame with respect to the CMB. The motion of the spacecraft in
the rest frame of the CMB is the composition of several components:

1. The motion of the spacecraft around L2;
2. The motion of the L2 point in the Ecliptic plane;
3. The motion of the Solar System around the Galactic Centre;
4. The motion of the Milky Way.

Components 1 and 2 are simulated by the LiteBIRD Simulation Framework
using appropriate models for the motions, while components 3 and 4
have been measured by the COBE experiment.

The motion of the spacecraft around L2 is modelled using a Lissajous
orbit similar to what was used for the WMAP experiment
:cite:`2008:wmap:cavaluzzi`, and it is encoded using the
:class:`SpacecraftOrbit` class.

Position and velocity of the spacecraft
---------------------------------------

The class :class:`.SpacecraftOrbit` describes the orbit of the
LiteBIRD spacecraft with respect to the Barycentric Ecliptic Reference
Frame; this class is necessary because the class
:class:`.ScanningStrategy` (see the chapter :ref:`scanning-strategy`)
only models the *direction* each instrument is looking at but knows
nothing about the velocity of the spacecraft itself.

The class :class:`.SpacecraftOrbit` is a dataclass that is able to
initialize its members to sensible default values, which are taken
from the literature. As the LiteBIRD orbit around L2 is not fixed yet,
the code assumes a WMAP-like Lissajous orbit.

To compute the position/velocity of the spacecraft, you call
:func:`.spacecraft_pos_and_vel`; it requires either a time span or a
:class:`.Observation` object, and it returns an instance of the class
:class:`SpacecraftPositionAndVelocity`:

.. testcode::

  import litebird_sim as lbs
  from astropy.time import Time

  orbit = lbs.SpacecraftOrbit(start_time=Time("2023-01-01"))
  posvel = lbs.spacecraft_pos_and_vel(
      orbit,
      start_time=orbit.start_time,
      time_span_s=86_400.0,  # One day
      delta_time_s=3600.0    # One hour
  )

  print(posvel)

.. testoutput::

  SpacecraftPositionAndVelocity(start_time=2023-01-01 00:00:00.000, time_span_s=86400.0, nsamples=25)

The output of the script shows that 25 «samples» have been computed;
this means that the ``posvel`` variable holds information about 25
position/velocity pairs evenly spaced between 2023-01-01 and
2023-01-02: one at midnight, one at 1:00, etc., till midnight
2023-01-02. The :class:`.SpacecraftPositionAndVelocity` class keeps
the table with the positions and the velocities in the fields
``positions_km`` and ``velocities_km_s``, respectively, which are
arrays of shape ``(nsamples, 3)``.

Here is a slightly more complex example that shows how to plot the
distance between the spacecraft and the Sun as a function of time, as
well as its speed. The latter quantity is of course most relevant when
computing the CMB dipole.

.. plot:: pyplots/spacecraft_demo.py
   :include-source:


Computing the dipole
--------------------

The CMB dipole is caused by a Doppler shift of the frequencies
observed while looking at the CMB blackbody spectrum, according to the formula

.. math::
   :label: dipole

   T(\vec\beta, \hat n) = \frac{T_0}{\gamma \bigl(1 - \vec\beta \cdot \hat n\bigr)},

where :math:`T_0` is the temperature in the rest frame of the CMB,
:math:`\vec \beta = \vec v / c` is the dimensionless velocity vector,
:math:`\hat n` is the direction of the line of sight, and
:math:`\gamma = \bigl(1 - \vec\beta \cdot \vec\beta\bigr)^2`.

However, CMB experiments usually employ the linear thermodynamic
temperature definition, where temperature differences :math:`\Delta_1 T`
are related to the actual temperature difference :math:`\Delta T` by
the relation

.. math::
   :label: linearized-dipole

   \Delta_1 T = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}(T_0 + \Delta T)}{\mathrm{BB}(T_0)} - 1\right) =
   \frac{T_0}{f(x)} \left(\frac{\exp x - 1}{\exp\left(x\frac{T_0}{T_0 + \Delta T}\right) - 1} - 1\right),

where :math:`x = h \nu / k_B T`,

.. math:: f(x) = \frac{x e^x}{e^x - 1},

and :math:`\mathrm{BB}(\nu, T)` is the spectral radiance of a
black-body according to Planck's law:

.. math:: \mathrm{BB}(\nu, T) = \frac{2h\nu^3}{c^2} \frac1{e^{h\nu/k_B T} - 1} = \frac{2h\nu^3}{c^2} \frac1{e^x - 1}.
      
There is no numerical issue in computing the full formula, but often
models use some simplifications, to make the math easier to work on
the blackboard. The LiteBIRD Simulation Framework implements several
simplifications of the formula, which are based on a series expansion
of :eq:`dipole`; the caller must pass an object of type
:class:`DipoleType` (an `enum class
<https://docs.python.org/3/library/enum.html>`_), whose value signals
which kind of approximation to use:

1. The most simple formula uses a series expansion of :eq:`dipole` at
   the first order:

   .. math:: \Delta T(\vec\beta, \hat n) = T_0 \vec\beta\cdot\hat n,

   which is associated to the constant ``DipoleType.LINEAR``.

2. The same series expansion for :eq:`dipole`, but stopped at the
   second order (``DipoleType.QUADRATIC_EXACT``):

   .. math:: \Delta T(\vec\beta, \hat n) = T_0\left(\vec\beta\cdot\hat n + \bigl(\vec\beta\cdot\hat n\bigr)^2\right),

   which discards a :math:`-T_0\,\beta^2/2` term (monopole).

3. The exact formula as in :eq:`dipole` (``DipoleType.TOTAL_EXACT``).

4. Using a series expansion to the second order of
   :eq:`linearized-dipole` instead of :eq:`dipole` and neglecting
   monopoles (``DipoleTotal.QUADRATIC_FROM_LIN_T``):

   .. math:: \Delta_2 T(\nu) = T_0 \left(\vec\beta\cdot\hat n + q(x) \bigl(\vec\beta\cdot\hat n\bigr)^2\right),

   where the dependence on the frequency ν is due to the presence of
   the term :math:`x = h\nu / k_B T` in the equation. This is the
   formula to use if you want the leading frequency-dependent term
   (second order) without the boosting induced monopoles.

5. Finally, linearizing :eq:`dipole` through :eq:`linearized-dipole`
   (``DipoleTotal.TOTAL_FROM_LIN_T``):

   .. math::

      \Delta T = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\left(T_0 / \gamma\bigl(1 - \vec\beta\cdot\hat n\bigr)\right)}{\mathrm{BB}(T_0)} - 1\right) =
      \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\bigl(\nu\gamma(1-\vec\beta\cdot\hat n), T_0\bigr)}{\bigl(\gamma(1-\vec\beta\cdot\hat n)\bigr)^3\mathrm{BB}(t_0)}\right).

   In this case too, the temperature variation depends on the
   frequency because of :eq:`linearized-dipole`. This is the formula
   that is typically used by CMB experiments.

You can *add* the dipole signal to an existing TOD through the
function :func:`.add_dipole_to_observations`, as the following example
shows:

.. plot:: pyplots/dipole_demo.py
   :include-source:

The example plots two minutes of a simulated timeline for a very
simple instrument, and it zooms over the very first points to show
that there is indeed some difference in the estimate provided by each
method.

The class :class:`.simulation` provides two simple functions that compute
poisition and velocity of the spacescraft :func:`.simulation.compute_pos_and_vel`, 
and add the solar and orbital dipole to all the observations of a given 
simulation :func:`.simulation.add_dipole`.

.. testcode::

  import litebird_sim as lbs
  from astropy.time import Time
  import numpy as np

  start_time = Time("2025-01-01")
  time_span_s = 1000.0
  sampling_hz = 10.0

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
  )

  sim.create_observations(detectors=det)

  sim.compute_pointings()

  sim.compute_pos_and_vel()

  sim.add_dipole()

  for i in range(5):
      print(f"{sim.observations[0].tod[0][i]:.5e}")

.. testoutput::

   3.44963e-03
   3.45207e-03
   3.45413e-03
   3.45582e-03
   3.45712e-03
           

API reference
-------------

.. automodule:: litebird_sim.spacecraft
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: litebird_sim.dipole
    :members:
    :undoc-members:
    :show-inheritance:
