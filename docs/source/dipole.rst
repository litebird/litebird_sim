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
``positions_km`` and ``velocities_km_s``, respectively.

Here is a slightly more complex example that shows how to plot the
distance between the spacecraft and the Sun as a function of time, as
well as its speed. The latter quantity is of course most relevant when
computing the CMB dipole.

.. plot:: pyplots/spacecraft_demo.py
   :include-source:


Computing the dipole
--------------------

The CMB dipole is caused by a Doppler shift of the frequencies
observed while looking at the CMB blackbody spectrum; the shift
depends on the relative velocity of the observer with respect to the
CMB rest frame, and this explains the reason why the class
:class:`.SpacecraftPositionAndVelocity` was implemented.

The CMB dipole signal is a deceptively simple phenomenon that has many
small quirks. Here are a few of them:

1. While the motion of the Solar System with respect to the CMB rest
   frame can be considered constant throughout the mission, this is
   not the case for the orbital motion of the spacecraft around the
   Sun. This means that the dipole has a constant component (which can
   be plotted as a map over the 4π sphere) and a variable component;
   the latter cannot be represented on a map but must be studied in
   the detector timelines.
2. When considering relativistic corrections to the classical Doppler
   formula, any motion with respect to the CMB rest frame changes the
   apparent arrival direction of photons, thus inducing higher-order
   perturbations (the so-called Doppler quadrupole, octupole, etc.).
   See :cite:`2015:quadrupole:notari` for more information.
3. The relativistic corrections to the quadrupole are
   frequency-dependent and need to be computed over the band response
   of each detector.
4. Finally, the dipole is always observed through a *realistic* beam,
   which means that the actual signal is the convolution of the dipole
   (plus its corrective terms at higher orders) with a beam function.



           
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
