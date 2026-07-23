.. _dipole-anisotropy:

Dipole anisotropy
=================

.. contents:: Table of Contents
   :depth: 2
   :local:

The LiteBIRD Simulation Framework provides tools to simulate the
signal associated with the relative velocity between the rest frame of
the spacecraft with respect to the CMB. The motion of the spacecraft
in the rest frame of the CMB is the composition of several components:

1. The motion of the spacecraft around L2;

2. The motion of the L2 point in the Ecliptic plane;

3. The motion of the Solar System around the Galactic Centre;

4. The motion of the Milky Way.

Components 1 and 2 are simulated by the LiteBIRD Simulation Framework
using appropriate motion models, while components 3 and 4 are included
using the Sun velocity derived by the solar dipole measured by the
Planck satellite.

The motion of the spacecraft around L2 is modelled using a Lissajous
orbit similar to what was used for the WMAP experiment
:cite:`2008:wmap:cavaluzzi`, and it is encoded using the
:class:`.SpacecraftOrbit` class.

Position and velocity of the spacecraft
---------------------------------------

The class :class:`.SpacecraftOrbit` describes the orbit of the
LiteBIRD spacecraft with respect to the Barycentric Ecliptic Reference
Frame and the motion of the Barycentric Ecliptic Reference Frame with
respect to the CMB; this class is necessary because the class
:class:`.ScanningStrategy` (see the chapter :ref:`scanning-strategy`)
only models the *direction* each instrument is looking at but knows
nothing about the velocity of the spacecraft itself.

The class :class:`.SpacecraftOrbit` is a dataclass that can initialize
its members to sensible default values taken from the literature. As
the LiteBIRD orbit around L2 is not fixed yet, the code assumes a
WMAP-like Lissajous orbit, :cite:`2008:wmap:cavaluzzi`. For the Sun
velocity it assumes Planck 2018 solar dipole
:cite:`2020:planck:hfi_data_processing`.

To compute the position/velocity of the spacecraft, you call
:func:`.spacecraft_pos_and_vel`; it requires either a period or an
:class:`.Observation` object, and it returns an instance of the class
:class:`.SpacecraftPositionAndVelocity`:

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
distance between the spacecraft and the Sun as a function of time and
speed. The latter quantity is, of course, most relevant when computing
the CMB dipole.

.. plot:: pyplots/spacecraft_demo.py
   :include-source:


Computing the dipole
--------------------

The CMB dipole is caused by a Doppler shift of the frequencies
observed while looking at the CMB blackbody spectrum. In thermodynamic
temperature units the observed temperature along the direction
:math:`\hat n` is

.. math::
   :label: dipole

   T(\vec\beta, \hat n) = \frac{T_0}{\gamma \bigl(1 - \vec\beta \cdot \hat n\bigr)},

where :math:`T_0` is the temperature in the rest frame of the CMB,
:math:`\vec \beta = \vec v / c` is the dimensionless velocity vector,
:math:`\hat n` is the direction of the line of sight, and
:math:`\gamma = \bigl(1 - \vec\beta \cdot \vec\beta\bigr)^{-1/2}`.
The TOD routines add the temperature fluctuation
:math:`\Delta T = T(\vec\beta,\hat n) - T_0`.

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

There is no numerical issue in computing the complete formula, but
series approximations are often useful when separating the dipole,
quadrupole, and octupole contributions. The LiteBIRD Simulation
Framework selects the model through :class:`.DipoleType`:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Type
     - Formula
     - Notes
   * - ``DipoleType.LINEAR``
     - :math:`T_0 b`
     - First order, where :math:`b=\vec\beta\cdot\hat n`.
   * - ``DipoleType.QUADRATIC_EXACT``
     - :math:`T_0\left(b + b^2\right)`
     - Thermodynamic expansion to second order. The monopole
       :math:`-T_0\beta^2/2` is omitted.
   * - ``DipoleType.CUBIC_EXACT``
     - :math:`T_0\left(b + b^2 + b^3\right)`
     - Thermodynamic expansion to third order, again omitting monopole
       terms from the :math:`\gamma` factor.
   * - ``DipoleType.TOTAL_EXACT``
     - :math:`T_0/\left[\gamma(1-b)\right] - T_0`
     - Exact thermodynamic-temperature formula.
   * - ``DipoleType.QUADRATIC_FROM_LIN_T``
     - :math:`T_0\left[b + q(x)b^2\right]`
     - Second-order expansion in linearized thermodynamic units.
   * - ``DipoleType.CUBIC_FROM_LIN_T``
     - :math:`T_0\left[b + q(x)b^2 + r(x)b^3\right]`
     - Third-order expansion in linearized thermodynamic units.
   * - ``DipoleType.TOTAL_FROM_LIN_T``
     - Full expression from :eq:`linearized-dipole`
     - The default model, typically used by CMB experiments.

The frequency-dependent weights in the linearized expansions are

.. math::

   q(x) = \frac{x}{2}\frac{e^x + 1}{e^x - 1},
   \qquad
   r(x) = \frac{x^2(e^{2x} + 4e^x + 1)}{6(e^x - 1)^2}.

Beam convolution is not a separate :class:`.DipoleType`; it is enabled
by passing beam S-parameters to :func:`.add_dipole` (the ``s_params``
keyword) or by setting ``apply_convolution=True`` on
:func:`.add_dipole_to_observations`, as described below.

You can *add* the dipole signal to an existing TOD through the
function :func:`.add_dipole_to_observations`, as the following example
shows:

.. plot:: pyplots/dipole_demo.py
   :include-source:

The example plots two minutes of a simulated timeline for a very
simple instrument, and it zooms over the very first points to show
that there is indeed some difference in the estimate provided by each
method.


Beam-convolved dipole
---------------------

When a real instrument observes the CMB, its response is spread over
the full 4π sky by the beam, including far sidelobes. This changes the
dipole template used for photometric calibration. The implementation
follows the moment expansion in Appendix C of the Planck NPIPE paper
:cite:`2020:planck:npipe`.

For the frequency-dependent linearized expansion the sky template is

.. math::
   :label: dipole-quad

   D(\hat n) = T_0 \bigl[\vec\beta \cdot \hat n
               \bigl(1 + q(x)\, \vec\beta \cdot \hat n\bigr)\bigr],
   \qquad
   q(x) = \frac{x}{2}\,\frac{e^x + 1}{e^x - 1},
   \quad x = \frac{h\nu}{k_B T_0},

which corresponds to ``DipoleType.QUADRATIC_FROM_LIN_T``.

A detector with beam pattern :math:`B(\hat n)` (normalized so that
:math:`\int B(\hat n)\,d\Omega = 1`) observes a beam-convolved signal

.. math::
   :label: dipole-conv

   \tilde D(\hat n_0) = \int d\Omega\, B(\hat n, \hat n_0)\, D(\hat n).

Expanding in Cartesian components and rotating the velocity into the
beam frame (boresight along :math:`\hat z`), the integral reduces to
a dot product with pre-computed beam moments (Eq. C.5 of NPIPE):

.. math::

   \tilde D =
   T_0 \bigl[S_i \beta_i
       + q(x)\, S_{ij} \beta_i \beta_j
       + r(x)\, S_{ijk} \beta_i \beta_j \beta_k\bigr],

where :math:`\boldsymbol\beta` is the velocity in the **beam frame**
and the S-parameters are

.. math::

   S_i     &= \int B(\hat n)\, \hat n_i\, d\Omega, \\
   S_{ij}  &= \int B(\hat n)\, \hat n_i\, \hat n_j\, d\Omega, \\
   S_{ijk} &= \int B(\hat n)\, \hat n_i\, \hat n_j\, \hat n_k\, d\Omega.

These integrals are computed **once** per detector from the full 4π
beam harmonics and then reused for every TOD sample. Convolution by
moment expansion is the only supported way to beam-convolve the dipole.
It supports the polynomial-expansion models ``DipoleType.LINEAR``,
``DipoleType.QUADRATIC_EXACT``, ``DipoleType.CUBIC_EXACT``,
``DipoleType.QUADRATIC_FROM_LIN_T``, and ``DipoleType.CUBIC_FROM_LIN_T``.
The total-formula models ``DipoleType.TOTAL_EXACT`` and
``DipoleType.TOTAL_FROM_LIN_T`` are not supported under convolution.

Computing S-parameters
~~~~~~~~~~~~~~~~~~~~~~

Given beam spherical harmonics in the beam frame (boresight at the
north pole), use :meth:`.BeamSParams.from_beam_alm`:

.. code-block:: python

    import numpy as np
    import litebird_sim as lbs

    beam_alm = lbs.gauss_beam_to_alm(
        lmax=64,
        mmax=64,
        fwhm_rad=np.deg2rad(30.0 / 60.0),
        psi_pol_rad=None,
    )
    s_params = lbs.BeamSParams.from_beam_alm(beam_alm)

The result is a :class:`.BeamSParams` object holding the 3-element
vector ``s_vec``, the 3×3 matrix ``s_mat``, and the 3×3×3 tensor
``s_ten``. If a polarized beam object is provided, only its temperature
component is used for the scalar dipole convolution.

For a circularly symmetric beam, :math:`S_x = S_y = 0` and
:math:`S_{xy} = S_{xz} = S_{yz} = 0` by symmetry, so only
:math:`S_z`, :math:`S_{xx} = S_{yy}`, and :math:`S_{zz}` are
non-zero. As a sanity check, :math:`S_{xx} + S_{yy} + S_{zz} = 1`
(trace equals the beam normalisation).

Adding a convolved dipole
~~~~~~~~~~~~~~~~~~~~~~~~~

For the low-level :func:`.add_dipole`, compute the beam S-parameters
with :meth:`.BeamSParams.from_beam_alm` and pass them via ``s_params``.
The pointing matrices must include the :math:`\psi` column, with shape
``(n_det, n_samples, 3)``. Choose any polynomial-expansion
:class:`.DipoleType`; the example below uses
``DipoleType.QUADRATIC_FROM_LIN_T``:

.. testcode::

    import litebird_sim as lbs
    import numpy as np

    n_samples = 3
    pointings = np.deg2rad(
        np.array([[[90, 0, 0], [90, 90, 0], [90, 180, 0]]], dtype=float)
    )
    velocity = np.tile([300.0, 0.0, 0.0], (n_samples, 1))
    tod = np.zeros((1, n_samples))

    beam_alm = lbs.gauss_beam_to_alm(
        lmax=32,
        mmax=32,
        fwhm_rad=np.deg2rad(30.0),
        psi_pol_rad=None,
    )
    s_params = lbs.BeamSParams.from_beam_alm(beam_alm)

    lbs.add_dipole(
        tod,
        pointings,
        velocity,
        t_cmb_k=lbs.T_CMB_K,
        frequency_ghz=np.array([100.0]),
        dipole_type=lbs.DipoleType.QUADRATIC_FROM_LIN_T,
        s_params=s_params,
    )

For more than one detector, ``s_params`` can be a single
:class:`.BeamSParams` object reused for all detectors, or a dictionary
keyed by detector index strings (``"0"``, ``"1"``, ...). The
observation-oriented :func:`.add_dipole_to_observations` instead takes
beam harmonics directly: set ``apply_convolution=True`` and pass
``beam_alms`` (or store them on the observation's ``blms`` attribute),
and it computes the S-parameters per detector for you. If ``beam_alms``
is a dictionary, its keys must be detector *or* channel names (not
index strings), consistent with :func:`.add_convolved_sky`.

The following plot compares the ordinary
``DipoleType.QUADRATIC_FROM_LIN_T`` dipole with the same model after
convolution with a wide Gaussian beam. The upper panel shows both
signals and the lower panel shows the difference, which is dominated by
the beam suppression of the dipole amplitude.

.. plot:: pyplots/dipole_convolved_demo.py
   :include-source:

Interpretation of the S-parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------+---------------------------------------------------+
| Beam                        | S-parameters                                      |
+=============================+===================================================+
| Perfect pencil (δ at ẑ)     | ``s_vec = [0,0,1]``, ``s_mat = diag(0,0,1)``      |
+-----------------------------+---------------------------------------------------+
| Isotropic (uniform 4π)      | ``s_vec = [0,0,0]``, ``s_mat = I/3``              |
+-----------------------------+---------------------------------------------------+
| Symmetric Gaussian (narrow) | ``s_vec ≈ [0,0,1]``, ``s_mat ≈ diag(ε,ε,1-2ε)``   |
|                             | with small ε > 0                                  |
+-----------------------------+---------------------------------------------------+

For the pencil beam the formula reduces to
:eq:`dipole-quad`, so the convolved result is identical to
``QUADRATIC_FROM_LIN_T``. A real beam with significant
sidelobes will have :math:`S_z < 1` and :math:`S_{xx} = S_{yy} > 0`,
which suppresses the dipole amplitude and introduces a small
pointing-independent offset (from :math:`S_{ij}\beta_i\beta_j`).

Methods of class simulation
---------------------------

The class :class:`.Simulation` provides two simple functions that compute
position and velocity of the spacecraft :func:`.Simulation.compute_pos_and_vel`,
and add the solar and orbital dipole to all the observations of a given
simulation :func:`.Simulation.add_dipole`.

.. testcode::

  import litebird_sim as lbs
  from astropy.time import Time
  import numpy as np

  start_time = Time("2025-01-01")
  time_span_s = 1000.0
  sampling_hz = 10.0

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
  )

  sim.create_observations(detectors=det)

  sim.prepare_pointings()

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

Note that even if :func:`Simulation.compute_pos_and_vel` is not explicitly
invoked, :func:`Simulation.add_dipole` takes care of that internally initializing
:class:`SpacecraftOrbit` and computing positions and velocities.

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
