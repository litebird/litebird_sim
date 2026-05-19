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

There is no numerical issue in computing the complete formula, but
often, models use some simplifications to make the math more
manageable to work on the blackboard. The LiteBIRD Simulation
Framework implements several simplifications of the formula, which are
based on a series expansion of :eq:`dipole`; the caller must pass an
object of type :class:`.DipoleType` (an `enum class
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
      \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\bigl(\nu\gamma(1-\vec\beta\cdot\hat n), T_0\bigr)}{\bigl(\gamma(1-\vec\beta\cdot\hat n)\bigr)^3\mathrm{BB}(T_0)}\right).

   In this case too, the temperature variation depends on the
   frequency because of :eq:`linearized-dipole`. This is the formula
   that is typically used by CMB experiments.

6. A beam-convolved version of formula 4 (``DipoleType.CONVOLVED``),
   described in the next section.

7. An exact beam-convolved version of formula 5
   (``DipoleType.CONVOLVED_TOTAL_FROM_LIN_T``), also described below.

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
the full 4π sky by the beam, including far sidelobes.  This distorts
the dipole template used for photometric calibration and must be
accounted for.  The method follows Appendix C of the Planck NPIPE
paper :cite:`2020:planck:npipe` (arXiv:2007.04997).

The frequency-dependent dipole+quadrupole template is

.. math::
   :label: dipole-quad

   D(\hat n) = T_0 \bigl[\vec\beta \cdot \hat n
               \bigl(1 + q(x)\, \vec\beta \cdot \hat n\bigr)\bigr],
   \qquad
   q(x) = \frac{x}{2}\,\frac{e^x + 1}{e^x - 1},
   \quad x = \frac{h\nu}{k_B T_0},

which corresponds to ``DipoleType.QUADRATIC_FROM_LIN_T``.

A detector with beam pattern :math:`B(\hat n)` (normalised so that
:math:`\int B(\hat n)\,d\Omega = 1`) observes a beam-convolved signal

.. math::
   :label: dipole-conv

   \tilde D(\hat n_0) = \int d\Omega\, B(\hat n, \hat n_0)\, D(\hat n).

Expanding in Cartesian components and rotating the velocity into the
beam frame (boresight along :math:`\hat z`), the integral reduces to
a dot product with pre-computed beam moments (Eq. C.5 of NPIPE):

.. math::

   \tilde D = T_0 \bigl[S_i \beta_i + q(x)\, S_{ij} \beta_i \beta_j\bigr],

where :math:`\boldsymbol\beta` is the velocity in the **beam frame**
and the S-parameters are

.. math::

   S_i    &= \int B(\hat n)\, \hat n_i\, d\Omega, \\
   S_{ij} &= \int B(\hat n)\, \hat n_i\, \hat n_j\, d\Omega.

These integrals need to be computed **once** per detector from the
full 4π beam map and then reused for every TOD sample.

Computing S-parameters
~~~~~~~~~~~~~~~~~~~~~~

Given a HEALPix beam map in the beam frame (boresight at the north
pole, RING ordering), use :func:`.compute_s_params_from_beam_map`:

.. code-block:: python

    import healpy as hp
    import numpy as np
    import litebird_sim as lbs

    # Load or simulate a beam map (RING-ordered HEALPix, boresight at north pole).
    # The map must be normalised so that sum(beam) * (4π / npix) = 1.
    nside = 512
    npix  = hp.nside2npix(nside)

    # Example: Gaussian beam with FWHM = 30 arcmin
    fwhm_rad = np.deg2rad(30.0 / 60.0)
    sigma = fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    pixel_area = 4.0 * np.pi / npix
    vecs = np.array(hp.pix2vec(nside, np.arange(npix)))
    cos_theta = vecs[2]  # z-component = cos(angular distance from north pole)
    beam_map = np.exp(-0.5 * (np.arccos(np.clip(cos_theta, -1, 1)) / sigma) ** 2)
    beam_map /= beam_map.sum() * pixel_area  # normalise to unit integral

    s_params = lbs.compute_s_params_from_beam_map(beam_map)

The result is a :class:`.BeamSParams` object holding the 3-element
vector ``s_vec`` and the 3×3 matrix ``s_mat``.

For a circularly symmetric beam, :math:`S_x = S_y = 0` and
:math:`S_{xy} = S_{xz} = S_{yz} = 0` by symmetry, so only
:math:`S_z`, :math:`S_{xx} = S_{yy}`, and :math:`S_{zz}` are
non-zero.  As a sanity check, :math:`S_{xx} + S_{yy} + S_{zz} = 1`
(trace equals the beam normalisation).

Using DipoleType.CONVOLVED
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the :class:`.BeamSParams` instance to :func:`.add_dipole` or
:func:`.add_dipole_to_observations` together with
``dipole_type=lbs.DipoleType.CONVOLVED``.  The pointing matrices
must include the ψ column (shape ``(n_det, n_samples, 3)``):

.. testcode::

    import litebird_sim as lbs
    import numpy as np

    # Pencil-beam S-parameters: delta function at the boresight.
    # With these parameters DipoleType.CONVOLVED is identical to
    # DipoleType.QUADRATIC_FROM_LIN_T (useful as a unit-test baseline).
    s_params = lbs.BeamSParams(
        s_vec=np.array([0.0, 0.0, 1.0]),
        s_mat=np.diag([0.0, 0.0, 1.0]),
    )

    pointings = np.deg2rad(
        np.array([[[0, 0, 0], [90, 0, 0], [180, 0, 0]]])
    )
    velocity = 299_792.458 * np.array(
        [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]
    )

    tod = np.zeros((1, 3))
    lbs.add_dipole(
        tod,
        pointings,
        velocity,
        t_cmb_k=1.0,
        frequency_ghz=[100.0],
        dipole_type=lbs.DipoleType.CONVOLVED,
        s_params=s_params,
    )

    for val in tod[0]:
        print(f"{val:.6f}")

.. testoutput::

    0.000000
    0.124395
    0.000000

The following plot compares the pencil-beam dipole with the one produced
by a Gaussian beam of FWHM = 60°, scanning the sky along the equator
with the velocity pointing in the +x direction.  The upper panel shows
both signals and the lower panel shows the difference, which arises from
the beam suppression of the dipole amplitude (by a factor :math:`S_z < 1`)
and the small pointing-independent offset from the quadrupole term.

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
| Symmetric Gaussian (narrow) | ``s_vec ≈ [0,0,1]``, ``s_mat ≈ diag(ε,ε,1-2ε)``  |
|                             | with small ε > 0                                  |
+-----------------------------+---------------------------------------------------+

For the pencil beam the formula reduces to
:eq:`dipole-quad`, so ``CONVOLVED`` with these S-parameters is
identical to ``QUADRATIC_FROM_LIN_T``.  A real beam with significant
sidelobes will have :math:`S_z < 1` and :math:`S_{xx} = S_{yy} > 0`,
which suppresses the dipole amplitude and introduces a small
pointing-independent offset (from :math:`S_{ij}\beta_i\beta_j`).


Exact convolution: DipoleType.CONVOLVED_TOTAL_FROM_LIN_T
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DipoleType.CONVOLVED`` truncates the Doppler shift at second order
in :math:`\beta` (i.e. it uses ``QUADRATIC_FROM_LIN_T`` as the
per-pixel integrand).  ``DipoleType.CONVOLVED_TOTAL_FROM_LIN_T``
instead evaluates the full :attr:`DipoleType.TOTAL_FROM_LIN_T` formula
for every beam pixel and sums the weighted contributions:

.. math::

   \tilde{D}(\hat{n}_0) = \sum_p w_p\,
   \frac{T_0}{f(x)} \left(
   \frac{\mathrm{BB}\!\left(T_0 / \gamma(1-\boldsymbol{\beta}\cdot\hat{n}_p)\right)}
   {\mathrm{BB}(T_0)} - 1
   \right),
   \qquad w_p = B_p \cdot \frac{4\pi}{N_\mathrm{pix}}

where :math:`\hat{n}_p` are the beam pixel unit vectors in the beam
frame and :math:`\boldsymbol{\beta}` is the velocity rotated into the
beam frame (exactly as for ``CONVOLVED``).

Because the full Planck function is evaluated at each pixel the result
is exact to all orders in :math:`\beta`; the runtime is
:math:`O(N_\mathrm{pix})` per TOD sample rather than :math:`O(1)`.

**Difference from CONVOLVED.**  The Taylor expansion of the exact
formula gives

.. math::

   \frac{T_0}{f(x)}\!\left(\frac{\mathrm{BB}(T_0/\gamma/(1-\mu))}{\mathrm{BB}(T_0)}-1\right)
   \approx T_0\!\left(\mu - \frac{\beta^2}{2} + q(x)\,\mu^2\right) + O(\beta^3),

so ``CONVOLVED_TOTAL_FROM_LIN_T`` minus ``CONVOLVED`` equals the
pointing-independent monopole correction :math:`-T_0\beta^2/2`, which
is the relativistic :math:`\gamma`-factor term absent from
``QUADRATIC_FROM_LIN_T``.  For the CMB dipole
(:math:`\beta\approx 10^{-3}`) this offset is :math:`\approx -1.4\,\mu\mathrm{K}`.

**Usage.**  Instead of a :class:`.BeamSParams` object, pass a
:class:`.BeamConvolutionData` object returned by
:func:`.compute_beam_convolution_data_from_beam_map`:

.. code-block:: python

   import litebird_sim as lbs
   import healpy as hp, numpy as np

   # Build the full-4π beam map (RING, normalised) at your preferred nside.
   nside = 64
   npix  = hp.nside2npix(nside)
   theta, _ = hp.pix2ang(nside, np.arange(npix))
   sigma_rad = np.deg2rad(10.0)
   beam_map = np.exp(-0.5 * theta**2 / sigma_rad**2)
   beam_map /= beam_map.sum() * (4 * np.pi / npix)   # normalise

   beam_conv = lbs.compute_beam_convolution_data_from_beam_map(beam_map)

   lbs.add_dipole(
       tod, pointings, velocity,
       t_cmb_k=lbs.T_CMB_K,
       frequency_ghz=freq_arr,
       dipole_type=lbs.DipoleType.CONVOLVED_TOTAL_FROM_LIN_T,
       beam_conv_data=beam_conv,
   )

The pointing matrices must include the :math:`\psi` column (shape
``(n_det, n_samples, 3)``), exactly as for ``DipoleType.CONVOLVED``.

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
