.. _planets:

Planets
=======

.. contents:: Table of Contents
   :depth: 2
   :local:

Mars, Jupiter, Saturn, Uranus, and Neptune are the brightest compact
sources in the sky at CMB frequencies (Mercury and Venus never enter the
field of view from L2, as they are always too close to the Sun). The
framework can add their signal to the TODs and flag the samples close to
them (see :ref:`observations`).

Signal of the planets
---------------------

:meth:`.Simulation.add_planets` adds the signal of the planets to the
TODs::

    sim.prepare_pointings()
    sim.add_planets()  # All the planets in lbs.PLANETS
    sim.add_planets(planets=["jupiter", "saturn"], component="planets")

The planets are point sources for LiteBIRD's beams: the signal of a
detector whose beam is centred at an angle :math:`\theta` from a planet is

.. math::

   \Delta T(t) = T_\text{p}\,\frac{\Omega_\text{p}(t)}{\Omega_\text{b}}\,
   \hat B(\theta(t)),

where :math:`\hat B` is the beam normalized to 1 at its peak,
:math:`\Omega_\text{b}` its solid angle, and
:math:`\Omega_\text{p} = \pi r_\text{eq} r_\text{pol} / R^2` the solid angle
of the disc of the planet at a distance :math:`R` from the spacecraft
(:data:`.PLANET_RADII_KM`). :math:`T_\text{p}` (see
:func:`.planet_brightness_k_cmb`) is the temperature in K\ :sub:`CMB` of a
source that fills the beam with the emission of the planet: the band
integral of the spectral radiance :math:`B(\nu, T_\text{planet}(\nu))`
divided by that of :math:`\partial B / \partial T` at the temperature of the
CMB, for a top-hat band (``bandcenter_ghz``, ``bandwidth_ghz``).

The direction and the distance of the planets are computed from the
position of the spacecraft (:meth:`.Simulation.compute_pos_and_vel`,
called if needed), so the times of the simulation must be
``astropy.time.Time`` objects. As for the flagging, the pointings of the
detectors are only computed in the segments of time in which the planets
are close to the boresight.

To give an idea of the amplitudes, Jupiter at 4.5 AU gives ~100 mK\ :sub:`CMB`
at the centre of a 30′ beam at 140 GHz.

Emission model
--------------

:data:`.PLANET_TEMPERATURES_K` contains whole-disc thermodynamic
temperatures of the planets with respect to blank sky, i.e., with the CMB
monopole occulted by the planet already subtracted, which is what enters
a differential TOD. :func:`.planet_temperature_k` interpolates them
linearly in :math:`\log\nu` and keeps them constant outside the range of
the measurements. The values come from:

- 100–857 GHz: Planck HFI, mean over the surveys (Table 4 of `Planck
  intermediate results LII <https://arxiv.org/abs/1612.07151>`_, A&A 607,
  A122, 2017);
- below 100 GHz, Jupiter and Saturn: Planck LFI (Table 3 of the same
  paper);
- below 100 GHz, Mars and Uranus: WMAP (`Weiland et al. 2011
  <https://arxiv.org/abs/1001.4731>`_, ApJS 192, 19), converted from
  Rayleigh-Jeans to thermodynamic temperatures. For Uranus, only the bands
  with errors below 5% are used, and there are no such measurements for
  Neptune, which therefore keeps the 100 GHz value at lower frequencies.

The model has known limitations, at the level of a few percent:

- Mars changes by ±5% with its distance from the Sun and its seasons, and
  the table contains a mean value;
- the HFI values of Saturn include its rings, whose contribution depends on
  their inclination, while the LFI values do not;
- the beams are circular Gaussians with the FWHM of the detectors,
  truncated at 3 FWHM, and the bands are top-hats. The radial profile of
  the beams would give realistic sidelobes;
- the planets are unpolarized (Planck limits the polarization of Mars to
  less than 2%).

API reference
-------------

.. automodule:: litebird_sim.planets
    :members:
    :undoc-members:
