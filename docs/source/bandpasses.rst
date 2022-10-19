Bandpasses
==========

Some theory
-----------

Any electromagnetic detector is sensitive only to some frequencies of
the spectrum, and it is vital to know what they are. We refer to these
as the «bandpass», i.e., the list of frequencies that can «pass
through the detector» and be detected. A bandpass is a function
:math:`B(\nu)` that associates the frequency :math:`\nu` with a pure
number in the range 0…1, representing the fraction of power that is
actually measured by the instrument. The functional form of
:math:`B(\nu)` is usually called the *bandshape*, and it is usually
nonzero in a limited subset of the real space.

A widely used analytical model of the bandshape is the top-hat bandpass:

.. math::
    :name: eq:tophatbandpass

    B(\nu) = \begin{cases}
    1\,\text{ if } \nu_0 - \Delta\nu/2 < \nu < \nu_0 + \Delta\nu/2,\\
    0\,\text{ otherwise.}
    \end{cases}

Given a bandshape :math:`B(\nu)`, two quantities are usually employed
to synthetically represent it:

- The *central frequency* :math:`\left<\nu\right>`, which is just the
  average value of :math:`\nu` when weighted with :math:`B(\nu)`:

  .. math::
      \left<\nu\right> = \frac{\int_0^\infty \nu B(\nu)\,\mathrm{d}\nu}{\int_0^\infty B(\nu)\,\mathrm{d}\nu}

  which is of course the formula of the weighted average. For the
  top-hat band, the central frequency is :math:`\nu_0`.

- The *bandwidth* :math:`\Delta\nu`, which is the *width* of the band,
  i.e., a measurement of how wide is the subrange of the
  electromagnetic spectrum sampled by the detector:

  .. math::

      \Delta\nu^2 = \frac{\left(\int_0^\infty B(\nu)\,\mathrm{d}\nu\right)^2}{\int_0^\infty B^2(\nu)\,\mathrm{d}\nu},

  which is again inspired by the formula of the standard square
  deviation. For the top-hat bandshape, the value of :math:`\Delta\nu`
  in :math:numref:`eq:tophatbandpass` is exactly the bandwidth.


Modeling bandshapes
-------------------

In the LiteBIRD Simulation Framework, bandpasses are encoded in the
class :class:`.BandPassInfo`, which can be used either to load
bandpasses from the IMO or to generate them on scratch. In the latter
case, you must specify the central frequency and the bandwidth,
alongside with other parameter that introduce a random component meant
to make the band more realistic. Here is a simple example, which shows
how to create a bandpass and then how to add «noise» to make it more
realistic:

.. plot:: pyplots/bandpass_demo.py
   :include-source:


API reference
-------------

.. automodule:: litebird_sim.bandpasses
    :members:
    :undoc-members:
    :show-inheritance:
