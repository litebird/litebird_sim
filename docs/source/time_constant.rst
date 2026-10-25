.. _time_constant:

Detector time constant
======================

A detector does not respond instantaneously to a change of the optical
power: TES bolometers behave as one-pole low-pass filters with a time
constant :math:`\tau`. In the frequency domain, a timeline :math:`d(f)`
becomes

.. math::
    d'(f) = H(f)\,d(f), \qquad H(f) = \frac{1}{1 + 2\pi i f \tau}.

At frequencies :math:`f \ll 1/(2\pi\tau)` the effect is mostly a delay of
the signal by :math:`\tau`. For a signal modulated by a half-wave plate at
:math:`4 f_\text{HWP}`, the phase shift :math:`\arctan(2\pi \cdot 4
f_\text{HWP}\,\tau)` corresponds to a rotation of the polarization angle by
half of it, while the amplitude is reduced by :math:`1/\sqrt{1 + (2\pi \cdot
4 f_\text{HWP}\,\tau)^2}`. Data are usually deconvolved by multiplying them
by :math:`1/H(f)`; any error in the estimate of :math:`\tau` leaves a
residual.

The framework provides functions to convolve and deconvolve the TOD with the
time constant. The time constant of each detector is stored in the field
``time_constant_ms`` of :class:`.DetectorInfo` (in ms; the default is 0, which
leaves the TOD unchanged), but it can also be passed explicitly.

The order of the calls is up to the user, as the filter is linear and acts
only on what is in the TOD when it is called. For instance, the noise of the
photons is filtered by the detector, while the readout noise is not: so the
first is added before the convolution and the second after it. If the
components are kept in separate TODs (see the ``component`` argument), each
of them can be filtered or not.

The filter is applied with FFTs to the samples of each observation, after
having extended the timeline with its mirror image. This is exact for
signals that have no power above the Nyquist frequency. Since a time
constant shorter than the sampling period is a sub-sample delay, whose
response decays slowly, the first and last ~100 samples of each observation
are affected by an error that is larger than in the rest of the timeline.

Examples
--------

.. code-block:: python

   (... importing modules, creating the simulation, setting the scanning
   strategy, the instrument, etc...)

   dets = [
       lbs.DetectorInfo(name="det_A", sampling_rate_hz=19.0, time_constant_ms=3.0),
       lbs.DetectorInfo(name="det_B", sampling_rate_hz=19.0, time_constant_ms=4.0),
   ]

   (obs,) = sim.create_observations(detectors=dets)

   (... fill the TOD by scanning a map, adding the dipole, etc...)

   # Convolve the signal with the time constants in DetectorInfo
   sim.convolve_time_constant()

   # Add the noise (here it is not filtered by the detector)
   sim.add_noise(noise_type="white", user_seed=1234)

   # Deconvolve with a time constant overestimated by 10%
   sim.deconvolve_time_constant(time_constant_ms=[3.3, 4.4])

The same can be done on a list of observations with
:func:`.convolve_time_constant_in_observations` and
:func:`.deconvolve_time_constant_in_observations`, or on a 2D array with
:func:`.convolve_time_constant` and :func:`.deconvolve_time_constant`.

API reference
-------------

.. automodule:: litebird_sim.time_constant
    :members:
    :undoc-members:
    :show-inheritance:
