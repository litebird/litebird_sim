Observations
============

In this section we describe the :class:`.Observation` class, which
represents a sequence of time-ordered-data (TOD) acquired by a
detector. An object of type :class:`Observation` is usually used to
encode the TOD acquired during a time span when the characteristics of
the instrument can be assumed to be stable (i.e., constant gain, no
variations in the properties of noise and bandpass, etc.).

A :class:`Observation` object encodes two types of information:

- Actual data samples (i.e., the samples that have been acquired by a
  detector);
- Time stamps for each data sample.

Time stamps can be represented either as floating-point values
(appropriate in 99% of the cases) or MJD; in the latter case, these
are handled through the `AstroPy <https://www.astropy.org/>`_ library.

Here are a few examples of how to create a :class:`Observation`
object::

  import litebird_sim as lbs
  from astropy.time import Time
  
  # First case: use floating-point values to keep track of time
  obs_no_mjd = lbs.Observation(
      detector="A",
      start_time=0.0,
      sampling_rate_hz=5.0,
      nsamples=5,
      use_mjd=False,
  )

  # Second case: use MJD to track the time
  obs_mjd = lbs.Observation(
      detector="B",
      start_time=Time("2020-02-20", format="iso"),
      sampling_rate_hz=5.0,
      nsamples=5,
      use_mjd=True,
  )


API reference
-------------

.. automodule:: litebird_sim.observations
    :members:
    :undoc-members:
    :show-inheritance:
