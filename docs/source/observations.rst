Observations
============

In this section we describe the :class:`.Observation` class, which
we use as the main container for the data acquired by the telescope during
a scaning period (and the relevant information about it).

For a dedicated discussion about how this data is organized in memory,
see :doc:`data_layout`

The :class:`.Observation` class
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
  obs = lbs.Observation(
      detectors=2,
      start_time=0.0,
      sampling_rate_hz=5.0,
      n_samples=5,
  )

  # Second case: use MJD to track the time
  obs_mjd = lbs.Observation(
      detectors=[{"name": "A"}, {"name": "B"}]
      start_time=Time("2020-02-20", format="iso"),
      sampling_rate_hz=5.0,
      nsamples=5,
  )

Note that the 2-D array ``obs.tod`` is created for you. Its shape is
``(n_detectors, n_samples)``, and the default type is
``numpy.float32``: the choice of a 32-bit type is usually good enough
for the purposes of LiteBIRD simulations, but if you need less/more
precision, you are free to use any of the floating-point types
provided by NumPy (``float16``, ``float32``, ``float64``,
``float128``) using the keyword ``dtype_tod``::

  import numpy as np
  
  obs = lbs.Observation(
      detectors=2,
      start_time=0.0,
      sampling_rate_hz=5.0,
      n_samples=5,
      dtype_tod=np.float64,  # Use a 64-bit floating point type
  )

  
In full scale simulations the TOD may get too large to fit in memory.
You can chunk it along the time or detector dimension (or both) using
``n_blocks_det, n_blocks_time, comm`` at construction time or the
`set_n_blocks` method. The same chunking is applied also to any
detector information that you add with `detector_global_info` or
`detector_info`. Note note ``det_idx`` is added for you at
construction.

When you distribute the observation the first
``n_blocks_det x n_blocks_time``  MPI ranks are organized in a row-major grid
with ``n_blocks_det`` rows and ``n_blocks_time`` columns. Each owns a block of
the TOD and the information of the corresponding to the rows in its block.


API reference
-------------

.. automodule:: litebird_sim.observations
    :members:
    :undoc-members:
    :show-inheritance:
