Map-making
==========

The primary aim of the LiteBIRD Simulation Framework is to create
synthetic timestreams as if they were acquired by the real LiteBIRD
Instrument. The process that creates maps out of these timelines is
called *map-making* and it is strictly considered a data-analysis
task, not a simulation task. However, since most of the assessments on
the quality of a timestream can only be done on maps, the LiteBIRD
Simulation Framework provides some facilities to produce maps out of
timestreams. These maps are created using the `Healpix
<https://en.wikipedia.org/wiki/HEALPix>`_ pixelization scheme and
saved in FITS files.

There are two available map-makers:

1. A *binner*, i.e., a simple map-maker that assumes that only
   uncorrelated noise is present in the timelines.

2. A *destriper*, i.e., a more advanced map-maker that can remove the
   effect of correlated instrumental noise from the timelines before
   producing a map. The uncorrelated noise is usually referred as 1/f
   noise, and the purpose of the destriper is to estimate its
   contribution and remove it from the timelines; then, a classical
   *binner* is ran over the cleaned timelines.

The LiteBIRD Simulation Framework provides a binner on its own, and it
internally uses `TOAST <https://github.com/hpc4cmb/toast>`_ to provide
the destriper.

In this chapter, we assume that you have already created the timelines
that must be provided as input to the destriper. Here is a sample code
that creates a simple timeline containing white noise for two
detectors::

    import numpy as np
    import astropy.units as u
    from numpy.random import MT19937, RandomState, SeedSequence
    
    import litebird_sim as lbs
    
    sim = lbs.Simulation(
        base_path="destriper_output",
        start_time=0,
        duration_s=86400.0,
    )

    sim.generate_spin2ecl_quaternions(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(30),  # CORE-specific parameter
            spin_rate_hz=0.5 / 60,  # Ditto
            # We use astropy to convert the period (4 days) in
            # seconds
            precession_rate_hz=1.0 / (4 * u.day).to("s").value,
        )
    )
    instr = lbs.InstrumentInfo(
        name="core",
        spin_boresight_angle_rad=np.deg2rad(65),
    )
    
    # We create two detectors, whose polarization angles are separated by π/2
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(name="0A", sampling_rate_hz=10),
            lbs.DetectorInfo(
                name="0B", sampling_rate_hz=10, quat=lbs.quat_rotation_z(np.pi / 2)
            ),
        ],
        dtype_tod=np.float64,  # Needed if you use the TOAST destriper
        n_blocks_time=lbs.MPI_COMM_WORLD.size,
        split_list_over_processes=False,
    )

    # Generate some white noise
    rs = RandomState(MT19937(SeedSequence(123456789)))
    for curobs in sim.observations:
        curobs.tod *= 0.0
        curobs.tod += rs.randn(*curobs.tod.shape)
  

Binner
------

Once you have generated a set of observations, either on a single
process or distributed over several mpi processes, you can create a 
simple binned map with the function :func:`.make_bin_map`. This function
takes: a single (or a list) of :class:`.Observations`, the Healpix
resolution of the output map (``nside``) and produces a coadded map.
It assumes white noise and each detector gets weighted by 
:math:`1 / NET^2`. Optionally, if the parameter do_covariance is True,
it can output the covariance matrix in an array of shape 
`(12 * nside * nside, 3, 3)`. This is how should be called::

    map, cov = lbs.make_bin_map(obs, 128, do_covariance=True)


Destriper
---------

To use the TOAST destriper, you must create a
:class:`.DestriperParameters` object that specifies which input
parameters (apart from the timelines) should be used::
  
    params = lbs.DestriperParameters(
        nside=16,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
    )
  
The parameters we use here are the resolution of the output map
(``nside=16``), and the kind of results that must be returned:
specifically, we are looking here for the *hit map* (i.e., a map that
specifies how many samples were observed while the detector was
looking at a specific pixel), the *binned map* (the same map that
would be produced by the *binner*, see above), and the *destriped map*
(the most important result of the computation, of course).

To run the destriper, you simply call :func:`.destripe`::

  result = lbs.destripe(sim, instr, params)

(You must provide an instance of the :class:`.InstrumentInfo` class,
as this is used by the destriper to determine pointing directions in
Ecliptic coordinates.) The result is an instance of the class
:class:`.DestriperResults` and contains the three maps we have asked
above (hit map, binned map, destriped map).

.. note::

   The TOAST destriper only works with timelines containing 64-bit
   floating point numbers. As the default data type for timelines
   created by ``sim.create_observations`` is a 32-bit float, if you
   plan to run the destriper you should pass the flag
   ``dtype_tod=np.float64`` to ``sim.create_observations`` (see the
   code above), otherwise ``destripe`` will create an internal copy of
   the TOD converted in 64-bit floating-point numbers, which is
   usually a waste of space.

Let's plot the binned map (the most reasonable output, as we have not
included correlated noise in our example)::

  import healpy

  # Plot the I map
  healpy.mollview(result.binned_map[0])

Here is the complete source code of the example and the result:

.. plot:: pyplots/destriper_demo.py
   :include-source:

   
API reference
-------------

.. automodule:: litebird_sim.mapping
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: litebird_sim.destriper
    :members:
    :undoc-members:
    :show-inheritance:
