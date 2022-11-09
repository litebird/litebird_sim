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

3. You can also use :func:`.save_simulation_for_madam` to save TODs
   and pointing information to disk and then manually call the `Madam
   mapmaker <https://arxiv.org/abs/astro-ph/0412517>`_.

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
:math:`1 / NET^2`. If the pointing information is not provided in the 
observation, it can be passed through the optional argument `pointings`, 
with a syntax similar to :func:`.scan_map_in_observations`.
The output map is in Galactic coordinates, unless the optional prameter
`output_map_in_galactic` is set to False. If the parameter `do_covariance` 
is True, it return also the white noise covariance per pixel in an array 
of shape `(12 * nside * nside, 3, 3)`. This is how it should be called::

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

  result = lbs.destripe(sim, params)

(The pointing information is included in the :class:`.Observations`,
alternatively pointings can be provided as a list of numpy arrays)
The result is an instance of the class :class:`.DestriperResults` and 
contains the three maps we have asked above (hit map, binned map, 
destriped map).

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


Saving files for Madam
----------------------

The function :func:`.save_simulation_for_madam` takes a :class:`.Simulation`
object, a list of detectors and a output path (the default is a subfolder of
the output path of the simulation) and saves a set of files in it:

1. Pointing information, saved as FITS files;
2. TOD samples, saved as FITS files;
3. A so-called «simulation file», named ``madam.sim``;
4. A so-called «parameter file», named ``madam.par``.

These files are ready to be used with the Madam map-maker; you just need
to pass the parameter file ``madam.par`` to one of the executables provided
by Madam (the other ones are referenced by the parameter file). For instance,
the following command will compute the amount of memory needed to run Madam:

.. code-block:: text

    $ inputcheck madam.par

The following command will run Madam:

.. code-block:: text

    $ madam madam.par

Of course, in a realistic situation you want to run ``madam`` using MPI,
so you should call ``mpiexec``, ``mpirun``, or something similar.


Creating several maps with Madam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are cases where you want to create several maps out of
one simulation. A common case is when you simulate several
components to be put in the TOD and store them in different
fields within each :class:`.Observation` object:

.. code-block:: python

  sim.create_observations(params)

  for cur_obs in sim.observations:
      # We'll include several components in the signal:
      # CMB, white noise, 1/f noise, and dipole
      cur_obs.wn_tod = np.zeros_like(cur_obs.tod)
      cur_obs.oof_tod = np.zeros_like(cur_obs.tod)
      cur_obs.cmb_tod = np.zeros_like(cur_obs.tod)
      cur_obs.dip_tod = np.zeros_like(cur_obs.tod)

      # Now fill each of the *_tod fields appropriately

In cases like this, it is often useful to generate several
sets of maps that include different subsets of the components:

1.  A map including just the CMB and white noise;
2.  A map including CMB, white noise and 1/f, but not the dipole;
3.  A map including all the components.

You could of course call :func:`.save_simulation_for_madam` three
times, but this would be a waste of space because you would end
up with three identical copies of the FITS file containing the
pointings, and the last set of FITS files would contain the same
components that were saved for the first two maps.
(Remember, :func:`.save_simulation_for_madam` saves both the
pointings and the TODs!)

A trick to avoid wasting so much space is to save the FITS files
only once, including *all* the TOD components, and then call
:func:`.save_simulation_for_madam` again for each other map using
``save_pointings=False`` and ``save_tods=False``:

.. code-block:: python

  save_files = True

  # Iterate over all the maps we want to produce. For each of
  # them we specify the name of the subfolder where the Madam
  # files will be saved and the list of components to include
  for (subfolder, components_to_bin) in [
      ("cmb+wn", ["wn_tod", "cmb_tod"]),
      ("cmb+wn+1f", ["wn_tod", "oof_tod", "cmb_tod"]),
      ("cmb+wn+1f+dip", ["wn_tod", "oof_tod", "cmb_tod", "dip_tod"]),
  ]:
      save_simulation_for_madam(
          sim=sim,
          params=params,
          madam_subfolder_name=subfolder,
          components=["cmb_tod", "wn_tod", "oof_tod", "dipole_tod"],
          components_to_bin=components_to_bin,
          save_pointings=save_files,
          save_tods=save_files,
       )

       # Set this to False for all the following iterations (maps)
       save_files = False


It is important that the `components` parameter in the call to
:func:`.save_simulation_for_madam` list *all* the components, even if they are not going to be used in the first and
second map. The reason is that this parameter is used by the function to create a «map» of the components as they are
supposed to be found in the FITS files; for example, the
``cmb_tod`` field is the *third* in each TOD file, but this
would not be apparent while producing the first map, where it is the
*second* in the list of components that must be used. The ``.par``
file will list the components that need to be actually used to
create the map, so it will not be confused if the TOD FITS files
will contain more components than needed. (This is a neat feature
of Madam.)

.. note::

   To understand how this kind of stuff works, it is useful to
   recap how Madam works, as the possibility to reuse TODs for
   different maps is linked to the fact that Madam requires
   *two* files to be run: the *parameter file* and the *simulation
   file*.

   The *simulation file* represents a «map» of the content of a
   set of FITS files. No information about the map-making process
   is included in a simulation file: it just tells how many FITS
   files should be read and what is inside each of them.

   The *parameter file* is used to tell Madam how you want maps
   to be created. It's in the parameter file that you can ask
   Madam to skip parts of the TODs, for example because you do
   not want to include the dipole in the output map.

   When you call :func:`.save_simulation_for_madam`, the
   `components` parameter is used to build the *simulation file*:
   thus, if you plan to build more than one map out of the same
   set of components, you want to have the very same simulation
   files, because they «describe» what's in the FITS files. This
   is the reason why we passed the same value to ``components``
   every time we called ``save_simulation_for_madam``.

   But when we create the three *parameter files*, each of them
   differs in the list of components that need to be included.
   If you inspect the three files ``cmb+wn/madam.par``,
   ``cmb+wn+1f/madam.par``, and ``cmb+wn+1f+dip/madam.par``, you
   will see that they only differ for the following lines::

      # cmb+wn/madam.par
      tod_1 = wn_tod
      tod_2 = cmb_tod

      # cmb+wn+1f/madam.par
      tod_1 = wn_tod
      tod_2 = oof_tod
      tod_3 = cmb_tod

      # cmb+wn+1f+dip/madam.par
      tod_1 = wn_tod
      tod_2 = oof_tod
      tod_3 = cmb_tod
      tod_4 = dip_tod


   That's it. The lines with ``tod_*`` are enough to
   do all the magic to build the three maps.


Of course, once the three directories ``cmb+wn``, ``cmb+wn+1f``, and
``cmb+wn+1f+dip`` are created, Madam will run successfully only in
the first one, ``cmb+wn``. The reason is that only that directory
includes the pointing and TOD FITS files! But if you are saving data
on a filesystem that supports `symbolic links
<https://en.wikipedia.org/wiki/Symbolic_link>`_, you can use them to
make the files appear in the other directories too. For instance, the
following commands will create them directly from a Unix shell (Bash,
Sh, Zsh):

.. code-block:: sh

  ln -srv cmb+wn/*.fits cmb+wn+1f
  ln -srv cmb+wn/*.fits cmb+wn+1f+dip

(The ``-s`` flag asks to create *soft* links, the ``-r`` flag requires
paths to be relative, and ``-v`` makes ``ln`` be more verbose.)

If you want a fully authomatic procedure, you can create the symbolic
links in Python, taking advantage of the fact that
:func:`.save_simulation_for_madam` returns a dictionary containing the
information needed to do this programmatically:

.. code-block:: python

   # First map
   params1 = save_simulation_for_madam(sim, params)

   # Second map
   params2 = save_simulation_for_madam(
       sim,
       params,
       madam_subfolder_name="madam2",
       save_pointings=False,
       save_tods=False,
   )

   # Caution: you want to do this only within the first MPI process!
   if litebird_sim.MPI_COMM_WORLD.rank == 0:
       for source_file, dest_file in zip(
           params1["tod_files"] + params1["pointing_files"],
           params2["tod_files"] + params2["pointing_files"],
       ):
           # This is a Path object associated with the symlink that we
           # want to create
           source_file_path = source_file["file_name"]
           dest_file_path = dest_file["file_name"]

           dest_file_path.symlink_to(source_file_path)

You can include this snippet of code in the script that calls
:func:`.save_simulation_for_madam`, so that the procedure will
be 100% automated.


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

.. automodule:: litebird_sim.madam
    :members:
    :undoc-members:
    :show-inheritance:
