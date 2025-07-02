.. _beamconvolution:

Convolve Alms with a Beam to fill a TOD
=======================================

The framework provides the function :func:`.add_convolved_sky`, which performs 
harmonic-space convolution of sky and beam alms, filling the detector timestreams. 
It supports both cases, with and without the HWP. The relevant mathematical details 
are found in :cite:`2010:prezeau:conviqt` and :cite:`2019:duivenvoorden:beamconv`.

To populate an existing TOD with a signal, use the function :func:`.add_convolved_sky_to_observations` 
as demonstrated in the following example:

.. testcode::

    import litebird_sim as lbs
    import numpy as np
    import healpy as hp

    start_time_s = 0
    time_span_s = 1

    lmax = 64
    mmax = lmax - 4

    # Create a simulation
    sim = lbs.Simulation(
       base_path="./output",
       start_time=start_time_s,
       duration_s=time_span_s,
       random_seed=12345,
    )

    # Define the scanning strategy
    sim.set_scanning_strategy(
       lbs.SpinningScanningStrategy(
           spin_sun_angle_rad=0.785_398_163_397_448_3,
           precession_rate_hz=8.664_850_513_998_931e-05,
           spin_rate_hz=0.000_833_333_333_333_333_4,
           start_time=start_time_s,
       ),
       delta_time_s=1,
    )

    sim.set_instrument(
       lbs.InstrumentInfo(
           boresight_rotangle_rad=0.0,
           spin_boresight_angle_rad=0.872_664_625_997_164_8,
           spin_rotangle_rad=3.141_592_653_589_793,
       ),
    )

    # Create a detector object
    det = lbs.DetectorInfo(
        name="Detector",
        sampling_rate_hz=10.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    # Initialize the observation
    (obs,) = sim.create_observations(detectors=[det])

    # Prepare the quaternions used to compute the pointings
    sim.prepare_pointings()

    ncoeff = lbs.SphericalHarmonics.num_of_alm_from_lmax(lmax,mmax)

    np.random.seed(1234)
    blms = lbs.SphericalHarmonics(
        values=(np.random.normal(0,1,3*ncoeff)+1j*np.random.normal(0,1,3*ncoeff)).reshape(3,ncoeff),
        lmax=lmax,
        mmax=mmax,
    )

    alms = lbs.SphericalHarmonics(
        values=hp.synalm(np.ones((4,lmax+1)),lmax=lmax, mmax=lmax,),
        lmax=lmax,
        mmax=lmax,
    )

    Convparams = lbs.BeamConvolutionParameters(
        lmax = lmax,
        mmax = mmax,
        single_precision = True,
        epsilon = 1e-5,
    )

    # Here scan the map and fill tod
    lbs.add_convolved_sky_to_observations(
        obs,
        alms,
        blms,
        input_sky_alms_in_galactic=True,
        convolution_params=Convparams,
        pointings_dtype=np.float32,
    )

    for i in range(obs.n_samples):
       value = np.round(obs.tod[0][i], 3)
       print(f"{value:.3f}")

.. testoutput::

    113.796
    116.186
    118.567
    120.935
    123.300
    125.655
    128.000
    130.335
    132.661
    134.975

Input Data Format
-----------------

The input sky alms and beam alms must be encapsulated in instances of the 
:class:`SphericalHarmonics` class. These can be stored in a dictionary, 
using either the channel or detector name as a keyword, or passed directly 
to :func:`.add_convolved_sky_to_observations`. The routines in :ref:`Mbs` 
already provide correctly formatted inputs. See below for a desctription 
of the dataclass :class:`SphericalHarmonics`.

Pointing Information
--------------------

Pointing data can either be embedded in the observation or passed explicitly 
via the `pointings` parameter. If both `observations` and `pointings` are provided,
they must be consistent—either as a single observation and a single numpy array 
or as lists of equal length.

If input alms are in ecliptic coordinates, set `input_sky_alms_in_galactic=False`. 
The HWP effect can be incorporated via pointing data (see :ref:`scanning-strategy`) 
or by using the `hwp` argument. The polarization angles of 
detectors are derived from the observation attributes.

The option `nside_centering=NSIDE` shifts the detector pointings to the centers of the 
corresponding HEALPix pixels at the given `NSIDE` resolution. This option is useful for 
debugging and for reducing sub-pixel effects.  

Convolution Parameters
----------------------

The convolution parameters must be specified via the :class:`BeamConvolutionParameters` 
dataclass. Allowed parameters:

- `lmax` (int): Maximum ℓ value for sky and beam coefficients.
- `mmax` (int): Maximum m (azimuthal moment) for beam coefficients, constrained to `mmax ≤ lmax - 4`.
- `single_precision` (bool): Set to `False` for 64-bit floating-point calculations. Default: `True`.
- `epsilon` (float): Desired relative accuracy of interpolation. Default: `1e-5`.
- `strict_typing` (bool): If `True` (default), a `TypeError` is raised if pointing types do not 
   match `single_precision`. If `False`, the code silently converts types at the expense of memory.

If convolution parameters are omitted, defaults are inferred from sky and beam alms, triggering a warning.


Container for Spherical Harmonics
---------------------------------

The :class:`SphericalHarmonics` class stores spherical harmonic coefficients. 
In libraries like HealPy, alms are stored in NumPy arrays, but their ℓ_max and
m_max values cannot be uniquely determined from array size (except when ℓ_max = m_max). 
This class ensures proper handling, enforcing `mmax ≤ lmax` and consistently 
using shape `(nstokes, ncoeff)`.

The class :class:`SphericalHarmonics` serves a container for the spherical harmonics 
coefficients. The convention used in libraries like HealPy is to keep the a_ℓm coefficients
of a spherical harmonic expansion in a plain NumPy array. However, this is
ambiguous because it is not possible to uniquely determine the value of
ℓ_max and m_max from the size of the array (unless you assume that ℓ_max == m_max)
This class allows to store any set of alms with the only restriction that `m_max≤l_max`
The shape of alms stored is *always* ``(nstokes, ncoeff)``, even if ``nstokes == 1``
It also provides :func:`.resize_alm`, allowing alms to be resized via zero-padding or truncation. 
Example usage:

.. testcode::

    import litebird_sim as lbs
    import numpy as np

    lmax = 10
    mmax = 3
    ncoeff = lbs.SphericalHarmonics.num_of_alm_from_lmax(lmax,mmax)

    np.random.seed(12345)
    coeff = np.random.normal(0,1,ncoeff)+1j*np.random.normal(0,1,ncoeff)

    alms = lbs.SphericalHarmonics(values=coeff,lmax=lmax, mmax=mmax)

    alms_resized = alms.resize_alm(lmax_out=3,mmax_out=2)

    alms_resized_real_part = np.real(alms_resized.values[0])

    for r in alms_resized_real_part:
        value = np.round(r,5)
        print(f"{value:.5f}")

.. testoutput::

    -0.20471
    0.47894
    -0.51944
    -0.55573
    -1.29622
    0.27499
    0.22891
    0.47699
    3.24894


Elliptical Gaussian Beam Spherical Harmonics
--------------------------------------------

The framework provides a function :func:`.gauss_beam_to_alm`, which analytically computes the 
spherical harmonic coefficients (a_ℓm) representing a 2D elliptical Gaussian beam, with optional 
polarization and cross-polar leakage.
The parameters are:

- `lmax`: Maximum spherical harmonic degree.
- `mmax`: Maximum harmonic order.
- `fwhm_rad`: Full width at half maximum of the beam, defined as fwhm = sqrt(fwhm_max*fwhm_min) (in radians).
- `ellipticity`: Ellipticity of the beam defined as fwhm_max/fwhm_min (1.0 for circular).
- `psi_ell_rad`: Orientation of the beam’s major axis with respect to the x-axis (radians).
- `psi_pol_rad`: Polarization reference angle with respect to the x-axis (radians). If None, only intensity is computed.
- `cross_polar_leakage`: Cross-polarization leakage factor.

The function returns a :class:`SphericalHarmonics` object with the intensity and (if requested) polarized 
components of the beam in harmonic space.

The function :func:`.generate_gauss_beam_alms` provides a convenient way to compute Gaussian beam harmonics for all 
detectors in an :class:`.Observation`. It wraps around :func:`.gauss_beam_to_alm` and automatically pulls relevant 
beam parameters from the Observation object. 
The parameters are:

- `observation`: An :class:`.Observation` object containing per-detector syntetic beam properties.
- `lmax`: Maximum spherical harmonic degree.
- `mmax`: Maximum harmonic order. Defaults to lmax.
- `store_in_observation`: If True, the result is stored in the observation.blms attribute. Default False

It returns a dictionary mapping each detector name to its corresponding :class:`SphericalHarmonics` object.
This is simple example of usage::

  blms = lbs.generate_gauss_beam_alms(
      observation=my_obs,
      lmax=512,
      store_in_observation=True
  )


Methods of the Simulation class
-------------------------------

The :class:`.Simulation` class offers various functions to streamline convolution:

- :func:`.Simulation.get_gauss_beam_alms`: Generates Gaussian beam alms for all detectors using :class:`.DetectorInfo`.
- :func:`.Simulation.get_sky`: Produces sky alms based on an instance of :class:`.mbs.MbsParameters`.
- :func:`.Simulation.convolve_sky`: Convolves sky and beam alms for all observations in the simulation.

These methods are MPI-compatible, distributing inputs based on the job’s detector configuration without requiring broadcast operations.

For a single-task execution, refer to the following example:

.. testcode::

    import litebird_sim as lbs
    import numpy as np

    start_time_s = 0
    time_span_s = 10

    nside = 256

    lmax = 2 * nside
    mmax = lmax - 4

    # Create a simulation
    sim = lbs.Simulation(
       base_path="./output",
       start_time=start_time_s,
       duration_s=time_span_s,
       random_seed=12345,
    )

    # Define the scanning strategy
    sim.set_scanning_strategy(
       lbs.SpinningScanningStrategy(
           spin_sun_angle_rad=0.785_398_163_397_448_3,
           precession_rate_hz=8.664_850_513_998_931e-05,
           spin_rate_hz=0.000_833_333_333_333_333_4,
           start_time=start_time_s,
       ),
       delta_time_s=1,
    )

    sim.set_instrument(
       lbs.InstrumentInfo(
           boresight_rotangle_rad=0.0,
           spin_boresight_angle_rad=0.872_664_625_997_164_8,
           spin_rotangle_rad=3.141_592_653_589_793,
       ),
    )

    # Create a detector object
    det = lbs.DetectorInfo(
        name="Detector",
        sampling_rate_hz=10.0,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        fwhm_arcmin=30.0,
        ellipticity=1.0,
        psi_rad=0.0,
        pol_angle_rad=0.0,
    )

    # Initialize the observation
    sim.create_observations(detectors=[det])

    # Prepare the quaternions used to compute the pointings
    sim.prepare_pointings()

    # Gaussian beam alms 
    blms = sim.get_gauss_beam_alms(lmax=lmax)

    # Create the alms to convolve
    mbs_params = lbs.MbsParameters(
        make_cmb=True,
        make_fg=False,
        nside=nside,
        units="K_CMB",
        gaussian_smooth=False,
        bandpass_int=False,
        store_alms=True,
        lmax_alms=lmax,
        seed_cmb=12345,
    )

    alms = sim.get_sky(parameters = mbs_params)

    Convparams = lbs.BeamConvolutionParameters(
        lmax = lmax,
        mmax = mmax,
        single_precision = True,
        epsilon = 1e-5,
    )

    sim.convolve_sky(sky_alms=alms,
                     beam_alms=blms,
                     convolution_params=Convparams,
                     input_sky_alms_in_galactic=True,
                     pointings_dtype=np.float32,
                     nthreads = 0)


API reference
-------------

.. automodule:: litebird_sim.beam_convolution
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: litebird_sim.spherical_harmonics
    :members:

.. automodule:: litebird_sim.beam_synthesis
    :members: