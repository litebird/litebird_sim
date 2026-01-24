.. _input_sky:

Synthetic Sky Maps
==================

The ``litebird_sim.input_sky`` module provides the tools necessary to produce synthetic sky. It is designed to generate realistic sky components (CMB, foregrounds, dipole) in either pixel space or harmonic space, serving as the input for both full timeline generation and fast map-based simulations.

The outputs are provided as :class:`~litebird_sim.maps_and_harmonics.HealpixMap` or :class:`~litebird_sim.maps_and_harmonics.SphericalHarmonics` objects. For more details on these data structures, please refer to the :ref:`maps_and_harmonics` documentation.


The framework utilizes the `PySM3 <https://pysm3.readthedocs.io/en/latest/>`_ library to generate foreground components and internal tools for CMB and dipole generation.

Usage Example
-------------

The core interface is the :class:`~litebird_sim.input_sky.SkyGenerator` class. It requires a configuration object (:class:`~litebird_sim.input_sky.SkyGenerationParams`) and a list of channel or detector definitions.

Here is an example showing how to generate a sky containing CMB and specific foregrounds (Dust and Synchrotron) using the dedicated ``Units`` Enum:

.. code-block:: python

    import litebird_sim as lbs
    from litebird_sim.input_sky import SkyGenerator, SkyGenerationParams
    from litebird_sim.units import Units

    # 1. Define Simulation Parameters
    params = SkyGenerationParams(
        nside=512,
        units=Units.K_CMB,       # Use strict units from the Enum
        output_type="map",       # Options: "map" or "alm"
        make_cmb=True,
        seed_cmb=12345,          # Ensure reproducibility with a seed
        make_fg=True,
        fg_models=["d0", "s1"],  # PySM3 model short codes
        apply_beam=True,         # Apply the beam defined in the channel info
        bandpass_integration=True
    )

    # 2. Define Channels
    # (In a full simulation, these are usually loaded from the IMO)
    channels = [
        lbs.FreqChannelInfo(
            name="L1-040",
            freq_ghz=40.0,
            fwhm_arcmin=60.0,
            # Bandpass data would be loaded here
        )
    ]

    # 3. Initialize and Run
    sky_gen = SkyGenerator(parameters=params, channels=channels)
    sky_maps = sky_gen.execute()

    # The result is a dictionary keyed by channel name
    # healpix_map = sky_maps["L1-040"]

Configuration Parameters
------------------------

The :class:`~litebird_sim.input_sky.SkyGenerationParams` class controls every aspect of the simulation. Below is a summary of its key parameters:

**General Settings**
    * ``nside`` (int): The HEALPix resolution parameter ($N_{side}$) for the output maps.
    * ``lmax`` (int): Maximum multipole moment ($\ell_{max}$) for spherical harmonic transforms.
    * ``output_type`` (str): The format of the output. Either ``"map"`` (pixel space) or ``"alm"`` (harmonic space).
    * ``units`` (Units | str): The desired output units (e.g., ``Units.K_CMB``, ``Units.MJy_sr``). If a string is provided, it is converted to the corresponding Enum.
    * ``return_components`` (bool): If True, returns a dictionary separated by component (CMB, FG, Dipole) instead of a summed sky.
    * ``apply_beam`` (bool): If True, convolves the sky with the beam width defined in the channel information.
    * ``apply_pixel_window`` (bool): If True, applies the HEALPix pixel window function during harmonic transforms.
    * ``bandpass_integration`` (bool): If True, integrates emission over the detector's frequency band. If False, computes emission at the band center.

**CMB Settings**
    * ``make_cmb`` (bool): Enable/Disable CMB generation.
    * ``seed_cmb`` (int): Random seed for the Gaussian realization.
    * ``cmb_r`` (float): Tensor-to-scalar ratio for the simulation.
    * ``cmb_ps_file`` (Path, optional): Path to a FITS file containing input power spectra ($C_\ell$). If None, defaults to Planck 2018 best-fit.

**Foreground Settings**
    * ``make_fg`` (bool): Enable/Disable foreground generation.
    * ``fg_models`` (list[str]): List of PySM3 model codes (e.g., ``["d1", "s1"]``).
    * ``fg_oversampling`` (int): Factor by which to oversample the resolution internally before downgrading.

**Dipole Settings**
    * ``make_dipole`` (bool): Enable/Disable solar dipole generation.
    * ``sun_velocity_kms`` (tuple, optional): Velocity of the observer in km/s. If None, uses the Solar System Barycenter velocity from ``litebird_sim.constants``.
    * ``sun_direction_galactic`` (tuple, optional): Direction of the velocity vector in Galactic coordinates (longitude, latitude) in degrees.

Component Generation Details
----------------------------

The module generates three distinct signal components which are combined linearly.

**1. CMB Generation**
The Cosmic Microwave Background is generated as a Gaussian random realization in harmonic space.
* **Spectra:** By default, it utilizes the Planck 2018 best-fit power spectra (TT, EE, BB, TE). A custom Tensor-to-Scalar ratio ($r$) can be injected via the ``cmb_r`` parameter.
* **Synthesis:** The $a_{\ell m}$ coefficients are synthesized from these spectra using the provided random seed. If ``output_type="map"``, these are transformed to pixel space using ``ducc0`` or ``healpy``.

**2. Foreground Generation**
Foregrounds are simulated using the **PySM3** library.
* **Bandpass Integration:** When ``bandpass_integration=True``, PySM3 integrates the emission laws over the specific frequency response of each channel. This captures color corrections accurately.
* **Resolution handling:** To prevent artifacts when smoothing is applied, the module can generate the foregrounds at a higher internal resolution (controlled by ``fg_oversampling``) before smoothing and downgrading to the requested ``nside``.

**3. Solar Dipole**
The module calculates the kinetic Doppler dipole caused by the motion of the observer relative to the CMB rest frame.
* It generates a map or $a_{\ell m}$ corresponding to a dipole amplitude derived from the velocity vector provided in ``sun_velocity_kms`` and the direction provided in ``sun_direction_galactic``.
* **Customization and Defaults:** You can modify the dipole direction by setting the ``sun_direction_galactic`` parameter. If this parameter is left as ``None``, the module utilizes the default values defined in ``litebird_sim.constants``. These defaults are based on the Planck 2018 results (`Planck Collaboration 2018 <https://arxiv.org/abs/1807.06207>`_).

Available Emission Models
-------------------------

The ``fg_models`` parameter in :class:`~litebird_sim.input_sky.SkyGenerationParams` accepts a list of string identifiers defining the foreground components. These strings correspond directly to the model short codes defined in the PySM3 library.

For detailed physical descriptions of each model, please refer to the official PySM3 documentation:

`PySM3 Models Documentation <https://pysm3.readthedocs.io/en/latest/models.html>`_

Below is a summary of the available model codes:

**Thermal Dust**
    * ``d0``: Fixed spectral index (beta=1.54) and temperature (20 K).
    * ``d1``: "Standard" model. Spatially varying temperature and spectral index (Planck Commander).
    * ``d2``, ``d3``: Spatially varying emissivity (beta=1.59).
    * ``d4``: Two-component dust model (Finkbeiner et al. 1999).
    * ``d5``: Hensley and Draine 2017 physical dust model.
    * ``d6``: Frequency decorrelation model.
    * ``d7``, ``d8``: Modifications of d5 (iron inclusions, fixed ISRF).
    * ``d9``, ``d10``, ``d11``: GNILC-based models (Planck 2018), including logpoltens small-scale fluctuations.
    * ``d12``: 3D model of polarized dust emission (MKD) with 6 layers.

**Synchrotron**
    * ``s1``: "Standard" power law with spatially varying spectral index (Miville-Deschenes et al. 2008).
    * ``s2``: Spectral index steepens off the Galactic plane.
    * ``s3``: Curved power law.
    * ``s4``: Fixed spectral index (-3.1).
    * ``s5``: Power law based on S-PASS with logpoltens small-scale fluctuations.
    * ``s6``: Like s5 but with stochastic small scales generated on-the-fly.
    * ``s7``: Curved power law based on s5 and ARCADE data.

**Anomalous Microwave Emission (AME)**
    * ``a1``: Sum of two spinning dust populations (Commander).
    * ``a2``: AME with 2% polarization fraction.

**Free-Free**
    * ``f1``: Analytic model from Commander fit to Planck 2015.

**CO Line Emission**
    * ``co1``: Galactic CO emission (J=1-0, 2-1, 3-2) from Planck/MILCA.
    * ``co2``: Polarized CO emission (0.1%).
    * ``co3``: Includes mock CO clouds off the Galactic plane.

**Extra-Galactic Sources**
    * ``cib1``: Cosmic Infrared Background (WebSky 0.4).
    * ``tsz1``: Thermal Sunyaev–Zeldovich effect (WebSky 0.4).
    * ``ksz1``: Kinetic Sunyaev–Zeldovich effect (WebSky 0.4).
    * ``rg1``, ``rg2``, ``rg3``: Radio Galaxies (WebSky 0.4) including brightest and background sources.

Coordinate Systems and Units
----------------------------

The ``SkyGenerator`` strictly operates in **Galactic coordinates**.

It provides robust handling of physical units via the ``litebird_sim.units.Units`` Enum. Two modes of calculation are supported:

1.  **Monochromatic**: If ``bandpass_integration`` is False, conversions use the standard Astropy equivalencies at the band center frequency.
2.  **Bandpass Integrated**: If ``bandpass_integration`` is True, the emission is integrated over the specific detector bandpass using PySM3, ensuring accurate color corrections for broad bands.

Structure of the Output
-----------------------

The :meth:`~litebird_sim.input_sky.SkyGenerator.execute` method returns a dictionary containing the generated sky objects. Depending on the ``output_type`` parameter, these will be instances of either ``HealpixMap`` or ``SphericalHarmonics``.

If ``return_components=True`` is set in the parameters, the output will be a nested dictionary separating the components:

.. code-block:: python

    {
        "cmb": { "channel_name": map_obj, ... },
        "foregrounds": { "channel_name": map_obj, ... },
        "dipole": { "channel_name": map_obj, ... }
    }

Otherwise, it returns the sum of all requested components.

API Reference
-------------

.. automodule:: litebird_sim.input_sky
    :members:
    :undoc-members:
    :show-inheritance:
