.. _maps_and_harmonics:

Maps and Spherical Harmonics
============================

The ``litebird_sim.maps_and_harmonics`` module provides robust data containers for HEALPix maps and Spherical Harmonic coefficients ($a_{\ell m}$).

While libraries like ``healpy`` provide low-level array manipulation, this module introduces an object-oriented layer designed to prevent common errors in cosmological simulation pipelines, such as mixing coordinate systems or misinterpreting array shapes.

Design Philosophy
-----------------

In standard pipelines, a map is often just a NumPy array. This leads to ambiguity:
is an array of shape ``(3, N)`` a T/Q/U map, or three T-only maps? Is the map in Galactic or Ecliptic coordinates?

The classes :class:`HealpixMap` and :class:`SphericalHarmonics` solve this by:

1.  **Strict Shape Enforcement**: Data is always stored as ``(nstokes, size)`` for single-frequency, or ``(nfreqs, nstokes, size)`` for multi-frequency. A temperature-only map is ``(1, npix)``, never ``(npix,)``.
2.  **Metadata Propagation**: Every object carries physical ``units`` and ``coordinates``. Algebraic operations (like ``map_a + map_b``) automatically check that these match, raising a ``ValueError`` if you try to add a Galactic map to an Ecliptic one.
3.  **Multi-Frequency Support**: Both classes support storing data for multiple frequencies simultaneously, with automatic validation that frequency arrays match during operations.
4.  **Backend Agnosticism**: While the API feels like Healpy, the heavy lifting (SHTs) is performed by `ducc0 <https://gitlab.mpcdf.mpg.de/mtr/ducc>`_, offering significant performance improvements and correct handling of spin-weighted transforms.

Data Structures
---------------

Healpix Maps
~~~~~~~~~~~~

The :class:`~litebird_sim.maps_and_harmonics.HealpixMap` class wraps a dense HEALPix map.

.. note::
    This class does not perform reordering (RING vs NESTED) internally. 
    It stores the ordering flag ``nest`` as metadata. Users must ensure they are using the correct geometry for their specific analysis tools.

Key features:

* **Geometry checks**: The number of pixels is validated against the ``nside`` upon initialization.
* **Algebra**: Supports ``+``, ``-``, ``*`` (scalar), and Stokes vector multiplication.
* **Multi-frequency support**: Can store maps for multiple frequencies with shape ``(nfreqs, nstokes, npix)``. The ``frequencies_ghz`` attribute holds the frequency array, and ``nfreqs`` indicates the number of frequencies.
* **Static Helpers**: Access HEALPix geometry math without importing ``healpy`` (e.g., :meth:`~litebird_sim.maps_and_harmonics.HealpixMap.nside_to_resolution_rad`).

Spherical Harmonics
~~~~~~~~~~~~~~~~~~~

The :class:`~litebird_sim.maps_and_harmonics.SphericalHarmonics` class wraps $a_{\ell m}$ coefficients.
It solves the "triangular array ambiguity" by explicitly storing ``lmax`` and ``mmax`` alongside the coefficients.

* **Storage**: Standard HEALPix/ducc triangular layout.
* **Convolution**: The :meth:`~litebird_sim.maps_and_harmonics.SphericalHarmonics.convolve` method allows easy application of beams or transfer functions in harmonic space. Supports flexible multi-frequency convolution with 1D (broadcast to all frequencies), 2D (TEB filters), or 3D (per-frequency-stokes) filter arrays.
* **Resizing**: The :meth:`~litebird_sim.maps_and_harmonics.SphericalHarmonics.resize_alm` method allows you to truncate or zero-pad coefficients to a new $\ell_{max}$.
* **Multi-frequency support**: Can store coefficients for multiple frequencies with shape ``(nfreqs, nstokes, nalms)``. The ``frequencies_ghz`` attribute holds the frequency array, and ``nfreqs`` indicates the number of frequencies.

Spherical Harmonics Ordering
----------------------------

A frequent source of confusion in CMB analysis is the ordering of $a_{\ell m}$ coefficients in the 1D array. 

The :class:`SphericalHarmonics` class strictly enforces the **Standard Healpix Ordering** (also known as **m-major**).

**Memory Layout**

The coefficients are stored sequentially by iterating over $m$ (outer loop) and then $\ell$ (inner loop). This allows efficient recursion for Legendre polynomial calculations.

1.  Block $m=0$: contains $\ell = 0, 1, \dots, \ell_{max}$
2.  Block $m=1$: contains $\ell = 1, 2, \dots, \ell_{max}$
3.  ...
4.  Block $m=m_{max}$: contains $\ell = m_{max}, \dots, \ell_{max}$

**Indexing Formula**

To find the index of a specific mode $(\ell, m)$, the class provides a static method :meth:`SphericalHarmonics.get_index`. It implements the standard formula:

.. math::

    \text{index} = m \cdot \frac{2 \ell_{max} + 1 - m}{2} + \ell

**Example: Converting coordinates to indices**

.. code-block:: python

    from litebird_sim import SphericalHarmonics
    import numpy as np

    lmax = 10
    
    # Get index for a single mode (l=2, m=2)
    # Note: Arguments are (lmax, l, m) to match healpy signature
    idx = SphericalHarmonics.get_index(lmax, 2, 2)
    
    # Vectorized usage
    ls = np.array([2, 3, 4])
    ms = np.array([2, 2, 2])
    indices = SphericalHarmonics.get_index(lmax, ls, ms)

.. _transforms:

Transforms (SHT)
----------------

We provide high-level wrappers around `ducc0` for spherical harmonic transforms. 
These functions handle the complexity of spin-0 (Temperature) vs spin-2 (Polarization) transforms automatically.

* :func:`~litebird_sim.maps_and_harmonics.estimate_alm`: Map $\rightarrow$ $a_{\ell m}$ (Analysis)
* :func:`~litebird_sim.maps_and_harmonics.pixelize_alm`: $a_{\ell m}$ $\rightarrow$ Map (Synthesis)
* :func:`~litebird_sim.maps_and_harmonics.interpolate_alm`: $a_{\ell m}$ $\rightarrow$ Values at arbitrary $(\theta, \phi)$
* :func:`~litebird_sim.maps_and_harmonics.compute_cl`: Compute auto- or cross-power spectra from $a_{\ell m}$
* :func:`~litebird_sim.maps_and_harmonics.compute_dl`: Compute $D_{\ell} = \ell(\ell+1)C_{\ell}/(2\pi)$ spectra

.. tip::
   All transform functions accept a ``nthreads`` argument. 
   Setting ``nthreads=0`` (default) uses all available hardware threads, which is optimal for standalone scripts but should be adjusted when running inside an MPI environment.

.. note::
   All transform functions support multi-frequency data. When operating on multi-frequency objects, transforms are applied independently to each frequency, and the output maintains the multi-frequency structure.

Cookbook
--------

Basic I/O and Algebra
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from litebird_sim.maps_and_harmonics import HealpixMap, SphericalHarmonics
    from litebird_sim.constants import Units
    from litebird_sim.coordinates import CoordinateSystem

    # Loading a map from disk (wrapper around hp.read_alm usually not needed for maps, 
    # but classes support direct instantiation)
    # Here we create a dummy map for demonstration
    import numpy as np
    
    nside = 128
    npix = HealpixMap.nside_to_npix(nside)
    
    # Create a Temperature-only map in Galactic coordinates
    m_gal = HealpixMap(
        values=np.zeros((1, npix)), 
        nside=nside, 
        units=Units.K_CMB, 
        coordinates=CoordinateSystem.Galactic
    )
    
    # Create a mask (unitless)
    mask = HealpixMap(
        values=np.ones((1, npix)), 
        nside=nside, 
        units=Units.None, 
        coordinates=CoordinateSystem.Galactic
    )
    
    # Multiplication works (Units.K_CMB * Units.None = Units.K_CMB)
    masked_map = m_gal * mask

Smoothing a Polarization Map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to smooth a T, Q, U map using a Gaussian beam.

.. code-block:: python

    from litebird_sim.maps_and_harmonics import estimate_alm, pixelize_alm
    from litebird_sim.beam_synthesis import gauss_bl

    # 1. Start with a 3-component map (T, Q, U)
    # shape must be (3, npix)
    m_pol = HealpixMap(..., nside=64, units=Units.uK_CMB)

    # 2. Analysis: Convert to alms
    # This automatically handles spin-0 for T and spin-2 for Q,U
    alms = estimate_alm(m_pol, lmax=128)

    # 3. Create a beam window function B_ell
    # gauss_bl returns an array of size lmax+1
    fwhm_rad = np.radians(1.0)
    b_ell = gauss_bl(fwhm_rad, lmax=128)

    # 4. Convolve (apply beam)
    # The 'convolve' method applies the filter to all Stokes components 
    # if a single array is passed.
    alms_smoothed = alms.convolve(b_ell)

    # 5. Synthesis: Convert back to map
    m_smoothed = pixelize_alm(alms_smoothed, nside=64)

Interpolating Alms
~~~~~~~~~~~~~~~~~~

If you need to evaluate a set of alms at specific locations (e.g., catalog positions):

.. code-block:: python

    from litebird_sim.maps_and_harmonics import interpolate_alm
    
    # Define locations: (theta, phi) in radians
    # Shape must be (N, 2)
    locations = np.array([
        [1.57, 0.0],  # Equator, phi=0
        [0.0, 0.0]    # North Pole
    ])

    # Interpolate
    # If alms.nstokes == 3, this returns tuple (T, Q, U)
    # where each is an array of length N (single-freq) or (nfreqs, N) (multi-freq)
    t_vals, q_vals, u_vals = interpolate_alm(alms, locations)

Multi-Frequency Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both :class:`HealpixMap` and :class:`SphericalHarmonics` support multi-frequency data:

.. code-block:: python

    from litebird_sim.maps_and_harmonics import SphericalHarmonics, compute_cl
    import numpy as np
    
    # Create multi-frequency alms
    lmax = 128
    nfreqs = 3
    frequencies = np.array([100.0, 143.0, 217.0])  # GHz
    
    # Shape: (nfreqs, nstokes, nalms)
    nalms = SphericalHarmonics.alm_array_size(lmax, lmax, 3)[1]
    values = np.random.randn(nfreqs, 3, nalms) + 1j * np.random.randn(nfreqs, 3, nalms)
    
    alms = SphericalHarmonics(
        values=values,
        lmax=lmax,
        frequencies_ghz=frequencies
    )
    
    # Multi-frequency smoothing: apply different FWHM to each frequency
    fwhm_arcmin = np.array([30.0, 23.0, 14.0])
    fwhm_rad = np.radians(fwhm_arcmin / 60.0)
    alms_smooth = alms.apply_gaussian_smoothing(fwhm_rad=fwhm_rad)
    
    # Compute power spectra: returns 2D arrays (nfreqs, lmax+1)
    cls = compute_cl(alms_smooth, alms_smooth)
    # cls["TT"] has shape (3, 129) for 3 frequencies
    # cls["frequency_ghz"] contains the frequency array
    
    # Arithmetic operations check frequency compatibility
    alms2 = alms.copy()
    alms_sum = alms + alms2  # Works: same frequencies
    
    # This would raise ValueError:
    # alms_bad = SphericalHarmonics(..., frequencies_ghz=np.array([95, 150, 220]))
    # alms + alms_bad  # ValueError: frequencies not compatible!

API Reference
-------------

.. autoclass:: litebird_sim.maps_and_harmonics.SphericalHarmonics
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: coordinates, frequencies_ghz, lmax, mmax, nfreqs, nstokes, units, values

.. autoclass:: litebird_sim.maps_and_harmonics.HealpixMap
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: coordinates, frequencies_ghz, nest, nfreqs, nside, nstokes, units, values

.. autofunction:: litebird_sim.maps_and_harmonics.estimate_alm
.. autofunction:: litebird_sim.maps_and_harmonics.pixelize_alm
.. autofunction:: litebird_sim.maps_and_harmonics.interpolate_alm