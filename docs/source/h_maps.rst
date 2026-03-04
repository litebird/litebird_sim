.. _h-maps:

Harmonic Maps (h_maps)
======================

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

The ``h_maps`` module provides tools to generate **complex harmonic maps**
:math:`h_{n,m}` from LiteBIRD observations. These maps are built by
accumulating spin-weighted quantities over the sky pixels hit during a
scanning strategy, for each detector and time split.

The harmonic map :math:`h_{n,m}` is defined as a pixel-domain map of the
quantity:

.. math::

   h_{n,m}(p) = \frac{1}{N_p} \sum_{t \in p}
   e^{i (n \psi_t + m \phi_{\mathrm{HWP},t})}

where :math:`\psi_t` is the polarization angle at time :math:`t`,
:math:`\phi_{\mathrm{HWP},t}` is the HWP angle, and :math:`N_p` is the
number of hits in pixel :math:`p`.

.. note::

   The map :math:`h_{0,0}` contains the **hit counts** per pixel instead
   of the mask of hit pixels expected from the strict mathematical definition.
   This is useful when combining h maps from different detectors or splits.

Basic usage
-----------

.. testcode::

   import litebird_sim as lbs
   import numpy as np
   from astropy.time import Time

   # Create a simple simulation
   sim = lbs.Simulation(
       start_time=Time("2025-01-01"),
       duration_s=1000.0,
       random_seed=12345,
   )

   # ... (set up detectors, scanning strategy, observations)

   result = lbs.mapmaking.make_h_maps(
       observations=sim.observations,
       nside=64,
       n_m_couples=np.array([[0, 0], [2, 0], [4, 0]]),
       output_coordinate_system=lbs.CoordinateSystem.Galactic,
       save_to_file=True,
       output_directory="./h_n_maps",
   )

The ``n_m_couples`` parameter
-----------------------------

The spin orders to compute are specified as a 2D array of ``(n, m)`` pairs::

   # Compute h_{0,0}, h_{2,0} and h_{4,0}
   n_m_couples = np.array([
       [0, 0],
       [2, 0],
       [4, 0],
   ])

The default value uses ``np.meshgrid`` to generate all combinations of
``n ∈ {0, 2, 4}`` and ``m = 0``.

Output: ``HnMapResult``
-----------------------

:func:`.make_h_maps` returns a :class:`.HnMapResult` object, which contains:

- ``h_maps``: a dictionary indexed by detector name, then by ``(n, m)``
  tuple, each entry being a :class:`.h_map_Re_and_Im` object.
- ``coordinate_system``: the output coordinate system
  (a :class:`.CoordinateSystem` object).
- ``detector_split``: the detector split used.
- ``time_split``: the time split used.
- ``duration``: the duration of the observation.
- ``sampling_rate``: the sampling rate of the observation.

Accessing individual maps
~~~~~~~~~~~~~~~~~~~~~~~~~

Each entry in ``h_maps`` is a :class:`.h_map_Re_and_Im` object with:

- ``real``: the real part of :math:`h_{n,m}` (NumPy array of shape ``(Npix,)``)
- ``imag``: the imaginary part (NumPy array of shape ``(Npix,)``)
- ``norm``: property returning :math:`\sqrt{\mathrm{Re}^2 + \mathrm{Im}^2}`,
  with unobserved pixels set to :attr:`.UNSEEN_PIXEL_VALUE`
- ``n``, ``m``: the spin orders
- ``det_info``: the detector name

Example::

   h_2_0 = result.h_maps["detector_name"][2, 0]
   print(h_2_0.real)   # real part
   print(h_2_0.imag)   # imaginary part
   print(h_2_0.norm)   # amplitude

Saving and loading maps
-----------------------

Maps are saved in **HDF5** format, one file per detector:

.. code-block:: text

   output_directory/
   └── h_maps_det_{detector_name}.h5
       ├── attrs: coordinate_system, det, detector_split,
       │          time_split, duration, sampling_rate
       ├── 0,0/
       │   ├── Re
       │   └── Im
       ├── 2,0/
       │   ├── Re
       │   └── Im
       └── ...

To reload maps from disk, use :func:`.load_h_map_from_file`::

   from litebird_sim.mapmaking.h_maps import load_h_map_from_file

   result = load_h_map_from_file("./h_n_maps/h_maps_det_mydetector.h5")

Detector and time splits
------------------------

The ``detector_split`` and ``time_split`` parameters follow the same
conventions as the rest of the litebird_sim mapmaking framework (see
:ref:`mapmaking`). Use ``"full"`` (default) to include all detectors
and all time samples.

MPI support
-----------

If observations are distributed over MPI processes, all processes in the
same communicator group must call :func:`.make_h_maps` collectively.

API reference
-------------

.. autoclass:: litebird_sim.mapmaking.h_maps.h_map_Re_and_Im
    :members:
    :undoc-members:

.. autoclass:: litebird_sim.mapmaking.h_maps.HnMapResult
    :members:
    :undoc-members:

.. autofunction:: litebird_sim.mapmaking.h_maps.make_h_maps

.. autofunction:: litebird_sim.mapmaking.h_maps.save_hn_maps

.. autofunction:: litebird_sim.mapmaking.h_maps.load_h_map_from_file