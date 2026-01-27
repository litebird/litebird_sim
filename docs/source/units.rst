.. _units:

Unit Management and Conversions
===============================

This module provides a centralized system for handling physical units, constants, and conversions
within the ``litebird_sim`` pipeline. It ensures consistency between thermodynamic 
temperatures (:math:`K_{CMB}`), brightness temperatures (:math:`K_{RJ}`), and flux densities.

The module leverages ``pysm3.units`` and ``astropy.units`` to perform physically accurate 
conversions including frequency-dependent equivalencies.

Supported Units
---------------

The following units are supported via the :class:`Units` Enum:

.. autoclass:: litebird_sim.units.Units
   :members:
   :undoc-members:
   :member-order: bysource

Utility Functions
-----------------

The :class:`UnitUtils` class provides static methods to handle conversion logic and 
plot labeling.

.. autoclass:: litebird_sim.units.UnitUtils
   :members:
   :show-inheritance:

Examples
--------

Converting from :math:`\mu K_{CMB}` to :math:`\mathrm{MJy/sr}` at 100 GHz:

.. code-block:: python

    from litebird_sim.units import Units, UnitUtils

    freq = 100.0  # GHz
    factor = UnitUtils.get_conversion_factor(Units.uK_CMB, Units.MJy_over_sr, freq_ghz=freq)
    print(f"Conversion factor: {factor}")

Getting a LaTeX label for a plot:

.. code-block:: python

    label = UnitUtils.get_label(Units.uK_RJ)
    # Returns: r"$\mu K_{\mathrm{RJ}}$"
