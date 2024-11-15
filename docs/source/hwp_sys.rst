.. _hwp_sys:

HWP systematics
===============

This module implements HWP non-idealities both using Jones’ formalism
(as described in `Giardiello et al. 2021
<https://arxiv.org/abs/2106.08031>`_) and Mueller’s. In Jones’
formalism, a non-ideal HWP is described by

.. math::
   :label: non_ideal_hwp_jones

   J_{\text{HWP}} = \begin{pmatrix} 1+h_{1} & \zeta_{1} e^{i \chi_1}\\ \zeta_{2} e^{i \chi_2}& -(1+h_{2}) e^{i \beta} \\ \end{pmatrix}

where:

*  :math:`h_1` and :math:`h_2` are the efficiencies, describing the deviation from the unitary
   transmission of light components :math:`E_x`, :math:`E_y`. In the ideal case,
   :math:`h_1 = h_2 = 0`;

*  :math:`\beta=\phi-\pi`, where :math:`\phi` is the phase shift between the two
   directions. It accounts for variations of the phase difference between :math:`E_x`
   and :math:`E_y` with respect to the nominal value of :math:`\pi` for an ideal HWP.
   In the ideal case, :math:`\beta=0`;

*  :math:`\zeta_{1,2}` and :math:`\chi_{1,2}` are amplitudes and phases of the
   off-diagonal terms, coupling :math:`E_x` and :math:`E_y`. In practice, if the
   incoming wave is fully polarized along x(y), a spurious y(x) component would
   show up in the outgoing wave. In the ideal case, :math:`\zeta_{1,2}=\chi_{1,2}=0`.

In the Mueller formalism, we have a general matrix

.. math::
   :label: non_ideal_hwp_mueller

   J_{\text{HWP}} = \begin{pmatrix} M^{TT} & M^{TQ} & M^{TU} \\ M^{QT} & M^{QQ} & M^{QU} \\ M^{UT} & M^{UQ} & M^{UU} \\ \end{pmatrix}

which, in the ideal case, would be

.. math::
   :label: ideal_hwp_mueller

   J_{\text{HWP}} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \\ \end{pmatrix}

In the most general case, the Jones non-ideal parameters and the Mueller matrix elements can vary inside a bandpass.

The class :class:`.hwp_sys.HwpSys` is a container for the parameters
of the HWP systematics. It defines three methods:

* :meth:`.hwp_sys.HwpSys.set_parameters`, which sets the defaults and
  handles the interface with the parameter file of the simulation. The
  relevant section is tagged by ``[hwp_sys]``. The HWP parameters can
  be passed both in the Jones' and Mueller's formalism. This choice is
  regulated by the flag ``mueller_or_jones``, which can have the
  values ``"mueller"`` or ``"jones"``. In case the Jones formalism is
  chosen, it is converted automatically into the Mueller one through
  the function ``.hwp_sys.JonesToMueller``. There is also the
  possibility of passing precomputed input maps (as a NumPy array)
  through the ``maps`` argument. Otherwise, the code computes input
  maps through the module Mbs (see :ref:`Mbs`). The argument
  ``integrate_in_band`` sets whether to perform band integration in
  the TOD computation; if ``built_map_on_the_fly = True``, the
  map-making can be performed internally (instead of using the
  litebird_sim binner); ``correct_in_solver`` sets whether non-ideal
  parameters can be used in the map-making (map-making assuming a
  non-ideal HWP, generally using different HWP non-ideal parameters
  than the one used in the TOD, representing our estimate of their
  true value); ``integrate_in_band_solver`` regulates whether band
  integration is performed in the map-making (to compute the
  :math:`B^T B` and :math:`B^T d` terms, see below).

*  :meth:`.hwp_sys.HwpSys.fill_tod` which fills the tod in a given Observation. The ``pointings``
   angles passed have to include no rotating HWP, since the effect of the rotating HWP to the
   polarization angle is included in the TOD computation.
   The TOD is computed performing this operation:

   .. math::

      d_{\text{obs}}\left(t_{i}\right)\,=\,\frac{\int d\nu\,\frac{\partial BB(\nu,T)}{\partial T_{\text{CMB}}}\,\tau\left(\nu\right)\,M_{i}^{TX}(\nu)\left(m_{\text{CMB}}+m_{\text{FG}}\left(\nu\right)\right)}{\int d\nu \frac{\partial BB(\nu,T)}{\partial T_{\text{CMB}}}\,\tau \left(\nu\right)},

   where :math:`\tau(\nu)` is the bandpass,
   :math:`\frac{\partial BB(\nu,T)}{\partial T_{\text{CMB}}}` converts from CMB thermodynamic temperature
   to differential source intensity (see eq.8 of https://arxiv.org/abs/1303.5070) and
   :math:`M_{i}^{TX}(\nu)` is the Mueller matrix element including the non-ideal HWP.

   If ``built_map_on_the_fly = True``, the code computes also

   .. math::

      m_{\text{out}} = {\,\left(\sum_{i} B_{i}^{T} B_{i} \right)^{-1} \left( \sum_{i} B_{i}^{T} d_{\text{obs}}(t_{i}) \right)},

   where the map-making matrix is

   .. math::

      B^X = \left(\frac{\int d\nu \,\frac{\partial BB(\nu,T)}{\partial T_{\text{CMB}}}\,\tau_{s}\left(\nu\right)\,M_{i,s}^{TX}(\nu)}{\int d\nu \frac{\partial BB(\nu,T)}{\partial T_{\text{CMB}}}\,\tau_{s}\left(\nu\right)}\,\right).

   :math:`\tau_s(\nu)` and :math:`M_{i,s}^{TX}(\nu)` are the estimate of
   the bandpass and Mueller matrix elements used in the map-making.

*  :meth:`.hwp_sys.HwpSys.make_map` which can bin the observations in a map. This is available only
   if ``built_map_on_the_fly`` variable is set to ``True``. With this method, it is possible to
   include non-ideal HWP knowledge in the map-making procedure, so use that instead of the general
   ``litebird_sim`` binner if you want to do so.

Defining a bandpass profile in ``hwp_sys``
------------------------------------------

It is possible to define more complex bandpass profiles than a top-hat when using ``hwp_sys``.
This can be done both for the TOD computation (:math:`\tau`) and the map-making procedure
(:math:`\tau_s`). All you have to do is create a dictionary with key "hwp_sys" in the parameter
file (a toml file) assigned to the simulation:

.. code-block:: python

  sim = lbs.Simulation(
      parameter_file=toml_filename,
      random_seed=0,
   )

The dictionary under the key "hwp_sys" will also contain the paths to the files from which the HWP
parameters are read in the multifrequency case (under the keys "band_filename/band_filename_solver"), or their values in the single frequency one. See the notebook ``hwp_sys/examples/simple_scan``
for more details.
To define the bandpasses to use, you need to have a dictionary with key "bandpass"
(for :math:`\tau`) or "bandpass_solver" (for :math:`\tau_s`) under "hwp_sys":

.. code-block:: python

  paramdict = {...
                "hwp_sys": {...
                   "band_filename": path_to_HWP_param_file,
                   "band_filename_solver": path_to_HWP_solver_param_file,
                   "bandpass": {"band_type": "cheby",
                                "band_low_edge": band_low_edge,
                                "band_high_edge": band_high_edge,
                                "bandcenter_ghz": bandcenter_ghz,
                                "band_ripple_dB": ripple_dB_tod,
                                "band_order": args.order_tod},
                   "bandpass_solver": {...},
                 ...}}

The above example is for a bandpass with Chebyshev filter, but there are other parameters to define
different bandpass profile. It is important to define the "band_type", which can be "top-hat",
"top-hat-exp", "top-hat-cosine" and "cheby" (see the ``bandpass`` module for more details)
and the band edges, which define the frequency range over which the bandpass transmission
is close or equal to 1. If not assigned, the "band_type" is automatically set to "top-hat"
and the band edges will correspond to the limits of the frequency array used (which, in the
``hwp_sys`` module, is read from the HWP parameter files). There are default values also for
the parameters defining the specific bandpass profiles (see the
``hwp_sys/hwp_sys/bandpass_template_module`` code).

There is also the possibility to read the bandpass profile from an external file, which has to be
a .txt file with two columns, the frequency and the bandpass transmission. It is important that the
frequency array used for "bandpass/bandpass_solver" coincides with the ones passed in the
"band_filename/band_filename_solver" file. Here is how to pass the bandpass file:

.. code-block:: python

  paramdict = {...
                "hwp_sys": {...
                   "band_filename": path_to_HWP_param_file,
                   "band_filename_solver": path_to_HWP_solver_param_file,
                   "bandpass": {"bandpass_file": path_to_bandpass_file },
                   "bandpass_solver": {"bandpass_file": path_to_bandpass_solver_file},
                 ...}}

You can find more examples for the bandpass construction in the ``hwp_sys/examples/simple_scan`` notebook.

API reference
-------------

HWP_sys
~~~~~~~

.. automodule:: litebird_sim.hwp_sys.hwp_sys
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource

Bandpass template
~~~~~~~~~~~~~~~~~

.. automodule:: litebird_sim.hwp_sys.bandpass_template_module
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource
