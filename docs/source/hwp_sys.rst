.. _hwp_sys:

HWP systematics
===============

This module implements HWP non-idealities using Mueller’ formalism
(as described in `Patanchon et al. 2023
<https://arxiv.org/pdf/2308.00967>`_). The Mueller matrix describing the effect of a HWP in the signal received by a detector can be expanded into the harmonics of the HWP rotation frequency f, from which we pick the 0th, 2nd and 4th terms.

.. math::
   :label: expansion_into_harmonics

   M^{\text{HWP}}_{ij} = M^{0f}_{ij}(\Theta) + M^{2f}_{ij}(\Theta, \Phi, 2\rho) + M^{2f}_{ij}(\Theta, \Phi, 4\rho)

where:

*  :math:`\Theta` is the incidence angle between the center of the HWP and the detector position;

*  :math:`\Phi` is the angle describing the azimuthal position of the detector in the focal plane;

*  :math:`\rho` is the HWP rotation angle in **focal plane coordinates** (please note that this is not the case in Patanchon et al. 2023) .


The :math:`\Theta` dependence is included in the input matrices, which are then multiplied at each sample by the trigonometric term depending harmonically on :math:`\rho` and :math:`\Phi`, such that:

.. math::
   :label: cosine_terms

   M^{0f}_{ij} = M_{input_{ij}}^{0f}

   M^{2f}_{ij} = M_{input_{ij}}^{2f} \cos{2\rho - 2\Phi - \phi_{ij}^{2f}}

   M^{4f}_{ij} = M_{input_{ij}}^{4f} \cos{4\rho - 4\Phi - \phi_{ij}^{4f}}


where :math:`\phi_{ij}` are harmonic and element dependent phases obtained trough EM simulations.   Because the Mueller matrix for an ideal HWP (for a detector at azimuthal angle :math:`\Phi = 0`) is given by:

.. math::
   :label: non_ideal_hwp_mueller

   M_{\text{IdealHWP}} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos{4\rho} & \sin{4\rho} \\ 0 & \sin{4\rho} & -\cos{4\rho} \end{pmatrix}

which, in the HWP reference frame (:math:`\rho = 0`) yields the usual matrix for an ideal HWP (diagonal 1, 1 -1 terms), we can now see that an ideal HWP is obtained by setting:


.. math::
   :label: non_ideal_hwp_mueller

   M^{0f}_{input_{ideal}} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

   M^{2f}_{input_{ideal}} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

   M^{4f}_{input_{ideal}} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}


The expected sin and negative cosine terms are accounted for by the phases :math:`\phi` in the expressions shown above, which is (in an ideal case) :math:`\pi` for the 4f UU element, and to :math:`\frac{\pi}{2}` for IU and UQ (sin terms).

Changing one of the values in the matrices above represents a non-ideality in the hwp.


The class :class:`.hwp_sys.HwpSys` is a container for the parameters
of the HWP systematics. It defines three methods:

* :meth:`.hwp_sys.HwpSys.set_parameters`, which sets the defaults and
  handles the interface with the parameter file of the simulation.There is also the
  possibility of passing precomputed input maps (as a NumPy array)
  through the ``maps`` argument. Otherwise, the code computes input
  maps through the module Mbs (see :ref:`Mbs`). If ``built_map_on_the_fly = True``, the
  map-making can be performed internally on-the-fly;  There are three boolean arguments related to the coupling of non-linearity effects with hwp systematic effects: ``if apply_non_linearity``, ``add_orbital_dipole`` and ``add_2f_hwpss``. 

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


Examples
-------------

The examples below skip the simulation and observation creation for brevity. If needed, the implementation for those parts are explained in other sections of the docs.

.. code-block:: python

   (... creating simulation, observation, etc...)
   (... DON'T FORGET to add hwp to the simulation with sim.set_hwp...)

   # creating the HwpSys object 
   hwp_sys = lbs.HwpSys(sim)

   # setting HwpSys parameters
   hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        Mbsparams=Mbsparams,
        build_map_on_the_fly=True, #if one wants to perform the mapmaking on-the-fly
        comm=sim.mpi_comm,
    )

    hwp_sys.fill_tod(
        observations=[obs],
        input_map_in_galactic=False,
    )

    output_maps = hwp_sys.make_map([obs])

To couple detector non-linearity with HWP systematics, three attributes must be defined when creating the DetectorInfo objects, and the respective booleans set to True in set_parameters, as in the example below:

.. code-block:: python

   (... creating simulation, observation, etc...)
   (... DON'T FORGET to add hwp to the simulation with sim.set_hwp...)

   det = lbs.DetectorInfo.from_imo(...)

   det.g_one_over_k = -0.144
   det.amplitude_2f_k = 2.0
   det.optical_power_k = 1.5

   (obs,) = sim.create_observations(
      detectors=dets,
   )

   (...)

   # creating the HwpSys object 
   hwp_sys = lbs.HwpSys(sim)

   # setting HwpSys parameters
   hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        Mbsparams=Mbsparams,
        build_map_on_the_fly=True, #if one wants to perform the mapmaking on-the-fly
        apply_non_linearity=True,
        add_orbital_dipole=True,
        add_2f_hwpss=True,
        comm=sim.mpi_comm,
    )

    hwp_sys.fill_tod(
        observations=[obs],
        input_map_in_galactic=False,
    )

    output_maps = hwp_sys.make_map([obs])

API reference
-------------

HWP_sys
~~~~~~~

.. automodule:: litebird_sim.hwp_sys.hwp_sys
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource
