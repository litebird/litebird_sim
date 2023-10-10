.. _hwp_sys:

HWP_sys
=======

This module implements HWP non-idealities both using Jones formalism (as 
described in `Giardiello et al. 2021 <https://arxiv.org/abs/2106.08031>`_) and the Mueller one. In
the Jones formalism, a non-ideal HWP is described by

.. math::
   :label: non_ideal_hwp_jones

   J_{HWP} = \begin{pmatrix} 1+h_{1} & \zeta_{1} e^{i \chi_1}\\ \zeta_{2} e^{i \chi_2}& -(1+h_{2}) e^{i \beta} \\ \end{pmatrix}

where:

*  h1 and h2 are the efficiencies, describing the deviation from the unitary
   transmission of light components :math:`E_x`, :math:`E_y`. In the ideal case, 
   :math:`h1 = h2 = 0`;
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

   J_{HWP} = \begin{pmatrix} M^{TT} & M^{TQ} & M^{TU} \\ M^{QT} & M^{QQ} & M^{QU} \\ M^{UT} & M^{UQ} & M^{UU} \\ \end{pmatrix}

which, in the ideal case, would be

.. math::
   :label: non_ideal_hwp_mueller

   J_{HWP} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \\ \end{pmatrix}

In the most general case, the Jones non-ideal parameters and the Mueller matrix elements can vary inside a bandpass.

The class :class:`.hwp_sys.HwpSys` is a contanier for the parameters of the HWP systematics.
It defines three methods:

*  :meth:`.hwp_sys.HwpSys.set_parameters` which sets the defaults and handles the interface with 
   the parameter file of the simulation. The relevant section is tagged by ``[hwp_sys]``. The HWP 
   parameters can be passed both in the Jones and Mueller formalism. This choice is regulated by 
   the flag `mueller_or_jones`, which has to be set to either ``"mueller"`` or ``"jones"``. In
   case the Jones formalism is chosen, it is converted automatically into the Mueller one through
   the function `.hwp_sys.JonesToMueller`. There is also the possibility to pass precomputed input 
   maps (as a numpy array) through the `maps` argument. Otherwise, input maps are internally 
   computed through the litebird_sim map based simulator (mbs).
   The argument `integrate_in_band` sets whether band integration is performed in the TOD 
   computation; if ``built_map_on_the_fly = True``, the map-making can be performed internally
   (instead of using the litebird_sim binner); `correct_in_solver` sets whether non-ideal 
   parameters can be used in the map-making (map-making assuming a non-ideal HWP, generally using 
   different HWP non-ideal parameters thatn the one used in the TOD, representing our estimate of
   their true value); `integrate_in_band_solver` regulates whether band integration is performed
   in the map-making (to compute the :math:`B^T B` and :math:`B^T d` terms, see below).  


*  :meth:`.hwp_sys.HwpSys.fill_tod` which fills the tod in a given Observation. The ``pointings`` 
   angles passed have to include no rotating HWP, since the effect of the rotating HWP to the 
   polarization angle is included in the TOD computation.
   The TOD is computed performing this operation:

   :math:`d_{obs}\left(t_{i}\right)\,=\,\frac{\int d\nu\,\frac{\partial BB(\nu,T)}{\partial T_{CMB}}\,\tau\left(\nu\right)\,M_{i}^{TX}(\nu)\left(m_{CMB}+m_{FG}\left(\nu\right)\right)}{\int d\nu \frac{\partial BB(\nu,T)}{\partial T_{CMB}}\,\tau \left(\nu\right)}`, where :math:`\tau` is the bandpass, 
   :math:`\frac{\partial BB(\nu,T)}{\partial T_{CMB}}` converts from CMB thermodynamic temperature 
   to differential source intensity (see eq.8 of https://arxiv.org/abs/1303.5070) and 
   :math:`M_{i}^{TX}(\nu)` is the Mueller matrix element including the non-ideal HWP.

   If `built_map_on_the_fly = True`, the code computes also :math:`m_{out} = {\,\left(\sum_{i} B_{i}^{T} B_{i} \right)^{-1} \left( \sum_{i} B_{i}^{T} d_{obs}(t_{i}) \right)}`, where the map-making
   matrix :math:`B^X = \left(\frac{\int d\nu \,\frac{\partial BB(\nu,T)}{\partial T_{CMB}}\,\tau_{s}\left(\nu\right)\,M_{i,s}^{TX}(\nu)}{\int d\nu \frac{\partial BB(\nu,T)}{\partial T_{CMB}}\,\tau_{s}\left(\nu\right)}\,\right)`, where :math:`\tau_s` and :math:`M_{i,s}^{TX}(\nu)` are the estimate of
   the bandpass and Mueller matrix elements used in the map-making.

*  :meth:`.hwp_sys.HwpSys.make_map` which can bin the observations in a map. This is available only
   if `built_map_on_the_fly` variable is set to ``True``. With this method, it is possible to 
   include non-ideal HWP knowledge in the map-making procedure.

API reference
-------------

.. automodule:: litebird_sim.hwp_sys.hwp_sys
    :members:
    :undoc-members:
    :show-inheritance:
    :private-members:
