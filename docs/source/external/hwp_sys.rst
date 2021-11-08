.. _hwp_sys:

HWP_sys
=======

This module implements HWP non-idealities in 2-dimensional Jones calculus as 
described in `Giardiello et al. 2021 <https://arxiv.org/abs/2106.08031>`_. In
this formalism a non-ideal HWP is described by

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

In the most general case these parameters can vary inside a bandpass.

The class :class:`.hwp_sys.HwpSys` is a contanier for the parameters of the HWP systematics.
It defines three methods:

*  :func:`.HwpSys.set_parameters` which sets the defaults and handles the interface with 
   the parameter file of the simulation. The relevant section is tagged by [hwp_sys].

*  :func:`.HwpSys.fill_tod` which fills the tod in a given Observation.

*  :func:`.HwpSys.make_map` which can bin the observations in a map. This is available only
   if `built_map_on_the_fly` variable is set to True.

API reference
-------------

.. automodule:: litebird_sim.hwp_sys.hwp_sys
    :members:
    :undoc-members:
    :show-inheritance: