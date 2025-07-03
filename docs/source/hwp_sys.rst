.. _hwp_sys:

HWP systematics
===============

This module implements HWP non-idealities using Mueller’ formalism
(as described in `Patanchon et al. 2023
<https://arxiv.org/pdf/2308.00967>`_). The Mueller matrix describing the effect of a HWP in the signal received by a detector can be expanded into the harmonics of the HWP rotation frequency f, from which we pick the 0th, 2nd and 4th terms.

.. math::
   :label: expansion_into_harmonics

   M^{\text{HWP}}_{ij} = M^{0f}_{ij}(\Theta) + M^{2f}_{ij}(\Theta, \Phi, 2\rho) + M^{4f}_{ij}(\Theta, \Phi, 4\rho)

where:

*  :math:`\Theta` is the incidence angle between the center of the HWP and the detector position;

*  :math:`\Phi` is the angle describing the azimuthal position of the detector in the focal plane;

*  :math:`\rho` is the HWP rotation angle in **focal plane coordinates** (please note that this is not the case in `Patanchon et al. 2023 <https://arxiv.org/pdf/2308.00967>`_.) .


The :math:`\Theta` dependence is included in the input matrices, which are then multiplied at each sample by the trigonometric term depending harmonically on :math:`\rho` and :math:`\Phi`, such that:

.. math::
   :label: cosine_terms

   M^{0f}_{ij} = M_{input_{ij}}^{0f}

   M^{2f}_{ij} = M_{input_{ij}}^{2f} \cos(2\rho - 2\Phi - \phi_{ij}^{2f})

   M^{4f}_{ij} = M_{input_{ij}}^{4f} \cos(4\rho - 4\Phi - \phi_{ij}^{4f})


where :math:`\phi_{ij}` are harmonic and element dependent phases obtained trough EM simulations.   Because the Mueller matrix for an ideal HWP (for a detector at azimuthal angle :math:`\Phi = 0`) is given by:

.. math::
   :label: non_ideal_hwp_mueller

   M_{\text{IdealHWP}} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos{4\rho} & \sin{4\rho} \\ 0 & \sin{4\rho} & -\cos{4\rho} \end{pmatrix}

which, in the HWP reference frame (:math:`\rho = 0`) yields the usual matrix for an ideal HWP (diagonal 1, 1 -1 terms), we can now see that an ideal HWP is obtained by setting:


.. math::
   :label: non_ideal_hwp_mueller_2

   M^{0f}_{input_{ideal}} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

   M^{2f}_{input_{ideal}} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

   M^{4f}_{input_{ideal}} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}


The expected sin and negative cosine terms are accounted for by the phases :math:`\phi` in the expressions shown above, which is (in an ideal case) :math:`\pi` for the 4f UU element, and to :math:`\frac{\pi}{2}` for IU and UQ (sin terms).

Changing one of the values in the matrices above represents a non-ideality in the hwp.


Jones Formalism
----------------


In the Jones formalism, a non-ideal static HWP can be described by perturbations :math:`\delta_{ij}` to each ij element of the ideal HWP Jones Matrix, i.e.:

.. math::
   J = \begin{pmatrix} 1+\delta_{11} & \delta_{11} \\ \delta_{21}  & -1 + \delta_{22}\end{pmatrix}


These perturbations can be related to the Mueller matrix elements through the relation:

.. math::
   M_{ij} = \frac{1}{2} Tr(\sigma_i J \sigma_j J^{\dagger})

that is, for each element of the matrix:

.. math::
   M_{II} = Re(\delta_{11} - \delta_{22} + 1)

   M_{QI}=Re(\delta_{11} + \delta_{12})

   M_{UI}=Re(\delta_{21} - \delta_{12})

   M_{IQ}=Re(\delta_{11} + \delta_{12})

   M_{IU}=Re(\delta_{12} - \delta_{21})

   M_{QQ}=Re(\delta_{12} - \delta_{22} + 1)

   M_{UU}=Re(\delta_{22} + \delta_{11} - 1)

   M_{UQ}=Re(\delta_{12} + \delta_{21})

   M_{QU}=Re(\delta_{12} + \delta_{21})


We expand each of the perturbations $\delta_{pq}$ into harmonics of the rotation frequency as:

.. math::
    \delta_{pq} = \sum_n \gamma_{pq}^{n_{f}} exp(in\alpha)


where :math:`p`, :math:`q` are the matrix indexes, :math:`n_{f} = 0,2,4` are the harmonics and :math:`\alpha` is the HWP angle in focal plane coordinates (assuming for simplicity that the azimuthal angle of the detector is set to 0, and ignoring the dependence on the incidence angle). If our input matrices have the :math:`\gamma` values as complex numbers :math:`A e^{i \phi}`, for each harmonic, we get:

.. math::
    \delta_{pq}^{n_{f}}( n_{f} \alpha) = \sum_{n_{f}} A_{pq}^{n_{f}} e^{i\phi} e^{in\alpha} 


Developing this and taking only the real part we get to:

.. math::
    \delta_{pq}^{n_{f}}(n_{f} \alpha) = \sum_{n_{f}} A_{pq}^{n_{f}} \  cos(n\alpha + n\phi_{ij}^{n_{f}})

In terms of implementation on lbs, this is done in 3 steps:

Then, the obtain matrix is rotated from the hwp frame to the focal plane frame, in order to apply the mueller formalism explained above. In an ideal case, the jones perturbation matrices will be composed of only zero values and the transformation will yield the ideal hwp mueller matrix.


The class :class:`.hwp_sys.HwpSys` is a container for the parameters
of the HWP systematics. It defines three methods:

* :meth:`.hwp_sys.HwpSys.set_parameters`, which sets the defaults and
  handles the interface with the parameter file of the simulation.There is also the
  possibility of passing precomputed input maps (as a NumPy array)
  through the ``maps`` argument. Otherwise, the code computes input
  maps through the module Mbs (see :ref:`Mbs`). ``mueller_or_jones`` sets the operations to be performed internally depending on the type of input matrices representing the hwp. If ``built_map_on_the_fly = True``, the map-making can be performed internally on-the-fly;  There are two boolean arguments related to the coupling of non-linearity effects with hwp systematic effects: ``if apply_non_linearity`` and ``add_2f_hwpss``. Applying the non-linearities inside the hwp_sys module is useful when one wants to use the mapmaking on the fly. To know more about detector non-linearity and HWPSS, see `this section <https://litebird-sim.readthedocs.io/en/master/non_linearity.html>`_.

*  :meth:`.hwp_sys.HwpSys.fill_tod` which fills the tod in a given Observation. The ``pointings``
   angles passed have to include no rotating HWP, since the effect of the rotating HWP to the
   polarization angle is included in the TOD computation.
   The TOD is computed by the **equation 5.2** in `Patanchon et al. 2023 <https://arxiv.org/pdf/2308.00967>`_.

   If ``built_map_on_the_fly = True``, the code computes also

   .. math::

      m_{\text{out}} = {\,\left(\sum_{i} B_{i}^{T} B_{i} \right)^{-1} \left( \sum_{i} B_{i}^{T} d_{\text{obs}}(t_{i}) \right)},

   where the map-making matrix is given by the user for each detector in the mueller_hwp_solver attribute of the DetectorInfo class.

*  :meth:`.hwp_sys.HwpSys.make_map` which can bin the observations in a map. This is available only
   if ``built_map_on_the_fly`` variable is set to ``True``. With this method, it is possible to
   include non-ideal HWP knowledge in the map-making procedure.


Examples
-------------

The examples below skip the simulation and observation creation for brevity. If needed, the implementation for those parts are explained in other sections of the docs.

.. code-block:: python

   (... importing modules, creating simulation, seetting scanning strategy, instrument etc...)

   sim.set_hwp(lbs.IdealHWP(hwp_radpsec))

   # creating the detectors (DetectorInfo objects - only one in this example)
   det = lbs.DetectorInfo.from_imo(...)

   # defining the mueller matrices for the HWP for this detector.
   # In this example, we set non-idealities in the IP leakage terms.
   det.mueller_hwp = {
            "0f": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "2f": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "4f": np.array([[4*1e-4, 0, 0], [3*1e-4, 1, 1], [0, 1, 1]]),
        }

   # defining the mueller matrices for the mapmaking (pointing matrix) for this detector
   # In this example, we consider an ideal pointing matrix. This could be left empty as it is the default case.
   det.mueller_hwp_solver = {
            "0f": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "2f": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "4f": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]),
        }

   (obs,) = sim.create_observations(
      detectors=dets,
   )

   sim.prepare_pointings()

   # creating the HwpSys object 
   hwp_sys = lbs.HwpSys(sim)

   # setting HwpSys parameters
   hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        mbs_params=mbs_params,
        build_map_on_the_fly=True, #if one wants to perform the mapmaking on-the-fly
        comm=sim.mpi_comm,
        mueller_or_jones="mueller",
    )

    hwp_sys.fill_tod(
        observations=[obs],
        input_map_in_galactic=False,
    )

    output_maps = hwp_sys.make_map([obs])


We can also give perturbation jones matrices as inputs, and set mueller_or_jones to "mueller", and the conversion to the Mueller matrices will be done internally.

To couple detector non-linearity with HWP systematics, three attributes must be defined when creating the DetectorInfo objects, and the respective booleans set to True in set_parameters, as in the example below:

.. code-block:: python

   (... importing modules, creating simulation, seetting scanning strategy, instrument etc...)

   sim.set_hwp(lbs.IdealHWP(hwp_radpsec))

   det = lbs.DetectorInfo.from_imo(...)

   det.mueller_hwp = {...}
   det.mueller_hwp_solver = {...}

   det.g_one_over_k = -0.144
   det.amplitude_2f_k = 2.0
   det.optical_power_k = 1.5

   (obs,) = sim.create_observations(
      detectors=dets,
   )

   sim.prepare_pointings()

   # creating the HwpSys object 
   hwp_sys = lbs.HwpSys(sim)

   # setting HwpSys parameters
   hwp_sys.set_parameters(
        nside=nside,
        maps=input_maps,
        Channel=channelinfo,
        mbs_params=mbs_params,
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
