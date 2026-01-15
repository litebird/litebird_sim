.. _hwp_sys:

HWP systematics
===============

This module implements HWP non-idealities using Mueller’s formalism
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


The class :class:`.hwp_sys.HwpSys` is a container for the parameters
of the HWP systematics. It defines three methods:

* :meth:`.hwp_sys.HwpSys.set_parameters`, which sets the defaults and
  handles the interface with the parameter file of the simulation.There is also the
  possibility of passing precomputed input maps (as a NumPy array)
  through the ``maps`` argument. Otherwise, the code computes input
  maps through the module input_sky (see :ref:`input_sky`). If ``built_map_on_the_fly = True``, the
  map-making can be performed internally on-the-fly;  There are two boolean arguments related to the coupling of non-linearity effects with hwp systematic effects: ``if apply_non_linearity`` and ``add_2f_hwpss``. Applying the non-linearities inside the hwp_sys module is useful when one wants to use the mapmaking on the fly. To know more about detector non-linearity and HWPSS, see `this section <https://litebird-sim.readthedocs.io/en/master/non_linearity.html>`_.

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
    )

    hwp_sys.fill_tod(
        observations=[obs],
        input_map_in_galactic=False,
    )

    output_maps = hwp_sys.make_map([obs])

To couple detector non-linearity with HWP systematics, three attributes must be defined when creating the DetectorInfo objects, and the respective booleans set to True in set_parameters, as in the example below:

.. code-block:: python

   (... importing modules, creating simulation, setting scanning strategy, instrument etc...)

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

HWP emissivity
==============

The module :mod:`.hwp_diff_emiss` implements the additive contamination coming from HWP thermal emission as described in `Micheli et al. 2024 <https://arxiv.org/pdf/2407.15294>`_. This model complements the framework of HWP systematics since this effect cannot be introduced directly with Mueller's formalism.

At finite temperature :math:`T_\textsc{hwp}`, HWPs emit blackbody radiation (in
:math:`\mathrm{W}\,\mathrm{str}^{-1}\,\mathrm{m}^{-2}\,\mathrm{Hz}^{-1}`
units) weighted by their frequency-dependent emissivity, :math:`\varepsilon(\nu)`  

.. math:: \varepsilon(\nu) I_\textsc{hwp}(\nu, T_\textsc{hwp}) = \varepsilon(\nu) \frac{2h\nu^3}{c^2} \frac{1}{e^{\frac{h\nu}{kT_\textsc{hwp}}}-1}\,,

If the optical properties differ
between the ordinary and extraordinary axes, the HWP will also exhibit
differential emissivity (`Pisano et al. 2015 <https://ieeexplore.ieee.org/document/7460631>`_).

In the HWP’s own reference frame, with the :math:`x` and :math:`y` axes
aligned to the ordinary and extraordinary axes respectively, the emitted
radiation can be represented as the Stokes vector

.. math::

   \mathbf{S}_\textsc{hwp}(\nu) = I_\textsc{hwp}(\nu, T_\textsc{hwp})\begin{pmatrix}
           \varepsilon(\nu) \\ \Delta\varepsilon(\nu) \\ 0
       \end{pmatrix}\,,

where :math:`\Delta\varepsilon(\nu)` parametrizes the polarized
component of the emission.

To simplify the model, we consider band-averaged quantities, which are calculated through a sensitivity calculation code.
Thus, for each channel :math:`i`, the Stokes
vector can be written as

.. math::

   \mathbf{S}_{\textsc{hwp},i} = I_{\textsc{hwp},i}(T_\textsc{hwp}) \begin{pmatrix}
           \varepsilon_i \\ \Delta\varepsilon_i \\ 0
       \end{pmatrix}\,.

To determine the corresponding contribution to the TOD, we rotate the Stokes vector from the HWP frame to the detector
frame by an angle :math:`(\xi-\rho)`, where :math:`\rho` and :math:`\xi`
are the HWP and detector orientations in the focal-plane reference
frame. This rotation is denoted by :math:`\mathcal{R}_{\xi-\rho}`, and
the detected signal is then

.. math:: d_{\textsc{hwp},i} = \begin{pmatrix} 1 & 1 & 0 \end{pmatrix} \mathcal{R}_{\xi-\rho} \mathbf{S}_{\textsc{hwp},i}\,.

More explicitly:

.. math::

   \begin{aligned}
       d_{\textsc{hwp},i} &= I_{\textsc{hwp},i}(T_\textsc{hwp}) \begin{pmatrix} 1 & 1 & 0 \end{pmatrix} \begin{pmatrix}
           1 & 0 & 0 \\
           0 & \cos[2(\xi-\rho)] & \sin[2(\xi-\rho)] \\
           0 & -\sin[2(\xi-\rho)] & \cos[2(\xi-\rho)] \\
       \end{pmatrix} \begin{pmatrix}
           \varepsilon_i \\ \Delta\varepsilon_i \\ 0
       \end{pmatrix}\nonumber\\
       &= I_{\textsc{hwp},i}(T_\textsc{hwp}) \begin{pmatrix} 1 & 1 & 0 \end{pmatrix} \begin{pmatrix}
           \varepsilon_i \\ \Delta\varepsilon_i \cos[2(\xi-\rho)] \\ -\Delta\varepsilon_i \sin[2(\xi-\rho)]
       \end{pmatrix}\nonumber\\
       &= \varepsilon_i\,I_{\textsc{hwp},i}(T_\textsc{hwp}) + \Delta\varepsilon_i\, I_{\textsc{hwp},i}(T_\textsc{hwp}) \cos[2(\xi-\rho)]\,.
   \end{aligned}

The first term represents the unpolarized thermal emission, while the 
second term is the polarized emission, modulated at twice the relative
angle between the HWP and detector. 

To simulate the 2f signal from a rotating, emitting HWP, one can use the method of
:class:`.Simulation` class :meth:`.Simulation.add_2f()`, or any of the
low-level functions: :func:`.add_2f_to_observations()`,
:func:`.add_2f_for_one_detector()`.

Examples
-------------

The examples below skip the simulation and observation creation for brevity. If needed, the implementation for those parts is explained in other sections of the docs.

.. code-block:: python

   (... importing modules, creating simulation, setting scanning strategy, instrument, etc...)

   sim.set_hwp(lbs.IdealHWP(hwp_radpsec))

   # creating the detectors (two mock detectors here)
   dets = [
        lbs.DetectorInfo(name="det_A", sampling_rate_hz=sampling_hz),
        lbs.DetectorInfo(name="det_B", sampling_rate_hz=sampling_hz),
    ]

   (obs,) = sim.create_observations(
      detectors=dets,
   )

   sim.prepare_pointings()

   # Define differential emission amplitude for the detectors, in the same temperature units you used for the TOD. 
   # If just one value is set, all the detectors will have the same 2f amplitude. 
   sim.observations[0].amplitude_2f_k = np.array([0.1, 0.2])

   # Adding 2f signal from HWP differential emission using the `Simulation` class method
   # By default, the TOD is added to ``Observation.tod``. If you want to add it to some
   # other field of the :class:`.Observation` class, use `component`

   sim.add_2f()


API reference
-------------

HWP_sys
~~~~~~~

.. automodule:: litebird_sim.hwp_sys.hwp_sys
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource

.. automodule:: litebird_sim.hwp_diff_emiss
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource
