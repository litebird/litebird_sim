.. _hwp_sys:

HWP Systematics
===============

The description and simulation of Half-Wave Plate (HWP) systematic effects in a realistic manner including explicitly the influence of the harmonics of the rotation frequency of the HWP.
However, in simple cases, a description of the HWP systematics that does not take the harmonics explicitly into account can be applied. For this reason, litebird_sim includes two ways of simulating the systematics effects of a HWP.

The simpler case is already described in `here <https://litebird-sim.readthedocs.io/en/master/map_scanning.html#equation-generichwp>`_, where a given hwp mueller matrix is directly used to compute the TOD with the `scan_map() https://litebird-sim.readthedocs.io/en/master/map_scanning.html#litebird_sim.scan_map.scan_map`_ routine.

The explicit harmonics case is explained below. The high-level way of choosing between one or the other description is by setting the attribute harmonic_expansion to True or False when instantiating the HWP object. See more about this in `here https://litebird-sim.readthedocs.io/en/master/hwp.html`_


HWP Harmonics Formalism
----------------

This module implements rotation frequency harmonics explicit HWP non-idealities using Mueller’s formalism (see Jones formalism below)
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
^^^^^^^^^^^^^^^^^^^^^


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



Band Integration
^^^^^^^^^^^^^^^^^^^^^

Currently, the integration in band is available only for the Jones formalism. It works by the trapezoidal rule:

.. math::
    \sum_i \sum_f \big[d_f(i) + d_{f+1}(i)\big] \frac{\nu_{f+1} - \nu_{f}}{2}

where :math:`d_{f}(i)` is the tod computed for a sample i, and at frequency at index f, and :math:`\nu_{f}` if the frequency in Hz at index f.

The steps to perform band integration are:
    1 - give the path to a csv/txt file containing the jones parameters for **all frequencies of the instrument** to the attribute **jones_per_freq_csv_path** when instantiating **NonIdealHWP** object.

    2 - set integrate_in_band to True when calling **scan_map()** (or **hwp_harmonics.fill_tod()** if working at a lower-level).

The csv/txt file should have the following columns:

..  csv-table::
    :header: "freq", "Jxx_0f", "Phxx_0f","Jxy_0f", "Phxy_0f","Jyx_0f", "Phyx_0f","Jyy_0f", "Phyy_0f","Jxx_2f", "Phxx_2f","Jxy_2f", "Phxy_2f","Jyx_2f", "Phyx_2f","Jyy_2f", "Phyy_2f",

    70, 0.93, 40, 0.001, 100, 0.003, 32, 0.95, 170, 0.012, 85, 0.0008, 95, 0.0025, 28, 0.011, 92
    71, 0.92, 42, 0.0012, 102, 0.0032, 34, 0.94, 168, 0.013, 87, 0.0009, 97, 0.0027, 30, 0.012, 94
    72, 0.94, 38, 0.0009, 98, 0.0028, 30, 0.96, 172, 0.011, 83, 0.0007, 93, 0.0023, 26, 0.010, 90


Examples
-------------

Examples can be seen in the notebooks:
    1 - `Performing a simulation with explicit HWP harmonics expansion <https://github.com/litebird/litebird_sim/blob/master/notebooks/litebird_sim_hwp_harmonics_example.ipynb>`_
    2 - `Performing band integration with a Jones Matrix <https://github.com/litebird/litebird_sim/blob/master/notebooks/litebird_sim_hwp_band_integration.ipynb>`_



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

HWP_harmonics
~~~~~~~

.. automodule:: litebird_sim.hwp_harmonics.hwp_harmonics
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource

.. automodule:: litebird_sim.hwp_harmonics.mueller_methos
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource

.. automodule:: litebird_sim.hwp_harmonics.jones_methods
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource

.. automodule:: litebird_sim.hwp_harmonics.common
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource

.. automodule:: litebird_sim.hwp_diff_emiss
    :members:
    :show-inheritance:
    :private-members:
    :member-order: bysource
