Non-linearity injection
=======================

Non-linearity is the effect of a non-ideal TES detectors' response. This means that the responsivity :math:`S` is not constant as the optical power varies. 
The LiteBIRD Simulation Framework provides a non-linearity simulation module to simulate the effect of non-linearity on TODs.

The framework provides the simplest case, which is a quadratic non-linearity. 
This case is described in `Micheli+2024 <https://arxiv.org/pdf/2407.15294>`_, where the effect of non-linearity is propagated to the estimation of the tensor-to-scalar ratio.
Considering a first order correction of the usual linear gain, a TOD :math:`d(t)` is modified according to:

.. math::
    d(t) = [1+g_1 d(t)] d(t)

where :math:`g_1` is the detector non-linearity factor in units of :math:`K^{-1}`.

To simulate a quadratic non-linearity, one can use the method of :class:`.Simulation` class :meth:`.Simulation.apply_quadratic_nonlin()`, 
or any of the low level functions: :func:`.apply_quadratic_nonlin_to_observations()`, :func:`.apply_quadratic_nonlin_for_one_detector()`. 
The following example shows the typical usage of the method and low level functions:


.. code-block:: python

    import numpy as np
    import litebird_sim as lbs
    from astropy.time import Time

    start_time = Time("2025-02-02T00:00:00")
    mission_time_days = 1
    sampling_hz = 19

    dets = [
        lbs.DetectorInfo(
            name="det_A", sampling_hz=sampling_hz
        ),
        lbs.DetectorInfo(
            name="det_B", sampling_hz=sampling_hz
        ),
    ]

    sim = lbs.Simulation(
        base_path="nonlin_example",
        start_time=start_time,
        duration_s=mission_time_days * 24 * 3600.0,
        random_seed=12345,
    )

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
    )
    
    # Creating fiducial TODs
    sim.observations[0].nl_2_self = np.ones_like(sim.observations[0].tod)
    sim.observations[0].nl_2_obs = np.ones_like(sim.observations[0].tod)
    sim.observations[0].nl_2_det = np.ones_like(sim.observations[0].tod)

One has to specify the :math:`g_1` parameter using the ``g_one_over_k`` argument as in the following example:

.. code-block:: python

    # Define non-linear parameters for the detectors. We choose the same value for both detectors in this example, but it is not necessary.
    sim.observations[0].g_one_over_k = np.ones(len(dets)) * 1e-3
    
    # Applying non-linearity using the `Simulation` class method
    sim.apply_quadratic_nonlin(component = "nl_2_self",)

    # Applying non-linearity on the given TOD component of an `Observation` object
    lbs.apply_quadratic_nonlin_to_observations(
        observations=sim.observations,
        component="nl_2_obs",
    )

    # Applying non-linearity on the TOD arrays of the individual detectors.
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        lbs.apply_quadratic_nonlin_for_one_detector(
            tod_det=tod,
            g_one_over_k=sim.observations[0].g_one_over_k[idx],
        )


Refer to the :ref:`gd-api-reference` for the full list of non-linearity simulation parameters.

.. _gd-api-reference:

API reference
-------------

.. automodule:: litebird_sim.non_linearity
    :members:
    :undoc-members:
    :show-inheritance:
