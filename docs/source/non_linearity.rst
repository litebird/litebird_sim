.. _non_linearity:

Non-linearity injection
=======================

Non-linearity is the effect of a non-ideal TES detector response,
where the responsivity :math:`S` is not constant as the optical power
varies. The LiteBIRD Simulation Framework provides a non-linearity
simulation module to simulate the effect of non-linearity on TODs.

The framework provides the simplest case, which is quadratic
non-linearity. This case is described in `Micheli+2024
<https://arxiv.org/pdf/2407.15294>`_, where the effect of
non-linearity is propagated to the estimation of the tensor-to-scalar
ratio.

Considering a first order correction of the usual linear gain, a TOD
:math:`d(t)` is modified according to:

.. math::
    d(t) = [1+g_1 d(t)] d(t)

where :math:`g_1` is the detector non-linearity factor in units of
:math:`K^{-1}`.

Examples
-------------
To simulate a quadratic non-linearity, one can use the method of
:class:`.Simulation` class
:meth:`.Simulation.apply_quadratic_nonlin()`, or any of the low-level
functions: :func:`.apply_quadratic_nonlin_to_observations()`,
:func:`.apply_quadratic_nonlin_for_one_detector()`. 

The examples below skip the simulation and observation creation for brevity.
If needed, the implementation for those parts is explained in other sections of the docs.

.. code-block:: python

   (... importing modules, creating simulation, setting scanning strategy, instrument, etc...)

   # creating the detectors (two mock detectors here)
   dets = [
        lbs.DetectorInfo(name="det_A", sampling_rate_hz=sampling_hz),
        lbs.DetectorInfo(name="det_B", sampling_rate_hz=sampling_hz),
    ]

   (obs,) = sim.create_observations(
      detectors=dets,
   )

   sim.prepare_pointings()
   
   (... fill your TOD scanning a map, or add dipole, noise, etc...)
   
   # Define nonlinearity amplitude distribution for the detectors, in 1/K units. 
   # The value is randomized for each detector, using either a user-provided seed or a list of
   # pre-initialized RNGs.
   
   nl_params = lbs.NonLinParams(sampling_gaussian_loc=-1.0, sampling_gaussian_scale=0.01)

   # Applying nonlinearity using the `Simulation` class method
   # By default, it modifies the ``Observation.tod``. If you want to apply it to some
   # other field of the :class:`.Observation` class, use `component`

   sim.apply_quadratic_nonlin(nl_params = nl_params)


.. _nl-api-reference:

API reference
-------------

.. automodule:: litebird_sim.non_linearity
    :members:
    :undoc-members:
    :show-inheritance:
