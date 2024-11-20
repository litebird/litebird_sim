Gain drift injection
====================

Gain drift is the systematic that is multiplicative to time-ordered
data. The LiteBIRD Simulation Framework provides a gain drift
simulation module based on the same module in ``toast3``. Though the
exact nature of the gain drift depends on the specifics of the
electronics, the gain drift module provides the functions to simulate
two kinds of gain drifts:

1. Linear gain drifts;

2. Thermal gain drifts.

For any gain drift, one can use either the method of
:class:`.Simulation` class :meth:`.Simulation.apply_gaindrift()`, or
any of the low-level functions:
:func:`.apply_gaindrift_to_observations()`,
:func:`.apply_gaindrift_to_tod()`,
:func:`.apply_gaindrift_for_one_detector()`. The following example
shows the typical usage of the method and low-level functions:

.. code-block:: python

    import numpy as np
    import litebird_sim as lbs
    from astropy.time import Time

    start_time = Time("2034-05-02")
    duration_s = 2*24*3600
    sampling_freq_Hz = 1

    # Creating a list of detectors. The detector name is used as one of
    # two seeds to introduce unique and reproducible randomness to the
    # gain drift for each detector.
    dets = [
        lbs.DetectorInfo(
            name="det_A_wafer_1", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_1"
        ),
        lbs.DetectorInfo(
            name="det_B_wafer_1", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_1"
        ),
        lbs.DetectorInfo(
            name="det_C_wafer_2", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_2"
        ),
    ]

    # Defining the gain drift simulation parameters
    drift_params = lbs.GainDriftParams(
        drift_type=lbs.GainDriftType.LINEAR_GAIN,
        sampling_uniform_low=0.2,
        sampling_uniform_high=0.7,
    )

    sim1 = lbs.Simulation(
        base_path="linear_gd_example",
        start_time=start_time,
        duration_s=duration_s,
        random_seed=12345,
    )

    sim1.create_observations(
        detectors=dets,
        split_list_over_processes=False,
        num_of_obs_per_detector=1,
    )

    # Creating fiducial TODs
    sim1.observations[0].gain_2_self = np.ones_like(sim1.observations[0].tod)
    sim1.observations[0].gain_2_obs = np.ones_like(sim1.observations[0].tod)
    sim1.observations[0].gain_2_tod = np.ones_like(sim1.observations[0].tod)
    sim1.observations[0].gain_2_det = np.ones_like(sim1.observations[0].tod)

    # Applying gain drift using the `Simulation` class method
    sim1.apply_gaindrift(
        drift_params=drift_params,
        component="gain_2_self",
    )

    # Applying gain drift on the given TOD component of an `Observation` object
    lbs.apply_gaindrift_to_observations(
        observations=sim1.observations,
        drift_params=drift_params,
        component="gain_2_obs",
    )

    # Applying gain drift on the TOD array. One must pass the name of the
    # associated detectors (or just an array of string objects) to seed
    # the reproducible randomness
    lbs.apply_gaindrift_to_tod(
        tod=sim1.observations[0].gain_2_tod,
        sampling_freq_hz=sampling_freq_Hz,
        det_name=sim1.observations[0].name,
        drift_params=drift_params,
    )

    # Applying gain drift on the TOD arrays of the individual detectors.
    # One must pass the name of the detector (or a string object) to seed
    # the reproducible randomness.
    for idx, tod in enumerate(sim1.observations[0].gain_2_det):
        lbs.apply_gaindrift_for_one_detector(
            det_tod=tod,
            sampling_freq_hz=sampling_freq_Hz,
            det_name=sim1.observations[0].name[idx],
            drift_params=drift_params,
        )

    # The four TODs we obtain this way are equal to each other.

One has to specify the gain drift simulation parameters as an instance
of the :class:`.GainDriftParams` class. The type of the gain drift can
be selected using the enum class :class:`.GainDriftType`. The
:class:`.GainDriftParams` class also offers the facility to specify
the distribution of the slope for the linear gain and the distribution
of the detector mismatch for the thermal gain, which can be specified
with the help of the enum class :class:`.SamplingDist`.

Following is an example of linear gain drift simulation parameters
where the slope of gain for different detectors follows Gaussian
distribution with a mean of 0.8 and a standard deviation of 0.2:

.. code-block:: python

    import litebird_sim as lbs

    drift_params = lbs.GainDriftParams(
        drift_type = lbs.GainDriftType.LINEAR_GAIN,
        sampling_dist = lbs.SamplingDist.GAUSSIAN,
        sampling_gaussian_loc = 0.8,
        sampling_gaussian_scale = 0.2,
    )

The following example shows the thermal gain drift simulation
parameters where the detector mismatch within a detector group has a
uniform distribution varying between the factors 0.2 to 0.8:

.. code-block:: python

   import litebird_sim as lbs

   drift_params = lbs.GainDriftParams(
       drift_type = lbs.GainDriftType.THERMAL_GAIN,
       sampling_dist = lbs.SamplingDist.UNIFORM,
       sampling_uniform_low = 0.2,
       sampling_uniform_high = 0.8,
   )

Refer to the :ref:`gd-api-reference` for the full list of gain drift simulation parameters.

Linear gain drift
-----------------

Linear gain drift is the linearly increasing factor for TODs. The
:mod:`.gaindrifts` module provides methods and functions to simulate
the linear gain drift with the possibility of periodic calibration.
The calibration event resets the gain factor to one periodically after
every calibration period interval. You can specify the calibration
period with the attribute
:attr:`.GainDriftParams.calibration_period_sec`. The following example
shows the time evolution of the linear gain drift factor over four
days with a calibration period of 24 hours:

.. plot:: pyplots/lingain_demo.py
   :include-source:

Note that the figure above shows only the nature of the linear gain
drift factor to be multiplied by the sky TOD.

The module generates different gain slopes for different detectors.
The factor :math:`\sigma_{drift}\times\delta` determines the slope (or
the peak amplitude) of the linear gain, where :math:`\sigma_{drift}`
is a dimensionless parameter specified by
:attr:`.GainDriftParams.sigma_drift` and :math:`\delta` is the random
factor generated uniquely for each detector. The distribution of
:math:`\delta` or, conversely, the distribution of the gain slopes
over all the detectors can be specified with attributes of
:class:`.SamplingDist` enum class and the associated parameters listed
in :class:`.GainDriftParams`.

Thermal gain drift
------------------

The thermal gain drift is modelled as the gain drift due to
:math:`1/f` fluctuation in focalplane temperature. In the first step,
the :math:`1/f` noise timestream is generated from oversampled power
spectral density given by

.. math::
    S(f) = \sigma_{drift}^2\left(\frac{f_{knee}}{f}\right)^{\alpha_{drift}}

The noise timestream is considered to be the same for all the
detectors belonging to a given detector group. One may specify which
detector parameter to use to make a detector group, using the
attribute :attr:`.GainDriftParams.focalplane_group`. Valid values are
`"wafer"`, `"pixtype"`, or `"channel"`. For example, if
:attr:`.GainDriftParams.focalplane_group = "wafer"`, all the detectors
with the same wafer name will be considered in one group and have the
same noise timestream.

Once the noise timestreams are obtained for all the groups in the
focal plane, the code inserts a mismatch for the detectors within a
group by a random factor and the detector mismatch factor. We can
express the noise timestream with detector mismatch using the
following expression:

.. math::
    t^{(mis)}_{stream} = (1 + \delta\times\alpha_{mis})t_{stream}

where :math:`\alpha_{mis}` is the detector mismatch factor specified
using the attribute :attr:`.GainDriftParams.detector_mismatch` and
:math:`\delta` is the random factor generated uniquely for each
detector. The distribution of :math:`\delta` or, conversely, the
distribution of noise timestream mismatch can be specified with
attributes of :class:`.SamplingDist` enum class and the associated
parameters listed in :class:`.GainDriftParams`.

The mismatched timestream is then scaled and passed through a
responsivity function to obtain the thermal gain factor
(:math:`\sigma`):

.. math::
    \sigma = \text{responsivity_function}\left(1+\frac{ t^{(mis)}_{stream} \times \delta_T }{T_{bath}}\right)

where :math:`\delta_T` is the amplitude of the thermal gain
fluctuation in Kelvin unit, specified with attribute
:attr:`.GainDriftParams.thermal_fluctuation_amplitude_K`, and
:math:`T_{bath}` is the temperature of the focalplane in Kelvin unit
specified with the attribute
:attr:`.GainDriftParams.focalplane_Tbath_K`.

The following example shows the comparison of thermal gain drift
factor with or without detector mismatch over 100 seconds.

.. plot:: pyplots/thermalgain_demo.py
   :include-source:

In the plots above, when there is no detector mismatch,
``det_A_wafer_1`` and ``det_B_wafer_1`` have the same gain drift
factor as they belong to the same focal plane group (grouped by wafer
name). However, when the detector mismatch is enabled, the two gain
drift factors have the same shape due to the same underlying noise
timestream but differ slightly in amplitude due to an additional
random mismatch factor.

.. _gd-api-reference:

API reference
-------------

.. automodule:: litebird_sim.gaindrifts
    :members:
    :undoc-members:
    :show-inheritance:
