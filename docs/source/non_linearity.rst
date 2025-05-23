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

To simulate a quadratic non-linearity, one can use the method of
:class:`.Simulation` class
:meth:`.Simulation.apply_quadratic_nonlin()`, or any of the low-level
functions: :func:`.apply_quadratic_nonlin_to_observations()`,
:func:`.apply_quadratic_nonlin_for_one_detector()`. The following
example shows the typical usage of the method and low-level functions:

.. code-block:: python

    import numpy as np
    import litebird_sim as lbs
    from astropy.time import Time

    start_time = Time("2025-02-02T00:00:00")
    mission_time_days = 1
    sampling_hz = 19

    dets = [
        lbs.DetectorInfo(
            name="det_A", sampling_rate_hz=sampling_hz
        ),
        lbs.DetectorInfo(
            name="det_B", sampling_rate_hz=sampling_hz
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

If the non-linearity parameter is not read from the IMo, one has to
specify :math:`g_1` using the ``g_one_over_k`` argument as in the
following example:

.. code-block:: python

    # Define non-linear parameters for the detectors. We choose the same
    # value for both detectors in this example, but it is not necessary.
    sim.observations[0].g_one_over_k = np.ones(len(dets)) * 1e-3

    # Applying non-linearity using the `Simulation` class method
    sim.apply_quadratic_nonlin(component = "nl_2_self",)

    # Applying non-linearity on the given TOD component of an `Observation`
    # object
    lbs.non_linearity.apply_quadratic_nonlin_to_observations(
        observations=sim.observations,
        component="nl_2_obs",
    )

    # Applying non-linearity on the TOD arrays of the individual detectors.
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        lbs.non_linearity.apply_quadratic_nonlin_for_one_detector(
            tod_det=tod,
            g_one_over_k=sim.observations[0].g_one_over_k[idx],
        )

In particular, the effect of detector non-linearity must be included
to assess its impact when coupled with other systematic effects. As
described in `Micheli+2024 <https://arxiv.org/pdf/2407.15294>`_, a
typical case is the coupling with HWP synchronous signal (HWPSS)
appearing at twice the rotation frequency of the HWP. This kind of
signal can be produced by non-idealities of the HWP, such as its
differential transmission and emission. 
Note that it is important to include in the TODs the contribution
of the CMB monopole and the orbital dipole _before_ applying non-linearity,
to properly account for the total signal hitting the detectors. 

In that case, the usual TOD :math:`d(t)` will contain an additional
term, and can be written as:

.. math::
    d(t) = d(t) = I + \mathrm{Re}[\epsilon_{\mathrm{pol}}e^{4i\chi}(Q+iU)]+A_2 \cos(2 \omega_{HWP} t) + N

where :math:`A_2` is the amplitude of the HWPSS and
:math:`\omega_{HWP}` is the rotation speed of the HWP. In presence of
detector non-linearity, the 2f signal is up-modulated to 4f, affecting
the science band.

The framework provides an independent module to introduce this signal
in the simulation, adding it to the TODs. To simulate the 2f signal
from a rotating, non-ideal HWP, one can use the method of
:class:`.Simulation` class :meth:`.Simulation.add_2f()`, or any of the
low-level functions: :func:`.add_2f_to_observations()`,
:func:`.add_2f_for_one_detector()`.

If the 2f amplitude is not read from the IMo, one has to specify
:math:`A_2` using the ``amplitude_2f_k`` argument. See the following example:

.. code-block:: python

    import numpy as np
    import litebird_sim as lbs
    from astropy.time import Time
    from litebird_sim.pointings import get_hwp_angle


    telescope = "LFT"
    channel = "L4-140"
    detlist = ["000_001_017_QB_140_T", "000_001_017_QB_140_B"]
    imo_version = "vPTEP"
    start_time = Time("2025-02-02T00:00:00")
    mission_time_days = 1

    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)


    sim = lbs.Simulation(
        base_path="nonlin_example",
        start_time=start_time,
        imo=imo,
        duration_s=mission_time_days * 24 * 3600.0,
        random_seed=12345,
    )

    # Load the definition of the instrument
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            imo,
            f"/releases/{imo_version}/satellite/{telescope}/instrument_info",
        )
    )

    dets = []
    for n_det in detlist:
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/{imo_version}/satellite/{telescope}/{channel}/{n_det}/detector_info",
            imo=imo,)
        det.sampling_rate_hz = 1
        dets.append(det)

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
    )

    sim.set_scanning_strategy(imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/")

    sim.set_hwp(
        lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60,),
    )

    sim.prepare_pointings()
    sim.precompute_pointings()

    # Creating fiducial TODs
    sim.observations[0].tod_2f = np.zeros_like(sim.observations[0].tod)

    # Define differential emission parameters for the detectors.
    sim.observations[0].amplitude_2f_k = np.array([0.1, 0.1])

    # Adding 2f signal from HWP differential emission using the `Simulation` class method
    sim.add_2f(component="tod_2f")


Refer to the :ref:`nl-api-reference` for the full list of non-linearity simulation parameters.

.. _nl-api-reference:

API reference
-------------

.. automodule:: litebird_sim.non_linearity
    :members:
    :undoc-members:
    :show-inheritance:
