Ideal Half Wave Plate
=====================

The rotation of the polarization angle induced by a HWP can be
included in the returned pointing information by passing an instance
of a descendant of the class :class:`.HWP` to the method
:meth:`.Simulation.set_hwp`. Here is an example::

    import litebird_sim as lbs

    sim = lbs.Simulation(
        start_time=0,
        duration_s=100.0,
        random_seed=12345,
    )

    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.785_398_163_397_448_3,
            precession_rate_hz=8.664_850_513_998_931e-05,
            spin_rate_hz=0.000_833_333_333_333_333_4,
            start_time=sim.start_time,
        ),
        delta_time_s=60,
    )

    sim.set_instrument(
        instrument = lbs.InstrumentInfo(
            boresight_rotangle_rad=0.0,
            spin_boresight_angle_rad=0.872_664_625_997_164_8,
            spin_rotangle_rad=3.141_592_653_589_793,
        )
    )

    sim.set_hwp(
        lbs.IdealHWP(ang_speed_radpsec=4.084_070_449_666_731),
    )

    det = lbs.DetectorInfo(
        name="Boresight_detector",
        sampling_rate_hz=1.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )
    obs, = sim.create_observations(detectors=[det])

    sim.prepare_pointings()

    pointing, hwp_angle = sim.observations[0].get_pointings()

This example uses the :class:`.IdealHWP`, which represents an ideal
spinning HWP. The method :func:`.get_pointings` returns both pointing
and hwp angle on the fly. As shown before a similar syntax allow to
precompute the pointing and the hwp angle::

    sim.prepare_pointings()
    sim.precompute_pointings()

    sim.observations[0].pointing_matrix.shape
    sim.observations[0].hwp_angle.shape

This fills the fields `pointing_matrix` and `hwp_angle` for all the
observations in the current simulation. 


API reference
-------------

.. automodule:: litebird_sim.hwp
    :members:
    :undoc-members:
    :show-inheritance:
