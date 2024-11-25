.. _pointing_sys:

Pointing systematics
====================
The pointing systematics causes some disturbances in the pointing direction of the detectors. Due to the systematics, we will have a signal from the sky where we are not pointing, and the obtained TOD will be perturbed by the systematics.

These TODs with systematics will be used for map-making, though the pointing matrix in the map-maker will be created with the expected pointings, i.e., pointings without systematics.

In order to simulate the pointing systematics, we multiply the systematic quaternions, which describe perturbations, by the expected quaternions.
For example, to add a 5-degree offset around the :math:`x`-axis to the positional quaternion :math:`q_z=(0,0,1,0)` indicating the :math:`z`-axis, the following operation is available:

.. code-block:: python

    import litebird_sim as lbs
    import numpy as np

    def print_quaternion(q):
        print("{:.3f} {:.3f} {:.3f} {:.3f}".format(*q))

    det = lbs.DetectorInfo(
        name="000_000_003_QA_040_T",
        sampling_rate_hz=1.0,
    )

    # Positional quaternion indicating z-axis
    q_z = lbs.RotQuaternion(
        quats=np.array([0.0, 0.0, 1.0, 0.0]),
    )

    # Systematic quaternion indicating 5-degree rotation around x-axis
    q_sys = lbs.RotQuaternion(
        quats=np.array(
            lbs.quat_rotation_x(np.deg2rad(5.0))
        ),
    )

    # The systematic quaternions are multiplied by the expected quaternions in-place.
    lbs.left_multiply_syst_quats(
        q_z,
        q_sys,
        det,
        start_time=0.0,
        sampling_rate_hz=det.sampling_rate_hz,
    )

    print("Rotation by 5 deg. around x-axis:")
    print_quaternion(q_z.quats[0])

.. testoutput::

    Rotation by 5 deg. around x-axis:
    -0.000 -0.044 0.999 -0.000

The function :func:`.left_multiply_syst_quats` multiplies the systematic quaternions by the expected quaternions in-place.

By using this function, the :class:`.PointingSys` class is implemented to inject the systematic quaternions into the quaternions in a specific coordinate.

:class:`.PointingSys` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`.PointingSys` class is used to inject the systematic quaternions into the quaternions in a specific coordinate.

**Note that every coordinate is defined by a left-handed coordinate system, and projected pointings by HEALPix are defined as a view from the spacecraft into the sky.**

The supported coordinates are:

* :class:`.FocalplaneCoord`: The coordinate of the focal plane with the :math:`z`-axis along the boresight.
    * :meth:`.FocalplaneCoord.add_offset`: Add the offset to the detectors in the focal plane.
    * :meth:`.FocalplaneCoord.add_disturb`: Add the time-dependent disturbance to the detectors in the focal plane.
* :class:`.SpacecraftCoord`: The coordinate of the spacecraft with the :math:`z`-axis along its spin axis.
    * :meth:`.SpacecraftCoord.add_offset`: Add the offset to the entire spacecraft.
    * :meth:`.SpacecraftCoord.add_disturb`: Add the time-dependent disturbance to the entire spacecraft.
* :class:`.HWPCoord`: Same coordinates as the focal plane but treats the pointing systematics due to the HWP rotation.
    * :meth:`.HWPCoord.add_hwp_rot_disturb`: Add the rotational disturbance synchronized with the HWP rotation.

These classes are accessible through the :attr:`.PointingSys.focalplane`, :attr:`.PointingSys.spacecraft`, and :attr:`.PointingSys.hwp` attributes.

How to inject the pointing systematics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is possible to operate :class:`.FocalplaneCoord`, :class:`.SpacecraftCoord`, and :class:`.HWPCoord` directly, but it is recommended to use the :class:`.PointingSys` class to inject the pointing systematics. The following is an example of injecting vibrations into the focal plane.

.. code-block:: python

    import numpy as np
    import litebird_sim as lbs
    from pathlib import Path
    import tomlkit
    import healpy as hp

    def get_hitmap(nside, pointings):
        npix = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside, pointings[:, :, 0], pointings[:, :, 1])
        hitmap, _ = np.histogram(ipix, bins=np.arange(npix + 1))
        return hitmap

    start_time = 0.0
    sampling_hz = 1.0
    telescope = "LFT"
    dets = []

    # Load the mock focal plane configuration
    path_of_toml = (
        Path(lbs.__file__).resolve().parent.parent
        / "test"
        / "pointing_sys_reference"
        / "mock_focalplane.toml"
    )
    with open(path_of_toml, "r", encoding="utf-8") as toml_file:
        toml_data = tomlkit.parse(toml_file.read())
        for i in range(len(toml_data[telescope])):
            dets.append(lbs.DetectorInfo.from_dict(toml_data[telescope][f"det_{i:03}"]))

    # Create a simulation object
    sim = lbs.Simulation(
        base_path="pntsys_example",
        start_time=start_time,
        duration_s=1000.0,
        random_seed=12345,
    )

    # Set the scanning strategy
    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            spin_rate_hz=0.05 / 60.0,
            precession_rate_hz=1.0 / (3.2 * 60 * 60),
        ),
        delta_time_s=1.0 / sampling_hz,
    )

    # Set the instrument
    sim.set_instrument(
        lbs.InstrumentInfo(
            name="mock_LiteBIRD",
            spin_boresight_angle_rad=np.deg2rad(50.0),
        ),
    )

    # Set the HWP
    sim.set_hwp(lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60))

    # Create the observations
    (obs,) = sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
    )

    # Initialize the pointing systematics object
    pntsys = lbs.PointingSys(sim, obs, dets)
    nquats = obs.n_samples + 1
    axes = ["x", "y"]
    noise_sigma_deg = 1.0

    for ax in axes:
        # Prepare the noise to add vibration to the focal plane
        noise_rad_array = np.zeros(nquats)
        lbs.add_white_noise(
            noise_rad_array, sigma=np.deg2rad(noise_sigma_deg), random=sim.random
        )
        # Add the vibration to the pointings
        pntsys.focalplane.add_disturb(noise_rad_array, ax)

    lbs.prepare_pointings(
        obs,
        sim.instrument,
        sim.spin2ecliptic_quats,
        sim.hwp,
    )

    pointings, hwp_angle = obs.get_pointings("all")
    nside = 64
    hitmap = get_hitmap(nside, pointings)
    hp.mollview(hitmap, title="Hit-map with focal plane vibration")

.. image:: images/hitmap_with_vibration.png

In this code snippet, the pointing systematics are injected into the focal plane by adding white noise with a standard deviation of 1 degree to the focal plane. The hit-map is created with the injected pointing systematics.

Tips for MPI distribution
~~~~~~~~~~~~~~~~~~~~~~~~~
In the case we consider time-dependent pointing systematics, we may have to increase the number of quaternions by changing ``delta_time_s`` in the :meth:`.Simulation.set_scanning_strategy` method. For example, it is changed by considering the nominal sampling rate of LiteBIRD, i.e., 19 Hz:

.. code-block:: python

    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            spin_rate_hz=0.05 / 60.0,
            precession_rate_hz=1.0 / (3.2 * 60 * 60),
        ),
        delta_time_s=1.0 / 19.0,
    )

However, after this execution, all of :attr:`.Simulation.spin2ecliptic_quats` from start to end in the observation are calculated and stored in memory. And this is not distributed by MPI, so that even when we use MPI, every process will calculate :attr:`.Simulation.spin2ecliptic_quats` from start to end of the observation. This causes a memory issue.

To avoid this issue, a simple trick to distribute the quaternion calculations is:

.. code-block:: python

    # After preparing:
    # `Simulation` with MPI_COMM_WORLD,
    # list of `DetectorInfo`,
    # `Imo` object...

    # Create observation
    (obs,) = sim.create_observations(
        detectors=dets,
        n_blocks_det=1,
        n_blocks_time=size,
        split_list_over_processes=False
    )

    pntsys = lbs.PointingSys(sim, obs, dets)

    # Define scanning strategy parameters from IMO
    scanning_strategy = lbs.SpinningScanningStrategy.from_imo(
        url= f"/releases/vPTEP/satellite/scanning_parameters/",
        imo=imo,
    )

    # Calculate quaternions for each process
    sim.spin2ecliptic_quats = scanning_strategy.generate_spin2ecl_quaternions(
        start_time=obs.start_time,
        time_span_s=obs.n_samples/obs.sampling_rate_hz,
        delta_time_s=1.0/19.0,
    )

    # After injecting the pointing systematics...

    lbs.prepare_pointings(
        obs,
        sim.instrument,
        sim.spin2ecliptic_quats,
        sim.hwp,
    )

Especially, this trick is needed when the time-dependent pointing systematics' spectrum has higher frequency than the quaternion's sampling rate given by ``delta_time_s``.

API Reference
-------------
.. automodule:: litebird_sim.pointing_sys
    :members:
    :undoc-members:
    :show-inheritance:
