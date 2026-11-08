import numpy as np
from astropy.time import Time

import litebird_sim as lbs
from litebird_sim.observation_utilities import ObservationPointings


def test_add_2f():
    # Test function to check consistency of wrappers and low level functions:
    #
    # 1. Simulation.add_2f()
    # 2. lbs.hwp_diff_emiss.add_2f_to_observations()
    # 3. lbs.hwp_diff_emiss.add_2f_for_one_detector()
    #
    # Each function builds on top of the following, and they must all agree

    telescope = "LFT"
    channel = "L4-140"
    detlist = ["000_001_017_QB_140_T", "000_001_017_QB_140_B"]
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
            f"/releases/vPTEP/satellite/{telescope}/instrument_info",
        )
    )

    dets = []
    for n_det in detlist:
        det = lbs.DetectorInfo.from_imo(
            url=f"/releases/vPTEP/satellite/{telescope}/{channel}/{n_det}/detector_info",
            imo=imo,
        )
        det.sampling_rate_hz = 1
        dets.append(det)

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
    )

    sim.set_scanning_strategy(imo_url="/releases/vPTEP/satellite/scanning_parameters/")

    sim.set_hwp(
        lbs.IdealHWP(
            sim.instrument.hwp_rpm * 2 * np.pi / 60,
        ),
    )

    sim.prepare_pointings()
    sim.precompute_pointings()

    # Creating fiducial TODs
    sim.observations[0].tod2f_2_self = np.zeros_like(sim.observations[0].tod)
    sim.observations[0].tod2f_2_obs = np.zeros_like(sim.observations[0].tod)
    sim.observations[0].tod2f_2_det = np.zeros_like(sim.observations[0].tod)

    # Define differential emission parameters for the detectors.
    sim.observations[0].amplitude_2f_k = np.array([0.1, 0.1])

    # Adding 2f signal from HWP differential emission using the `Simulation` class method
    sim.add_2f(
        component="tod2f_2_self",
    )

    # Adding 2f signal from HWP differential emission to the given TOD component of an `Observation` object
    lbs.hwp_diff_emiss.add_2f_to_observations(
        observations=sim.observations,
        hwp=sim.hwp,
        component="tod2f_2_obs",
    )

    # Adding 2f signal from HWP differential emission to  the TOD arrays of the individual detectors.
    for idx, tod in enumerate(sim.observations[0].tod2f_2_det):
        lbs.hwp_diff_emiss.add_2f_for_one_detector(
            tod_det=tod,
            angle_det_rad=ObservationPointings(
                sim.observations[0], hwp=sim.hwp
            ).hwp_angle,
            offset_angle_rad=sim.observations[0].pol_angle_rad[idx],
            amplitude_k=sim.observations[0].amplitude_2f_k[idx],
        )

    # Check if the three tods are equal after adding 2f signal from HWP differential emission

    np.testing.assert_array_equal(
        sim.observations[0].tod2f_2_self, sim.observations[0].tod2f_2_obs
    )
    np.testing.assert_array_equal(
        sim.observations[0].tod2f_2_self, sim.observations[0].tod2f_2_det
    )


def test_add_2f_amplitude_per_observation():
    # The amplitude stored in each observation must be used for that
    # observation only, not reused for the following ones
    def make_obs(amplitude):
        sim = lbs.Simulation(start_time=0.0, duration_s=60.0, random_seed=1)
        sim.set_scanning_strategy(lbs.SpinningScanningStrategy(0.7, 1e-4, 1e-2))
        sim.set_instrument(lbs.InstrumentInfo(name="i", spin_boresight_angle_rad=0.8))
        sim.set_hwp(lbs.IdealHWP(ang_speed_radpsec=1.0))
        sim.create_observations(
            detectors=[lbs.DetectorInfo(name="d", sampling_rate_hz=10.0)]
        )
        sim.prepare_pointings()
        obs = sim.observations[0]
        obs.amplitude_2f_k = np.array([amplitude])
        return obs, sim.hwp

    obs1, hwp = make_obs(1.0)
    obs2, _ = make_obs(5.0)

    lbs.hwp_diff_emiss.add_2f_to_observations([obs1, obs2], hwp=hwp)

    for obs in (obs1, obs2):
        expected = np.zeros_like(obs.tod)
        lbs.hwp_diff_emiss.add_2f(
            tod=expected,
            hwp_angle=ObservationPointings(obs, hwp=hwp).hwp_angle,
            pol_angle_rad=obs.pol_angle_rad,
            amplitude_2f_k=obs.amplitude_2f_k,
        )
        np.testing.assert_allclose(obs.tod, expected)
