import numpy as np
import litebird_sim as lbs
from astropy.time import Time
from litebird_sim.pointings_in_obs import _get_hwp_angle


def test_add_2f():
    # Test function to check consistency of wrappers and low level functions
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
            imo=imo,
        )
        det.sampling_rate_hz = 1
        dets.append(det)

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
    )

    sim.set_scanning_strategy(
        imo_url=f"/releases/{imo_version}/satellite/scanning_parameters/"
    )

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
            angle_det_rad=_get_hwp_angle(sim.observations[0], sim.hwp)
            - sim.observations[0].pol_angle_rad[idx],
            amplitude_k=sim.observations[0].amplitude_2f_k[idx],
        )

    # Check if the three tods are equal after adding 2f signal from HWP differential emission

    np.testing.assert_array_equal(
        sim.observations[0].tod2f_2_self, sim.observations[0].tod2f_2_obs
    )
    np.testing.assert_array_equal(
        sim.observations[0].tod2f_2_self, sim.observations[0].tod2f_2_det
    )
