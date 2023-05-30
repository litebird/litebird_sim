import numpy as np
import hashlib
import litebird_sim as lbs
from astropy.time import Time


def _hash_function(
    input_str: str,
    user_seed: int = 12345,
) -> int:
    """This functions generates a unique and reproducible hash for a given pair of
    `input_str` and `user_seed`. A copy of the function with the same name
    defined in `litebird_sim/gaindrifts.py`.
    """

    bytesobj = (str(input_str) + str(user_seed)).encode("utf-8")

    hashobj = hashlib.md5()
    hashobj.update(bytesobj)
    digest = hashobj.digest()

    return int.from_bytes(bytes=digest, byteorder="little")


class Test_wrappers_gain_drift:
    """Class to group tests for checking the consistency of wrappers and
    low level functions"""

    start_time = Time("2034-05-02")
    duration_s = 2 * 24 * 3600
    sampling_freq_Hz = 1

    # Three detectors with two different focalplane attributes
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

    drift_params = lbs.GainDriftParams(
        drift_type=lbs.GainDriftType.LINEAR_GAIN,
        sampling_freq_Hz=sampling_freq_Hz,
        sampling_uniform_low=0.2,
        sampling_uniform_high=0.7,
    )

    def test_wrappers_linear_gain_drift(self, tmp_path):
        """This function test if the high level wrappers produce same
        results as the low level function for linear gain drift.
        """

        sim1 = lbs.Simulation(
            base_path=tmp_path / "gd_wrapper_test",
            start_time=self.start_time,
            duration_s=self.duration_s,
        )

        sim1.create_observations(
            detectors=self.dets,
            split_list_over_processes=False,
            num_of_obs_per_detector=1,
        )

        sim1.observations[0].gain_2_self = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_2_obs = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_2_tod = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_2_det = np.ones_like(sim1.observations[0].tod)

        # Applying gain drift using four different functions
        sim1.apply_gaindrift(
            drift_params=self.drift_params,
            component="gain_2_self",
        )

        lbs.apply_gaindrift_to_observations(
            obs=sim1.observations,
            drift_params=self.drift_params,
            component="gain_2_obs",
        )

        lbs.apply_gaindrift_to_tod(
            tod=sim1.observations[0].gain_2_tod,
            det_name=sim1.observations[0].name,
            drift_params=self.drift_params,
        )

        for idx, tod in enumerate(sim1.observations[0].gain_2_det):
            lbs.apply_gaindrift_for_one_detector(
                det_tod=tod,
                det_name=sim1.observations[0].name[idx],
                drift_params=self.drift_params,
            )

        # Testing if the four gain drift tods are same
        np.testing.assert_array_equal(
            sim1.observations[0].gain_2_self, sim1.observations[0].gain_2_obs
        )
        np.testing.assert_array_equal(
            sim1.observations[0].gain_2_self, sim1.observations[0].gain_2_tod
        )
        np.testing.assert_array_equal(
            sim1.observations[0].gain_2_self, sim1.observations[0].gain_2_det
        )

        sim1.flush()

    def test_wrapper_thermal_gain_drift(self, tmp_path):
        """This function test if the high level wrappers produce same
        results as the low level function for the thermal gain drift.
        """

        sim1 = lbs.Simulation(
            base_path=tmp_path / "gd_wrapper_test",
            start_time=self.start_time,
            duration_s=self.duration_s,
        )

        sim1.create_observations(
            detectors=self.dets,
            split_list_over_processes=False,
            num_of_obs_per_detector=1,
        )

        sim1.observations[0].gain_2_self = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_2_obs = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_2_tod = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_2_det = np.ones_like(sim1.observations[0].tod)

        self.drift_params.drift_type = lbs.GainDriftType.THERMAL_GAIN

        # Applying gain drift using four different functions
        sim1.apply_gaindrift(
            drift_params=self.drift_params,
            component="gain_2_self",
        )

        lbs.apply_gaindrift_to_observations(
            obs=sim1.observations,
            drift_params=self.drift_params,
            component="gain_2_obs",
        )

        lbs.apply_gaindrift_to_tod(
            tod=sim1.observations[0].gain_2_tod,
            det_name=sim1.observations[0].name,
            drift_params=self.drift_params,
            focalplane_attr=getattr(
                sim1.observations[0], self.drift_params.focalplane_group
            ),
        )

        for idx, tod in enumerate(sim1.observations[0].gain_2_det):
            lbs.apply_gaindrift_for_one_detector(
                det_tod=tod,
                det_name=sim1.observations[0].name[idx],
                drift_params=self.drift_params,
                focalplane_attr=getattr(
                    sim1.observations[0], self.drift_params.focalplane_group
                )[idx],
            )

        # Testing if the four gain drift tods are same
        np.testing.assert_array_equal(
            sim1.observations[0].gain_2_self, sim1.observations[0].gain_2_obs
        )
        np.testing.assert_array_equal(
            sim1.observations[0].gain_2_self, sim1.observations[0].gain_2_tod
        )
        np.testing.assert_array_equal(
            sim1.observations[0].gain_2_self, sim1.observations[0].gain_2_det
        )

        sim1.flush()


def test_linear_gain_drift(tmp_path):
    """This function test if the linear gain drifts are applied correctly."""

    start_time = Time("2034-05-02")
    duration_s = 2 * 24 * 3600
    sampling_freq_Hz = 3

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

    drift_params = lbs.GainDriftParams(
        drift_type=lbs.GainDriftType.LINEAR_GAIN,
        sampling_freq_Hz=sampling_freq_Hz,
        sampling_dist=lbs.SamplingDist.GAUSSIAN,
        sampling_gaussian_loc=0.5,
        sampling_gaussian_scale=0.2,
    )

    sim1 = lbs.Simulation(
        base_path=tmp_path / "gd_lineargain_test",
        start_time=start_time,
        duration_s=duration_s,
    )

    sim1.create_observations(
        detectors=dets,
        split_list_over_processes=False,
        num_of_obs_per_detector=1,
    )

    # gain_wrapper stores the tod applied with the wrapper function and is
    # tested against gain_native where the gain is applied right within
    # this function
    sim1.observations[0].gain_wrapper = np.ones_like(sim1.observations[0].tod)
    sim1.observations[0].gain_native = np.ones_like(sim1.observations[0].tod)

    sim1.apply_gaindrift(
        drift_params=drift_params,
        component="gain_wrapper",
        user_seed=987654321,
    )

    tod_size = len(sim1.observations[0].gain_native[0])

    for idx, tod in enumerate(sim1.observations[0].gain_native):
        rng = np.random.default_rng(
            seed=_hash_function(
                input_str=sim1.observations[0].name[idx],
                user_seed=987654321,
            )
        )

        rand = rng.normal(
            loc=drift_params.sampling_gaussian_loc,
            scale=drift_params.sampling_gaussian_scale,
        )

        gain_arr_size = (
            drift_params.sampling_freq_Hz * drift_params.calibration_period_sec
        )

        gain_arr = 1.0 + rand * drift_params.sigma_drift * np.linspace(
            0, 1, gain_arr_size
        )

        div, mod = (
            tod_size // gain_arr_size,
            tod_size % gain_arr_size,
        )

        for i in np.arange(div):
            tod[i * gain_arr_size : (i + 1) * gain_arr_size] *= gain_arr

        tod[div * gain_arr_size :] *= gain_arr[:mod]

    # Testing if the two tods are same
    np.testing.assert_array_equal(
        sim1.observations[0].gain_wrapper, sim1.observations[0].gain_native
    )

    sim1.flush()


class Test_thermal_gain:
    """Class to group tests for checking if the thermal gain drift is
    applied correctly.

    The idea is to first generate a gain drift tod `gain_wrapper` using
    the wrapper function. It will be
    `1 + therm_fluc_amplitude*noise_stream*(1+rand*det_mismatch)/focalplane_Tbath`.
    We compare this with
    `1 + therm_fluc_amplitude*(1+rand*det_mismatch)/focalplane_Tbath`
    that is stored in `gain_native`. Since elements of `noise_timestream`
    are less than 1, we expect the elements of `gain_wrapper` to be
    smaller than the elements of `gain_native`. The test functions test
    exactly this.
    """

    start_time = Time("2034-05-02")
    duration_s = 2 * 24 * 3600
    sampling_freq_Hz = 1

    # Three detectors with two different focalplane attributes
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

    def test_thermal_gain_drift_with_mismatch(self, tmp_path):
        """This function test if the thermal gain drifts are applied
        correctly with detector mismatch.
        """

        drift_params = lbs.GainDriftParams(
            drift_type=lbs.GainDriftType.THERMAL_GAIN,
            sampling_freq_Hz=self.sampling_freq_Hz,
            sampling_dist=lbs.SamplingDist.GAUSSIAN,
            sampling_gaussian_loc=0.5,
            sampling_gaussian_scale=0.2,
            detector_mismatch=0.9,
        )

        sim1 = lbs.Simulation(
            base_path=tmp_path / "gd_thermalgain_test",
            start_time=self.start_time,
            duration_s=self.duration_s,
        )

        sim1.create_observations(
            detectors=self.dets,
            split_list_over_processes=False,
            num_of_obs_per_detector=1,
        )

        # gain_wrapper stores the tod applied with the wrapper function
        # and is tested against gain_native where the gain is applied right
        # within this function
        sim1.observations[0].gain_wrapper = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_native = np.ones_like(sim1.observations[0].tod)

        sim1.apply_gaindrift(
            drift_params=drift_params,
            component="gain_wrapper",
            user_seed=987654321,
        )

        for idx, tod in enumerate(sim1.observations[0].gain_native):
            rng = np.random.default_rng(
                seed=_hash_function(
                    input_str=sim1.observations[0].name[idx],
                    user_seed=987654321,
                )
            )

            rand = rng.normal(loc=0.7, scale=0.5)
            thermal_factor = drift_params.thermal_fluctuation_amplitude_K * (
                1.0 + rand * drift_params.detector_mismatch
            )

            tod *= 1.0 + thermal_factor / drift_params.focalplane_Tbath_K

        for i in np.arange(len(self.dets)):
            assert (
                sim1.observations[0].gain_wrapper[i]
                < sim1.observations[0].gain_native[i]
            ).all(), (
                f"The assertion is failed for detector {sim1.observations[0].name[i]}"
            )

        sim1.flush()

    def test_thermal_gain_drift_no_mismatch(self, tmp_path):
        """This function test if the thermal gain drifts are applied
        correctly without detector mismatch.
        """

        drift_params = lbs.GainDriftParams(
            drift_type=lbs.GainDriftType.THERMAL_GAIN,
            sampling_freq_Hz=self.sampling_freq_Hz,
            sampling_dist=lbs.SamplingDist.GAUSSIAN,
            sampling_gaussian_loc=0.5,
            sampling_gaussian_scale=0.2,
            detector_mismatch=0.0,
        )

        sim1 = lbs.Simulation(
            base_path=tmp_path / "gd_thermalgain_test",
            start_time=self.start_time,
            duration_s=self.duration_s,
        )

        sim1.create_observations(
            detectors=self.dets,
            split_list_over_processes=False,
            num_of_obs_per_detector=1,
        )

        sim1.observations[0].gain_wrapper = np.ones_like(sim1.observations[0].tod)
        sim1.observations[0].gain_native = np.ones_like(sim1.observations[0].tod)

        sim1.apply_gaindrift(
            drift_params=drift_params,
            component="gain_wrapper",
            user_seed=987654321,
        )

        for idx, tod in enumerate(sim1.observations[0].gain_native):
            rng = np.random.default_rng(
                seed=_hash_function(
                    input_str=sim1.observations[0].name[idx],
                    user_seed=987654321,
                )
            )

            rand = rng.normal(loc=0.7, scale=0.5)
            thermal_factor = drift_params.thermal_fluctuation_amplitude_K * (
                1.0 + rand * drift_params.detector_mismatch
            )

            tod *= 1.0 + thermal_factor / drift_params.focalplane_Tbath_K

        for i in np.arange(len(self.dets)):
            assert (
                sim1.observations[0].gain_wrapper[i]
                < sim1.observations[0].gain_native[i]
            ).all(), (
                f"The assertion is failed for detector {sim1.observations[0].name[i]}"
            )

        sim1.flush()
