import numpy as np
import litebird_sim as lbs
from astropy.time import Time
import matplotlib.pyplot as plt

start_time = Time("2034-05-02")
duration_s = 100
sampling_freq_Hz = 1

# Creating a list of detectors. The three detectors belong to two
# different wafer groups.
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

# Defining the gain drift simulation parameters with no detector mismatch
drift_params_no_mismatch = lbs.GainDriftParams(
    drift_type=lbs.GainDriftType.THERMAL_GAIN,
    sampling_freq_Hz=sampling_freq_Hz,
    focalplane_group="wafer",
    detector_mismatch=0.0,
)

# Defining the gain drift simulation parameters with detector mismatch
drift_params_with_mismatch = lbs.GainDriftParams(
    drift_type=lbs.GainDriftType.THERMAL_GAIN,
    sampling_freq_Hz=sampling_freq_Hz,
    focalplane_group="wafer",
    detector_mismatch=1.0,
)

sim1 = lbs.Simulation(
    start_time=start_time,
    duration_s=duration_s,
)

sim1.create_observations(
    detectors=dets,
    split_list_over_processes=False,
    num_of_obs_per_detector=1,
)

sim1.observations[0].thermalgain_tod_no_mismatch = np.ones_like(
    sim1.observations[0].tod
)
sim1.observations[0].thermalgain_tod_with_mismatch = np.ones_like(
    sim1.observations[0].tod
)

# Generating the gain drift factors with no detector mismatch
sim1.apply_gaindrift(
    drift_params=drift_params_no_mismatch,
    component="thermalgain_tod_no_mismatch",
    user_seed=12345,
)

# Generating the gain drift factors with detector mismatch
sim1.apply_gaindrift(
    drift_params=drift_params_with_mismatch,
    component="thermalgain_tod_with_mismatch",
    user_seed=12345,
)

plt.figure(figsize=(8, 10))

plt.subplot(211)
for idx in range(sim1.observations[0].tod.shape[0]):
    plt.plot(
        sim1.observations[0].thermalgain_tod_no_mismatch[idx],
        label=sim1.observations[0].name[idx],
    )

plt.xlabel("Time (in seconds)")
plt.ylabel("Linear gain factor amplitude")
plt.title("Thermal gain drift factor with no detector mismatch")
plt.legend()

plt.subplot(212)
for idx in range(sim1.observations[0].tod.shape[0]):
    plt.plot(
        sim1.observations[0].thermalgain_tod_with_mismatch[idx],
        label=sim1.observations[0].name[idx],
    )

plt.xlabel("Time (in seconds)")
plt.ylabel("Linear gain factor amplitude")
plt.title("Thermal gain drift factor with detector mismatch")
plt.legend()
plt.tight_layout()
