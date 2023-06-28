import numpy as np
import litebird_sim as lbs
from astropy.time import Time
import matplotlib.pyplot as plt

start_time = Time("2034-05-02")
duration_s = 4 * 24 * 3600
sampling_freq_Hz = 1

# Creating a list of detectors.
dets = [
    lbs.DetectorInfo(
        name="det_A_wafer_1", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_1"
    ),
    lbs.DetectorInfo(
        name="det_B_wafer_1", sampling_rate_hz=sampling_freq_Hz, wafer="wafer_1"
    ),
]

# Defining the gain drift simulation parameters
drift_params = lbs.GainDriftParams(
    drift_type=lbs.GainDriftType.LINEAR_GAIN,
    calibration_period_sec=24 * 3600,
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

sim1.observations[0].lingain_tod = np.ones_like(sim1.observations[0].tod)

# Applying gain drift using the `Simulation` class method
sim1.apply_gaindrift(
    drift_params=drift_params,
    component="lingain_tod",
)

plt.figure(figsize=(8, 5))

time_domain = (
    np.arange(sim1.observations[0].tod.shape[1])
    / sampling_freq_Hz
    / 24
    / 3600
)

for idx in range(sim1.observations[0].tod.shape[0]):
    plt.plot(
        time_domain,
        sim1.observations[0].lingain_tod[idx],
        label=sim1.observations[0].name[idx],
    )

plt.xlabel("Time (in days)")
plt.ylabel("Linear gain factor amplitude")
plt.legend()
