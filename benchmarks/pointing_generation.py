# -*- encoding: utf-8 -*-

"""
This program tests the speed of the code that generates quaternions
and pointings.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add the `..` directory to PYTHONPATH, so that we can import "litebird_sim"
sys.path.append(str(Path(__file__).parent / ".."))

import litebird_sim as lbs  # noqa:E402


if len(sys.argv) == 2:
    duration_s = float(sys.argv[1])
else:
    duration_s = 86400.0

sim = lbs.Simulation(
    start_time=0.0,
    duration_s=duration_s,
    random_seed=12345,
)

sim.create_observations(
    detectors=[lbs.DetectorInfo(name="dummy", sampling_rate_hz=50.0)],
    num_of_obs_per_detector=1,
    split_list_over_processes=False,
)
assert len(sim.observations) == 1
obs = sim.observations[0]

scanning_strategy = lbs.SpinningScanningStrategy(
    spin_sun_angle_rad=0.0, precession_rate_hz=1.0, spin_rate_hz=1.0
)

# Generate the quaternions (one per each second)
start = time.perf_counter_ns()
sim.set_scanning_strategy(
    scanning_strategy=scanning_strategy, delta_time_s=1.0, append_to_report=False
)
stop = time.perf_counter_ns()
elapsed_time = (stop - start) * 1.0e-9

print(
    "Elapsed time for set_scanning_strategy: {} s".format(elapsed_time),
)
print("Shape of the quaternions: ", sim.spin2ecliptic_quats.quats.shape)
print(
    "Speed: {:.1e} quat/s".format(
        sim.spin2ecliptic_quats.quats.shape[0] / elapsed_time
    ),
)

instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(15.0))

# Compute the pointings by running a "slerp" operation
start = time.perf_counter_ns()
pointings_and_polangle = lbs.get_pointings(
    obs,
    spin2ecliptic_quats=sim.spin2ecliptic_quats,
    detector_quats=np.array([[0.0, 0.0, 0.0, 1.0]]),
    bore2spin_quat=instr.bore2spin_quat,
)
stop = time.perf_counter_ns()
elapsed_time = (stop - start) * 1.0e-9

print("Elapsed time for get_pointings: {} s".format((stop - start) * 1e-9))
print("Shape of the pointings: ", pointings_and_polangle.shape)
print(
    "Speed: {:.1e} pointings/s".format(pointings_and_polangle.shape[1] / elapsed_time),
)
