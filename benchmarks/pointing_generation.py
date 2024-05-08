#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
This program tests the speed of the code that generates quaternions
and pointings.
"""

import sys
import time
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
    detectors=[
        lbs.DetectorInfo(name="dummy1", sampling_rate_hz=30.0),
        lbs.DetectorInfo(name="dummy2", sampling_rate_hz=30.0),
        lbs.DetectorInfo(name="dummy3", sampling_rate_hz=30.0),
        lbs.DetectorInfo(name="dummy4", sampling_rate_hz=30.0),
    ],
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
sim.set_instrument(instr)

# Compute the pointings by running a "slerp" operation
sim.prepare_pointings()

start = time.perf_counter_ns()

pointings_and_orientation = np.empty(
    shape=(len(sim.detectors), sim.observations[0].n_samples, 3),
    dtype=np.float64,
)
for cur_obs in sim.observations:
    for det_idx in range(cur_obs.n_detectors):
        (cur_pointings, hwp_angle) = cur_obs.get_pointings(det_idx)
        pointings_and_orientation[det_idx, :, :] = cur_pointings

stop = time.perf_counter_ns()
elapsed_time = (stop - start) * 1.0e-9

print("Elapsed time for get_pointings: {} s".format((stop - start) * 1e-9))
print("Shape of the pointings: ", pointings_and_orientation.shape)
print(
    "Speed: {:.1e} pointings/s".format(
        pointings_and_orientation.shape[1] / elapsed_time
    ),
)

array_file = Path("pointings.npy")

if array_file.exists():
    with array_file.open("rb") as inp_f:
        reference = np.load(inp_f)
        np.save(file=Path("difference.npy"), arr=reference - pointings_and_orientation)
        np.testing.assert_array_almost_equal(reference, pointings_and_orientation)
        print(f'The array looks the same as the one in "{array_file}"')
else:
    with array_file.open("wb") as out_f:
        np.save(out_f, pointings_and_orientation)
    print(f'Array saved for reference in "{array_file}"')
