#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
This program measures the time spent by the map-makers (binner, destriper,
pair-differencing) on precomputed pointings. Each map-maker runs once to
compile the Numba kernels, then the best of a few runs is printed.

Usage: python benchmarks/map_makers.py [DURATION_S] [NSIDE]
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add the `..` directory to PYTHONPATH, so that we can import "litebird_sim"
sys.path.append(str(Path(__file__).parent / ".."))

import litebird_sim as lbs  # noqa:E402

duration_s = float(sys.argv[1]) if len(sys.argv) > 1 else 86400.0
nside = int(sys.argv[2]) if len(sys.argv) > 2 else 256
repetitions = 3

sim = lbs.Simulation(start_time=0.0, duration_s=duration_s, random_seed=12345)
sim.set_scanning_strategy(
    lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=np.deg2rad(45.0),
        precession_rate_hz=1 / (192.348 * 60),
        spin_rate_hz=0.05 / 60,
    )
)
sim.set_instrument(
    lbs.InstrumentInfo(name="i", spin_boresight_angle_rad=np.deg2rad(50.0))
)
# Two T/B pairs, for the pair-differencing
sim.create_observations(
    detectors=[
        lbs.DetectorInfo(
            name=f"{pixel}{pol}",
            sampling_rate_hz=10.0,
            net_ukrts=50.0,
            pol_angle_rad=angle + 0.3 * pixel,
            pol_efficiency=0.95,
            wafer="W",
            pixel=pixel,
            pol=pol,
            quat=lbs.RotQuaternion(
                np.array(lbs.quat_rotation_y(np.deg2rad(1.0 + pixel)))
            ),
        )
        for pixel in range(2)
        for pol, angle in (("T", 0.0), ("B", np.pi / 2))
    ],
    tod_dtype=np.float64,
)
sim.set_hwp(lbs.IdealHWP(ang_speed_radpsec=2 * np.pi * 46 / 60))
sim.prepare_pointings()
sim.precompute_pointings()
for obs in sim.observations:
    obs.tod[:] = np.random.default_rng(1).normal(size=obs.tod.shape)

n_samples = sum(obs.tod.size for obs in sim.observations)
print(f"{n_samples} samples, nside {nside}")


def best_time(function):
    function()  # Compile the Numba kernels
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


destriper_params = lbs.DestriperParameters(
    samples_per_baseline=600, iter_max=10, threshold=1e-30
)
benchmarks = {
    "binner": lambda: lbs.make_binned_map(nside, sim.observations),
    "destriper": lambda: lbs.make_destriped_map(
        nside, sim.observations, params=destriper_params
    ),
    "pair-differencing": lambda: lbs.make_pair_differenced_map(nside, sim.observations),
}
for name, function in benchmarks.items():
    print(f"{name:20s} {best_time(function):8.3f} s")
