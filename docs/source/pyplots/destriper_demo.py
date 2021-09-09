import numpy as np
import astropy.units as u
from numpy.random import MT19937, RandomState, SeedSequence

import litebird_sim as lbs

sim = lbs.Simulation(
    base_path="/tmp/destriper_output", start_time=0, duration_s=86400.0
)

sim.generate_spin2ecl_quaternions(
    scanning_strategy=lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=np.deg2rad(30),  # CORE-specific parameter
        spin_rate_hz=0.5 / 60,  # Ditto
        # We use astropy to convert the period (4 days) in
        # seconds
        precession_rate_hz=1.0 / (4 * u.day).to("s").value,
    )
)
instr = lbs.InstrumentInfo(name="core", spin_boresight_angle_rad=np.deg2rad(65))

# We create two detectors, whose polarization angles are separated by π/2
sim.create_observations(
    detectors=[
        lbs.DetectorInfo(name="0A", sampling_rate_hz=10),
        lbs.DetectorInfo(
            name="0B", sampling_rate_hz=10, quat=lbs.quat_rotation_z(np.pi / 2)
        ),
    ],
    dtype_tod=np.float64,
    n_blocks_time=lbs.MPI_COMM_WORLD.size,
    distribute=False,
)

# Generate some white noise
rs = RandomState(MT19937(SeedSequence(123456789)))
for curobs in sim.observations:
    curobs.tod *= 0.0
    curobs.tod += rs.randn(*curobs.tod.shape)

params = lbs.DestriperParameters(
    nside=16, return_hit_map=True, return_binned_map=True, return_destriped_map=True
)

result = lbs.destripe(sim, instr, params)

import healpy

# Plot the I map
healpy.mollview(result.binned_map[0])
