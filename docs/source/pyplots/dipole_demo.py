from astropy.time import Time
import numpy as np
import litebird_sim as lbs
import matplotlib.pylab as plt

start_time = Time("2022-01-01")
time_span_s = 120.0  # Two minutes
sampling_hz = 20

sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

# We pick a simple scanning strategy where the spin axis is aligned
# with the Sun-Earth axis, and the spacecraft spins once every minute
scanning = lbs.SpinningScanningStrategy(
    spin_sun_angle_rad=np.deg2rad(0),
    precession_rate_hz=0,
    spin_rate_hz=1 / 60,
    start_time=start_time,
)

# The spacecraft spins pretty fast (once per minute!), so we
# need to pick delta_time_s ≪ 1/spin_rate_hz
spin2ecliptic_quats = scanning.set_scanning_strategy(
    start_time, time_span_s, delta_time_s=5.0
)

# We simulate an instrument whose boresight is perpendicular to
# the spin axis.
instr = lbs.InstrumentInfo(
    boresight_rotangle_rad=0.0,
    spin_boresight_angle_rad=np.deg2rad(90),
    spin_rotangle_rad=np.deg2rad(75),
)

# A simple detector looking along the boresight direction
det = lbs.DetectorInfo(
    name="Boresight_detector",
    sampling_rate_hz=sampling_hz,
    bandcenter_ghz=100.0,
)

(obs,) = sim.create_observations(detectors=[det])
pointings = lbs.get_pointings(
    obs,
    spin2ecliptic_quats=spin2ecliptic_quats,
    detector_quats=[det.quat],
    bore2spin_quat=instr.bore2spin_quat,
)

# Simulate the orbit of the spacecraft and compute positions and
# velocities
orbit = lbs.SpacecraftOrbit(obs.start_time)
pos_vel = lbs.spacecraft_pos_and_vel(orbit, obs, delta_time_s=60.0)

t = obs.get_times()
t -= t[0]  # Make `t` start from zero

# Create two plots: the first shows the full two-minute time span, and
# the second one shows a zoom over the very first points of the TOD.
_, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Make a plot of the TOD using all the dipole types
for type_idx, dipole_type in enumerate(lbs.DipoleType):
    obs.tod[0][:] = 0  # Reset the TOD
    lbs.add_dipole_to_observations(obs, pos_vel, dipole_type=dipole_type)

    axes[0].plot(t, obs.tod[0], label=str(dipole_type))
    axes[1].plot(t[0:20], obs.tod[0][0:20], label=str(dipole_type))

axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Signal [K]")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Signal [K]")
axes[1].legend()
