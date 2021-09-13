import numpy as np
import litebird_sim as lbs
from astropy.time import Time
import matplotlib.pylab as plt

orbit = lbs.SpacecraftOrbit(start_time=Time("2023-01-01"))

posvel = lbs.spacecraft_pos_and_vel(
    orbit,
    start_time=orbit.start_time,
    time_span_s=3.15e7,  # One year
    delta_time_s=86_400.0,  # One day
)

# posvel.positions_km is a N×3 array containing the XYZ position
# of the spacecraft calculated every day for one year. We compute
# the modulus of the position, which is of course the
# Sun-LiteBIRD distance.
sun_distance_km = np.linalg.norm(posvel.positions_km, axis=1)

# We do the same with the velocities
speed_km_s = np.linalg.norm(posvel.velocities_km_s, axis=1)

# Plot distance and speed as functions of time
_, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))

ax[0].plot(sun_distance_km)
ax[0].set_xlabel("Day")
ax[0].set_ylabel("Distance [km]")

ax[1].plot(speed_km_s)
ax[1].set_xlabel("Day")
ax[1].set_ylabel("Speed [km/s]")
