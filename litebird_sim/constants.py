import numpy as np
from astropy.constants import c as c_light
from astropy.constants import h, k_B

# Name of the environment variable used in the convolution
NUM_THREADS_ENVVAR = "OMP_NUM_THREADS"
NUMBA_NUM_THREADS_ENVVAR = "OMP_NUM_THREADS"

ARCMIN_TO_RAD = np.pi / 180 / 60

C_LIGHT_M_OVER_S = c_light.value
H = h.value
K_B = k_B.value

C_LIGHT_KM_OVER_S = C_LIGHT_M_OVER_S / 1e3
H_OVER_K_B = H / K_B

T_CMB_K = 2.72548  # Fixsen 2009 http://arxiv.org/abs/0911.1955

# Dipole parameters from https://arxiv.org/abs/1807.06207
SOLAR_VELOCITY_KM_S = 369.8160
SOLAR_VELOCITY_GAL_LAT_RAD = 0.842_173_724
SOLAR_VELOCITY_GAL_LON_RAD = 4.608_035_744_4

EARTH_L2_DISTANCE_KM = 1_496_509.30522
