# -*- encoding: utf-8 -*-
import numpy as np
from astropy.constants import c as c_light
from astropy.constants import h, k_B

ARCMIN_TO_RAD = np.pi / 180 / 60

C_LIGHT_KM_S = c_light.value / 1e3
H_OVER_K_B = h.value / k_B.value

T_CMB_K = 2.72548  # Fixsen 2009 http://arxiv.org/abs/0911.1955

SOLAR_VELOCITY_KM_S = 369.8160
SOLAR_VELOCITY_GAL_LAT_RAD = 0.842_173_724
SOLAR_VELOCITY_GAL_LON_RAD = 4.608_035_744_4

EARTH_L2_DISTANCE_KM = 1_496_509.30522
