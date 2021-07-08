from numba import njit
import numpy as np
from astropy.constants import c as c_light

C_LIGHT_KM_S = c_light.value/1e3


@njit
def compute_scalar_product(
	theta,
	phi,
	vx,
	vy,
	vz,
	):
	dx, dy, dz = np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)
	prod = dx*vx+dy*vy+dz*vz
	return prod

	
@njit
def compute_dipole_for_one_sample(
	theta,
	phi,
	vx,
	vy,
	vz,
	dipoletype,
    dipoleunits,
    T_CMB,
	):
    
    beta = compute_scalar_product(theta,phi,vx,vy,vz)/C_LIGHT_KM_S

   	dip = T_CMB*(1+beta)

    return dip


@njit
def add_dipole_for_one_detector(
	tod_det,
	theta_det,
	phi_det,
	velocity,
	dipoletype,
    dipoleunits,
    T_CMB,
	):
    
    #
    for row in range(len(tod_det)):
    	tod_det[row] += compute_dipole_for_one_sample(
    		theta_det[row],
    		phi_det[row],
    		velocity[row][0],
    		velocity[row][1],
    		velocity[row][2],
    		dipoletype,
    		dipoleunits,
    		T_CMB,
    		)



def add_dipole(
    obs,
    pointings,
    velocity, 
    dipoletype,
    dipoleunits, #?
    T_CMB,
    ):
    
    assert obs.tod.shape == pointings.shape[0:2]

    assert obs.tod.shape[1] == velocity.shape[0]
 
    for idet in range(obs.n_detectors):
    	add_dipole_for_one_detector(
    		obs.tod[idet],
    		pointings[idet,:,0],
    		pointings[idet,:,1],
    		velocity,
    		)
    


    

