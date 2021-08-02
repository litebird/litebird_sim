from numba import njit
import numpy as np
from astropy.constants import c as c_light
import astropy

from astropy.constants import h, k_B

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

    return dx*vx+dy*vy+dz*vz

	
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
    q_x
    ):
    
    beta_dot_n = compute_scalar_product(theta,phi,vx,vy,vz)/C_LIGHT_KM_S
    beta = np.sqrt(vx*vx+vy*vy+vz*vz)/C_LIGHT_KM_S   #here? or returned from compute_scalar_product?
    gamma = 1/np.sqrt(1-beta**2)

    if dipoletype == 'linear':
        dip = T_CMB*(1+beta_dot_n) 
    if dipoletype == 'quadratic_from_lin_T':    #up to second order in beta, using the linear temperature approx. (linearization of thermodyn. T)  
        dip = T_CMB*(1+beta_dot_n+q_x*(beta_dot_n)**2-0.5*beta**2)
    if dipoletype == 'quadratic_exact':  #up to second order in beta, including second order in the expansion of thermodyn. temperature
        dip = T_CMB*(1+beta_dot_n+(beta_dot_n)**2-0.5*beta**2)
    if dipoletype == 'total':
        dip = T_CMB/gamma/(1-beta_dot_n)

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
    q_x
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
            q_x
            )



def add_dipole(
    obs,
    pointings,
    velocity, #should be the vel. vector from lbs.l2_pos_and_vel_in_obs but with
              #the length of nsamples in order to make the product with the direction
    dipoletype,
    dipoleunits, #?
    T_CMB,
    frequency    #e.g. central frequency of channel from 
                #lbs.FreqChannelInfo.from_imo(url="/releases/v1.0/satellite/"+telescope+"/"+channel+"/channel_info",imo=imo).bandcenter_ghz
    ):
    
    nu = frequency*10**6.  #freq in GHz
    x = h.value*nu/(k_B.value*T_CMB)

    q_x = 0.5*x*(np.exp(x)+1)/(np.exp(x)-1)  

    assert obs.tod.shape == pointings.shape[0:2]

    assert obs.tod.shape[1] == velocity.shape[0]
 
    for idet in range(obs.n_detectors):
        add_dipole_for_one_detector(
            obs.tod[idet],
            pointings[idet,:,0],
            pointings[idet,:,1],
            velocity,
            dipoletype,
            dipoleunits,
            T_CMB,
            q_x
            )
    


    

