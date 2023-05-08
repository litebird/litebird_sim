import numpy as np
from numba import njit
import logging
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm


def apply_non_linearities(
    obs,
    T_c=180e-3,
    transition_width=2e-3,
    R_normal=1.0, 
    P_loading=0.5e-12,
    T_bath=100e-3,
    z_series=0.1+0j,
    n_index=3.6,
    tau_intrinsic=33e-3,
    V_bias=0.8e-6,
    P_sat_for_G=1.3e-12,
    ):
    
    for ob in obs:
        tod = getattr(ob,'tod')	
        assert len(tod.shape) == 2
        num_of_dets = tod.shape[0]

        for i in range(num_of_dets):
            #0.18423854 K to pW @140ghz
            P_loads=np.arange(0.18423854*1e-12*tod[i].min(), 0.18423854*1e-12*tod[i].max(), 0.18423854*1e-12*(tod[i].max()-tod[i].min())/1e4)
            #P_loads=np.arange(0*1e-12, 1.1e-12, 1e-15)
            S=[]
            for p_load in tqdm(P_loads):
                dic=calc_tes(P_loading=p_load)
                S.append(dic["S"]*dic["S_relative"])

            S=np.array(S)
            #print(S.shape)
            dI=S.cumsum()*0.18423854*1e-12*(tod[i].max()-tod[i].min())/1e4
            
            P_loads = P_loads*1e12 #to pW
            I_sim = dI-dI.min()+0.8e-6
            I_sim = I_sim*1e6 #to muA
            
            plt.plot(P_loads,I_sim)
            
            Curr_NL = interpolate.interp1d(P_loads,-I_sim,fill_value='extrapolate')
            
            tod[i] = Curr_NL(tod[i]) #TOD in muA
            
            
            coef = np.polyfit(x,y,1)
            poly1d_fn = np.poly1d(coef)
            
            
   

#2 Jun 2022 - TdH

# TES Model for LiteBIRD 
# Differs from my 2020 model in the following ways:
# 1. G(T) rather than some fixed G
# 2. Safety factor definition P_tot = 2.5*P_opt, so P_electrical = 1.5*P_opt
# 3. Parasitics modeled as an effective complex series impedance




### Derive detector properties

def calc_tes(T_c=180e-3, transition_width=2e-3, R_normal=1.0, 
             P_loading=0.5e-12, T_bath=100e-3,
             z_series=0.1+0j, n_index=3.6, tau_intrinsic=33e-3,
             V_bias=0.8e-6, P_sat_for_G=1.3e-12,
             debug=False): # SI units
    T = np.arange(0,360e-3,0.1e-3) # K
    R_frac_T = (np.arctan((T-T_c)/(transition_width/2.)) / (np.pi/2) + 1)/2
    R_T = R_normal * R_frac_T # Ohms
    dR_dT_T = R_normal * (transition_width/2.) / (np.pi*((T-T_c)**2 + (transition_width/2.)**2)) # Ohms/K ????
    alpha_T = T/R_T*dR_dT_T
    # old:
    # P_opt + V^2/R = G (T - T_bath)
    # G = P_sat / (T_c - T_bath) # W/K 
    # new:
    # P_opt + R V^2/|R+z_s|^2 = K(T^n - T_bath^n)
    K = P_sat_for_G / (T_c**n_index  - T_bath**n_index)
    loop_gain_T = V_bias**2*alpha_T/(K*n_index*(T_c**n_index)) * (2*R_T**2/np.abs(R_T+z_series)**3 - R_T/np.abs(R_T+z_series)**2)
    tau_T = tau_intrinsic / (1. + loop_gain_T)
    S_relative_T_complex = loop_gain_T / (1. + loop_gain_T) * (2*(R_T+z_series.conjugate()) - np.abs(R_T+z_series))/(2*R_T-np.abs(R_T+z_series))
    S_relative_T = np.abs(S_relative_T_complex)
    S_T = -1. * np.sqrt(2) / V_bias * S_relative_T # A/W
    # solve for T by requiring power balance 
    power_balance = P_loading + R_T*V_bias**2/np.abs(R_T+z_series)**2 - K*(T**n_index - T_bath**n_index)
    #print(np.sign(power_balance).shape)
    #print(np.diff(np.sign(power_balance)).shape)
    wh = np.where(np.diff(np.sign(power_balance)))[0]
    #print(wh)
    rightmost_wh = wh[-1]
    T_0 = np.interp(0., [power_balance[rightmost_wh+1],power_balance[rightmost_wh]],
                    [T[rightmost_wh+1],T[rightmost_wh]])                    
   
    R_0 = interp1d(T, R_T, kind='cubic')(T_0)
    loop_gain_0 = interp1d(T, loop_gain_T, kind='cubic')(T_0)
    tau_0 = interp1d(T, tau_T, kind='cubic')(T_0)
    S_relative_0 = interp1d(T, S_relative_T, kind='cubic')(T_0)
    S_0 = interp1d(T,S_T)(T_0)
    P_electrical = R_0*V_bias**2/np.abs(R_0+z_series)**2
    
    # output results
    return {'T':T_0,'R':R_0,'L':loop_gain_0,'tau':tau_0, 
            'S_relative':S_relative_0,'S':S_0,'P_electrical':P_electrical,
            'P_sat_for_G':P_sat_for_G}