import toml
import distutils
import litebird_sim as lbs
import logging as log
import numpy as np
import healpy as hp
from astropy import constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo

COND_THRESHOLD = 1e10

def _dBodTrj(nu):
    return 2*const.k_B.value*nu*nu*1e18/const.c.value/const.c.value

def _dBodTth(nu):
    x = const.h.value*nu*1e9/const.k_B.value/cosmo.Tcmb0.value
    ex=np.exp(x)
    exm1=ex-1.e0
    return 2*const.h.value*nu*nu*nu*1e27/const.c.value/const.c.value/exm1/exm1*ex*x/cosmo.Tcmb0.value

class HwpSys:
    def __init__(self, simulation):
        self.sim = simulation
        self.imo = self.sim.imo

#        self.parameters = simulation.parameters['hwp_sys']


    def read_parameters(self,channel):
        #booo=self.parameters['booo']
        
        self.nside = 256
        self.npix = hp.nside2npix(self.nside)

        self.h1 = 0.0
        self.h2 = 0.0
        self.beta = 0.0
        self.z1 = 0.0
        self.z2 = 0.0

        self.integrate_in_band = False
        self.built_map_on_the_fly = True

        if self.integrate_in_band:
            self.band_filename = 'inputs/MFT_band166_noxpol.txt'
            self.freqs,self.h1,self.h2,self.beta,self.z1,self.z2 = np.loadtxt(
            	self.band_filename,unpack=True,skiprows=1)

            self.nfreqs = len(self.freqs)

            self.cmb2bb = _dBodTth(self.freqs)
            self.norm = self.cmb2bb.sum()

            myinstr = {}
            for ifreq in range(self.nfreqs):
                 myinstr['ch'+str(ifreq)] = {'bandcenter_ghz':self.freqs[ifreq], 'bandwidth_ghz': 0,
                    'fwhm_arcmin':channel.fwhm_arcmin, 'p_sens_ukarcmin':0.}

            params = lbs.MbsParameters(
                make_cmb =True,
                make_fg = True,
                fg_models =["pysm_synch_0", "pysm_freefree_1","pysm_dust_0"],
                nside = self.nside,
                gaussian_smooth = True,
                bandpass_int = False,
            )

            mbs = lbs.Mbs(
                simulation = self.sim,
                parameters = params,
                instrument = myinstr,
            )

            maps = mbs.run_all()[0]
            self.maps = np.empty((self.nfreqs,3,self.npix))
            for ifreq in range(self.nfreqs):
                self.maps[ifreq] = maps['ch'+str(ifreq)]
            del(maps)

        else:

            params = lbs.MbsParameters(
                make_cmb =True,
                make_fg = True,
                fg_models =["pysm_synch_0", "pysm_freefree_1","pysm_dust_0"],
                nside = self.nside,
                gaussian_smooth = True,
                bandpass_int = False,
            )

            mbs = lbs.Mbs(
            	simulation=self.sim,
            	parameters=params,
                channel_list=channel
            )

            self.maps = mbs.run_all()[0][channel.channel]

    def fill_tod(self,obs,pointings,hwp_radpsec):
                
        times = obs.get_times()
        
        if self.built_map_on_the_fly:
        	self.atd = np.zeros((self.npix,3))
        	self.ata = np.zeros((self.npix,3,3))

        for idet in range(obs.n_detectors):
            pix = hp.ang2pix(self.nside,pointings[idet,:,0],pointings[idet,:,1])

            #add hwp rotation
            ca = np.cos(pointings[idet,:,2]+2*times*hwp_radpsec)
            sa = np.sin(pointings[idet,:,2]+2*times*hwp_radpsec)

            if self.integrate_in_band:
                J11 = ((1+self.h1[:,np.newaxis])*ca**2-
                	(1+self.h2[:,np.newaxis])*sa**2*np.exp(1j*self.beta[:,np.newaxis])-
                	(self.z1[:,np.newaxis]+self.z2[:,np.newaxis])*ca*sa)
                J12 = (((1+self.h1[:,np.newaxis])+(1+self.h2[:,np.newaxis])*
                	np.exp(1j*self.beta[:,np.newaxis]))*ca*sa+
                	self.z1[:,np.newaxis]*ca**2-self.z2[:,np.newaxis]*sa**2)
                
                if self.built_map_on_the_fly:
                	tod = ((0.5*(np.abs(J11)**2+np.abs(J12)**2)*self.maps[:,0,pix] + 
                        0.5*(np.abs(J11)**2-np.abs(J12)**2)*self.maps[:,1,pix] +
                	    (J11*J12.conjugate()).real*self.maps[:,2,pix]
                	    )*self.cmb2bb[:,np.newaxis]).sum(axis=0)/self.norm
                else: 
                    obs.tod[idet,:] = ((0.5*(np.abs(J11)**2+np.abs(J12)**2)*self.maps[:,0,pix] + 
                        0.5*(np.abs(J11)**2-np.abs(J12)**2)*self.maps[:,1,pix] +
                	    (J11*J12.conjugate()).real*self.maps[:,2,pix]
                	    )*self.cmb2bb[:,np.newaxis]).sum(axis=0)/self.norm
                
            else:
                J11 = (1+self.h1)*ca**2-(1+self.h2)*sa**2*np.exp(1j*self.beta)-(self.z1+self.z2)*ca*sa
                J12 = ((1+self.h1)+(1+self.h2)*np.exp(1j*self.beta))*ca*sa+self.z1*ca**2-self.z2*sa**2

                if self.built_map_on_the_fly:
                	tod = (0.5*(np.abs(J11)**2+np.abs(J12)**2)*self.maps[0,pix] + 
                    0.5*(np.abs(J11)**2-np.abs(J12)**2)*self.maps[1,pix] +
                	(J11*J12.conjugate()).real*self.maps[2,pix])
                else:
                    obs.tod[idet,:] = (0.5*(np.abs(J11)**2+np.abs(J12)**2)*self.maps[0,pix] + 
                        0.5*(np.abs(J11)**2-np.abs(J12)**2)*self.maps[1,pix] +
                        (J11*J12.conjugate()).real*self.maps[2,pix])

            if self.built_map_on_the_fly:
                	#this needs to be changed when the profiles for the solver will be introduced
                    self.atd[pix,0] += tod*0.5
                    self.atd[pix,1] += tod*ca
                    self.atd[pix,2] += tod*sa

                    self.ata[pix,0,0] += 0.25
                    self.ata[pix,1,0] += 0.5*ca
                    self.ata[pix,2,0] += 0.5*sa
                    self.ata[pix,1,1] += ca*ca
                    self.ata[pix,2,1] += ca*sa
                    self.ata[pix,2,2] += sa*sa

        return

    def make_map(self):
        #mpi?
        self.ata[:,0,1] = self.ata[:,1,0]
        self.ata[:,0,2] = self.ata[:,2,0]
        self.ata[:,1,2] = self.ata[:,2,1]
        
        cond = np.linalg.cond(self.ata)
        res = np.full_like(self.atd, hp.UNSEEN)
        mask = cond < COND_THRESHOLD
        res[mask] = np.linalg.solve(self.ata[mask], self.atd[mask])
        return res.T

#    def fill_ata_atd(self,obs,pointings,hwp_radpsec):








