import toml
import distutils
import litebird_sim as lbs
import logging as log
import numpy as np
import healpy as hp


class HwpSys:
    def __init__(self, simulation):
        self.sim = simulation
        self.imo = self.sim.imo

#        self.parameters = simulation.parameters['hwp_sys']

    def read_parameters(self):
        #booo=self.parameters['booo']	
        self.h1=0.0
        self.h2=0.0
        self.beta=0.0
        self.z1=0.0
        self.z2=0.0

    def built_tod(self,tod,pointings,times,hwp_radpsec,maps):
        #tod (ndet,nsamp)
        #pointings (ndet,nsamp,3). Last field: theta,phi,psi in radians
        #maps (ndet,3,npix)
        
        ndet=tod.shape[0]
        nside=hp.npix2nside(len(maps[0,0,:]))

        #is the loop over detectors really necessary?
        for idet in range(ndet):
            #add hwp rotation
            ca=np.cos(pointings[idet,:,2]+times*hwp_radpsec)
            sa=np.cos(pointings[idet,:,2]+times*hwp_radpsec)
            J11=(1+self.h1)*ca**2-(1+self.h2)*ca**2*np.exp(1j*self.beta)-(self.z1+self.z2)*ca*sa
            J12=((1+self.h1)+(1+self.h2)*np.exp(1j*self.beta))*sa*ca+self.z1*ca**2-self.z2*sa**2
            Tterm=0.5*(np.abs(J11)**2+np.abs(J12)**2)
            Qterm=0.5*(np.abs(J11)**2-np.abs(J12)**2)
            Uterm=(J11*J12.conjugate()).real
            pix=hp.ang2pix(nside,pointings[idet,:,0],pointings[idet,:,1])
            tod[idet,:]=Tterm*maps[idet,0,pix]+Qterm*maps[idet,1,pix]+Uterm*maps[idet,2,pix]
            


