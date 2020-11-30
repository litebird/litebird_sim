import toml
import distutils
import litebird_sim as lbs
import logging as log
import numpy as np
import healpy as hp
import os
import math
import pysm3
import pysm3.units as u
from datetime import date


def from_sens_to_rms(sens, nside):
    rms = sens/hp.nside2resol(nside, arcmin=True)
    return rms

class Mbs:
    def __init__(self, simulation, instrument=None, detector_list=None, channel_list=None):
        self.sim = simulation
        self.imo = self.sim.imo
        self.parameters = simulation.parameters['map_based_sims']
        self.instrument = instrument
        self.det_list = detector_list
        self.ch_list = channel_list

    def read_instrument(self):
        if self.det_list:
            self.instrument = {}
            try:
                len(self.det_list)
            except TypeError:
                self.det_list = [self.det_list]
            for d in self.det_list:
                name = d.name
                name = name.replace(" ", "_")
                self.instrument[name] = {
                    'freq':d.bandcenter_ghz, 'freq_band':d.bandwidth_ghz,
                    'beam':d.fwhm_arcmin, 'P_sens':d.pol_sensitivity_ukarcmin}
        elif self.ch_list:
            self.instrument = {}
            try:
                len(self.ch_list)
            except TypeError:
                self.ch_list = [self.ch_list]
            for c in self.ch_list:
                name = c.channel
                name = name.replace(" ", "_")
                self.instrument[name] = {
                    'freq':c.bandcenter_ghz, 'freq_band':c.bandwidth_ghz,
                    'beam':c.fwhm_arcmin, 'P_sens':c.pol_sensitivity_channel_ukarcmin}
        elif self.instrument:
            log.info("using the passed instrument to generate maps")
        else:
            config_inst = self.parameters['instrument']
            custom_instrument = None
            if self.instrument==None:
                try:
                    custom_instrument = config_inst['custom_insturment']
                except KeyError:
                    self.imo_version = config_inst['IMO_version']
                    try:
                        self.telescopes = config_inst['telescopes']
                    except KeyError:
                        try:
                            self.channels = config_inst['channels']
                        except KeyError:
                                log.info("instrument dictonary should be pass to Mbs class")
                if custom_instrument:
                    if 'toml' in custom_instrument:
                        self.instrument = toml.load(custom_instrument)
                    elif 'npy' in custom_instrument:
                        self.instrument = np.load(custom_instrument, allow_picke=True).item()
                    else:
                        raise NameError('Wrong instrument dictonary format')
                else:
                    self.instrument = {}
                    if self.telescopes:
                        channels = []
                        for tel in self.telescopes:
                            channels.append(self.imo.query(
                                f'/releases/v{self.imo_version}/satellite/{tel}/instrument_info').metadata['channel_names'])
                        channels = [item for sublist in channels for item in sublist]
                    else:
                        channels = self.channels
                    for ch in channels:
                        if 'L' in ch:
                            tel = 'LFT'
                        elif 'M' in ch:
                            tel = 'MFT'
                        elif 'H' in ch:
                            tel = 'HFT'
                        data_file = self.imo.query(f'/releases/v{self.imo_version}/satellite/{tel}/{ch}/channel_info')
                        freq = data_file.metadata['bandcenter_ghz']
                        freq_band = data_file.metadata['bandwidth_ghz']
                        fwhm_arcmin = data_file.metadata['fwhm_arcmin']
                        P_sens = data_file.metadata['pol_sensitivity_channel_ukarcmin']
                        self.instrument[ch] = {'freq':freq, 'freq_band': freq_band, 'beam': fwhm_arcmin, 'P_sens': P_sens }

    def check_parameters(self):
        config_mbs = self.parameters
        self.nside = config_mbs['general']['nside']
        try: self.save = config_mbs['general']['save']
        except KeyError:
            self.save = True
            log.info("setting save = True")
        try: self.gaussian_smooth = config_mbs['general']['gaussian_smooth']
        except KeyError:
            self.gaussian_smooth = False
            log.info("setting gaussian_smooth = False")
        try: self.bandpass_int = config_mbs['general']['bandpass_int']
        except KeyError:
            self.bandpass_int = False
            log.info("setting bandpass_int = False")
        try: self.bandpass_int = config_mbs['general']['bandpass_int']
        except KeyError:
            self.bandpass_int = False
            log.info("setting bandpass_int = False")
        try: self.coadd = config_mbs['general']['coadd']
        except KeyError:
            if self.save==True:
                self.coadd = False
                log.info("setting coadd = False")
            else:
                self.coadd = True
        try: self.parallel_mc = config_mbs['general']['parallel_mc']
        except KeyError:
            self.parallel_mc = False
            log.info("setting parallel_mc = False")
        if (self.save==False) and (self.parallel_mc==True):
            self.parallel_mc = False
            log.info("setting parallel_mc = False as no MC will be made")
        self.units = 'muK_thermo'
        log.info("all maps will be in muK_thermo")
        try: self.make_noise = config_mbs['noise']['make_noise']
        except KeyError:
            self.make_noise = False
            log.info("Noise maps will NOT be generated")
        if self.make_noise:
            try: self.nmc_noise = config_mbs['noise']['nmc_noise']
            except KeyError:
                self.nmc_noise = 1
                log.info("setting nmc_noise=1")
            if (self.save==False) and (self.nmc_noise>1):
                self.nmc_noise = 1
                log.info("setting nmc_noise=1")
                log.info("MC simulations can be perfomed only if save=True")
            try: self.seed_noise = config_mbs['noise']['seed_noise']
            except KeyError: self.seed_noise = None
            try:
                self.N_split = config_mbs['noise']['N_split']
                if self.N_split==1:
                    self.N_split = False
            except KeyError:
                self.N_split = False
            if (self.save==False) and (self.N_split):
                self.N_split = False
                log.info("setting N_split=1")
                log.info("N_split can be > 1 only if save=True")
        try: self.make_cmb = config_mbs['cmb']['make_cmb']
        except KeyError:
            self.make_cmb = False
            log.info("CMB maps will NOT be generated")
        if self.make_cmb:
            try: self.cmb_ps_file = config_mbs['cmb']['cmb_ps_file']
            except KeyError: self.cmb_ps_file = False
            if self.cmb_ps_file == False:
                log.info("No CMB power spectrum defined, using the default one")
                try: self.cmb_r = config_mbs['cmb']['cmb_r']
                except KeyError:
                    self.cmb_r = 0
                    log.info("setting tensor-to-scalar ratio r=0")
            try: self.nmc_cmb = config_mbs['cmb']['nmc_cmb']
            except KeyError:
                self.nmc_cmb = 1
                log.info("setting nmc_cmb=1")
            if (self.save==False) and (self.nmc_cmb>1):
                self.nmc_cmb = 1
                log.info("setting nmc_cmb=1")
                log.info("MC simulations can be perfomed only if save=True")
            try: self.seed_cmb = config_mbs['cmb']['seed_cmb']
            except KeyError: self.seed_cmb = None
        try: self.make_fg = config_mbs['fg']['make_fg']
        except KeyError:
            self.make_fg = False
            log.info("Foreground maps will NOT be generated")
        if self.make_fg:
            self.fg_models = config_mbs['fg']
            del self.fg_models['make_fg']
            self.fg_components = list(self.fg_models.keys())
        try: self.out_dir = config_mbs['output']['output_directory']
        except KeyError: self.out_dir = './'
        log.info(f'saving files in {self.out_dir}')
        try: self.out_string = config_mbs['output']['output_string']
        except KeyError:
            self.out_string = f'date_{date.today().strftime("%y%m%d")}'
            log.info('no file_string defined, today date is set as string', rank)

    def generate_noise(self):
        instr = self.instrument
        nmc_noise = self.nmc_noise
        nside = self.nside
        npix = hp.nside2npix(nside)
        root_dir = self.out_dir
        out_dir = f'{root_dir}/noise/'
        seed_noise = self.seed_noise
        N_split = self.N_split
        file_str = self.out_string
        channels = instr.keys()
        N_channels = len(channels)
        parallel = self.parallel_mc
        col_units = [self.units, self.units, self.units]
        rank = 0
        size = 1
        if parallel:
            comm = lbs.MPI_COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        if not os.path.exists(out_dir) and rank==0 and self.save:
                os.makedirs(out_dir)
        nmc_noise = math.ceil(nmc_noise/size)*size
        if nmc_noise!=self.nmc_noise:
            log.info(f'WARNING: setting nmc_noise = {nmc_noise}', rank)
        perrank = nmc_noise//size
        chnl_seed = 12
        if not self.save:
            noise_map_matrix = np.zeros((N_channels, 3, npix))
        for Nchnl, chnl in enumerate(channels):
                chnl_seed += 67
                P_sens = instr[chnl]['P_sens']
                P_rms = from_sens_to_rms(P_sens, nside)
                T_rms = P_rms/np.sqrt(2.)
                tot_rms = np.array([T_rms, P_rms, P_rms]).reshape(3,1)
                for nmc in range(rank*perrank, (rank+1)*perrank):
                    if seed_noise:
                        np.random.seed(seed_noise+nmc+chnl_seed)
                    nmc_str = str(nmc).zfill(4)
                    if (self.save) and not (os.path.exists(out_dir+nmc_str)):
                            os.makedirs(out_dir+nmc_str)
                    if N_split:
                        split_rms = tot_rms*np.sqrt(N_split)
                        noise_map = np.zeros((3, npix))
                        for hm in range(N_split):
                            noise_map_split = np.random.randn(3, npix)*split_rms
                            noise_map += noise_map_split
                            file_name = f'{chnl}_noise_SPLIT_{hm+1}of{N_split}_{nmc_str}_{file_str}.fits'
                            file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
                            lbs.write_healpix_map_to_file(file_tot_path, noise_map_split, column_units=col_units)
                        noise_map = noise_map/N_split
                    else:
                        noise_map = np.random.randn(3, npix)*tot_rms
                    if self.save:
                        file_name = f'{chnl}_noise_FULL_{nmc_str}_{file_str}.fits'
                        file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
                        lbs.write_healpix_map_to_file(file_tot_path, noise_map, column_units=col_units)
                    else:
                        noise_map_matrix[Nchnl] = noise_map
        if not self.save:
            return noise_map_matrix
        else:
            return None

    def generate_cmb(self):
        instr = self.instrument
        nmc_cmb = self.nmc_cmb
        nside = self.nside
        npix = hp.nside2npix(nside)
        smooth = self.gaussian_smooth
        parallel = self.parallel_mc
        root_dir = self.out_dir
        out_dir = f'{root_dir}/cmb/'
        file_str = self.out_string
        channels = instr.keys()
        N_channels = len(channels)
        seed_cmb = self.seed_cmb
        cmb_ps_file = self.cmb_ps_file
        col_units = [self.units, self.units, self.units]
        rank = 0
        size = 1
        if parallel:
            comm = lbs.MPI_COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        if not os.path.exists(out_dir) and rank==0:
            os.makedirs(out_dir)
        if cmb_ps_file:
            cl_cmb = hp.read_cl(cmb_ps_file)
        else:
            cmb_ps_scalar_file = os.path.join(
                os.path.dirname(__file__),
                '../datautils/Cls_Planck2018_for_PTEP_2020_r0.fits')
            cl_cmb_scalar = hp.read_cl(cmb_ps_scalar_file)
            cmb_ps_tensor_r1_file = os.path.join(
                os.path.dirname(__file__),
                '../datautils/Cls_Planck2018_for_PTEP_2020_tensor_r1.fits')
            cmb_r = self.cmb_r
            cl_cmb_tensor = hp.read_cl(cmb_ps_tensor_r1_file)*cmb_r
            cl_cmb = cl_cmb_scalar+cl_cmb_tensor
        nmc_cmb = math.ceil(nmc_cmb/size)*size
        if nmc_cmb!=self.nmc_cmb:
            log.info(f'setting nmc_cmb = {nmc_cmb}', rank)
        perrank = nmc_cmb//size
        if not self.save:
            cmb_map_matrix = np.zeros((N_channels, 3, npix))
        for nmc in range(rank*perrank, (rank+1)*perrank):
            if seed_cmb:
                np.random.seed(seed_cmb+nmc)
            nmc_str = str(nmc).zfill(4)
            if not os.path.exists(out_dir+nmc_str):
                os.makedirs(out_dir+nmc_str)
            cmb_temp = hp.synfast(cl_cmb, nside, new=True, verbose=False)
            file_name = f'cmb_{nmc_str}_{file_str}.fits'
            file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
            lbs.write_healpix_map_to_file(file_tot_path, cmb_temp, column_units=col_units)
            os.environ["PYSM_LOCAL_DATA"] = f'{out_dir}'
            sky = pysm3.Sky(nside=nside, component_objects=[pysm3.CMBMap(nside, map_IQU=f'{nmc_str}/{file_name}')])
            for Nchnl, chnl in enumerate(channels):
                freq = instr[chnl]['freq']
                if self.bandpass_int:
                    band = instr[chnl]['freq_band']
                    fmin = freq-band/2.
                    fmax = freq+band/2.
                    fsteps = np.int(np.ceil(fmax-fmin)+1)
                    bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                    weights = np.ones(len(bandpass_frequencies))
                    cmb_map = sky.get_emission(bandpass_frequencies, weights)
                    cmb_map = cmb_map*pysm3.bandpass_unit_conversion(bandpass_frequencies, weights, u.uK_CMB)
                else:
                    cmb_map = sky.get_emission(freq*u.GHz)
                    cmb_map = cmb_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
                fwhm = instr[chnl]['beam']
                if smooth:
                    cmb_map_smt = hp.smoothing(cmb_map, fwhm = np.radians(fwhm/60.), verbose=False)
                else:
                    cmb_map_smt = cmb_map
                if self.save:
                    file_name = f'{chnl}_cmb_{nmc_str}_{file_str}.fits'
                    file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
                    lbs.write_healpix_map_to_file(file_tot_path, cmb_map_smt, column_units=col_units)
                else:
                    cmb_map_matrix[Nchnl] = cmb_map_smt
        if not self.save:
            return cmb_map_matrix
        else:
            return None


    def generate_fg(self):
        parallel = self.parallel_mc
        instr = self.instrument
        nside = self.nside
        npix = hp.nside2npix(nside)
        smooth = self.gaussian_smooth
        root_dir = self.out_dir
        out_dir = f'{root_dir}/foregrounds/'
        file_str = self.out_string
        channels = instr.keys()
        N_channels = len(channels)
        fg_models = self.fg_models
        components = self.fg_components
        ncomp = len(components)
        rank = 0
        col_units = [self.units, self.units, self.units]
        if parallel:
            comm = lbs.MPI_COMM_WORLD
            rank = comm.Get_rank()
        if not os.path.exists(out_dir) and rank==0 and self.save:
            os.makedirs(out_dir)
        if rank==0:
            if not self.save:
                dict_fg = {}
            for cmp in components:
                if not os.path.exists(out_dir+cmp) and rank==0 and self.save:
                    os.makedirs(out_dir+cmp)
                fg_config_file_name = fg_models[cmp]
                if ('lb' in fg_config_file_name) or ('pysm' in fg_config_file_name):
                    fg_config_file_path = os.path.join(
                        os.path.dirname(__file__), 'fg_models/')
                    fg_config_file = f'{fg_config_file_path}/{fg_config_file_name}.cfg'
                else:
                    fg_config_file = f'{fg_config_file_name}'
                sky = pysm3.Sky(nside=nside, component_config=fg_config_file)
                if not self.save:
                    fg_map_matrix = np.zeros((N_channels, 3, npix))
                for Nchnl, chnl in enumerate(channels):
                    freq = instr[chnl]['freq']
                    fwhm = instr[chnl]['beam']
                    if self.bandpass_int:
                        band = instr[chnl]['freq_band']
                        fmin = freq-band/2.
                        fmax = freq+band/2.
                        fsteps = np.int(np.ceil(fmax-fmin)+1)
                        bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                        weights = np.ones(len(bandpass_frequencies))
                        sky_extrap = sky.get_emission(bandpass_frequencies, weights)
                        sky_extrap = sky_extrap*pysm3.bandpass_unit_conversion(bandpass_frequencies, weights, u.uK_CMB)
                    else:
                        sky_extrap = sky.get_emission(freq*u.GHz)
                        sky_extrap = sky_extrap.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
                    if smooth:
                        sky_extrap_smt = hp.smoothing(sky_extrap, fwhm = np.radians(fwhm/60.), verbose=False)
                    else:
                        sky_extrap_smt = sky_extrap
                    if self.save:
                        file_name = f'{chnl}_{cmp}_{file_str}.fits'
                        file_tot_path = f'{out_dir}{cmp}/{file_name}'
                        lbs.write_healpix_map_to_file(file_tot_path, sky_extrap_smt, column_units=col_units)
                    else:
                        fg_map_matrix[Nchnl] = sky_extrap_smt
                if not self.save:
                    dict_fg[cmp] = fg_map_matrix
        if not self.save:
            return dict_fg
        else:
            return None

    def write_coadded_maps(self):
        root_dir = self.out_dir
        fg_dir = f'{root_dir}/foregrounds/'
        cmb_dir = f'{root_dir}/cmb/'
        nside = self.nside
        file_str = self.out_string
        instr = self.instrument
        channels = instr.keys()
        coadd_dir = f'{root_dir}/coadd_signal_maps/'
        col_units = [self.units, self.units, self.units]
        if not os.path.exists(coadd_dir):
            os.makedirs(coadd_dir)
        if os.path.exists(fg_dir):
            fg_models = self.fg_models
            components = self.fg_components
            for chnl in channels:
                fg_tot = np.zeros((3, hp.nside2npix(nside)))
                for cmp in components:
                    fg_dir_cmp = f'{fg_dir}{cmp}/'
                    fg_file_name = f'{chnl}_{cmp}_{file_str}.fits'
                    try:
                        fg_cmp = hp.read_map(f'{fg_dir_cmp}{fg_file_name}', (0,1,2), verbose=False)
                    except IndexError:
                        fg_cmp = hp.read_map(f'{fg_dir_cmp}{fg_file_name}', verbose=False)
                        fg_cmp = np.array([fg_cmp, fg_cmp*0., fg_cmp*0.])
                    fg_tot += fg_cmp
                if os.path.exists(cmb_dir):
                    nmc_cmb = self.nmc_cmb
                    for nmc in range(nmc_cmb):
                        nmc_str = str(nmc).zfill(4)
                        cmb_file_name = f'{chnl}_cmb_{nmc_str}_{file_str}.fits'
                        cmb = hp.read_map(f'{cmb_dir}{nmc_str}/{cmb_file_name}', (0,1,2), verbose=False)
                        map_tot = fg_tot+cmb
                        if not os.path.exists(f'{coadd_dir}{nmc_str}'):
                            os.makedirs(f'{coadd_dir}{nmc_str}')
                        tot_file_name = f'{chnl}_coadd_signal_map_{nmc_str}_{file_str}.fits'
                        lbs.write_healpix_map_to_file(f'{coadd_dir}{nmc_str}/{tot_file_name}', map_tot, column_units=col_units)
                else:
                    tot_file_name = f'{chnl}_coadd_signal_map_{file_str}.fits'
                    lbs.write_healpix_map_to_file(f'{coadd_dir}/{tot_file_name}', map_tot, column_units=col_units)

    def run_all(self):
        self.read_instrument()
        self.check_parameters()
        rank = 0
        instr = self.instrument
        nside = self.nside
        npix = hp.nside2npix(nside)
        channels = instr.keys()
        N_channels = len(channels)
        if not self.save:
            tot = np.zeros((N_channels, 3, npix))
        if self.parallel_mc:
            comm = lbs.MPI_COMM_WORLD
            rank = comm.Get_rank()
        if self.make_noise:
            log.info('generating and saving noise simulations')
            noise = self.generate_noise()
            if not self.save:
                tot = tot+noise
        if self.make_cmb:
            log.info('generating and saving cmb simulations')
            cmb = self.generate_cmb()
            if not self.save:
                tot = tot+cmb
        if self.make_fg:
            log.info('generating and saving fg simulations')
            fg = self.generate_fg()
            if not self.save:
                for cmp in fg.keys():
                    tot = tot+fg[cmp]
        if rank==0:
            if self.save and self.coadd:
                log.info('saving coadded signal maps')
                self.write_coadded_maps()
            if not self.save:
                tot_dict = {}
                for nch, chnl in enumerate(channels):
                    tot_dict[chnl] =  tot[nch]
                return tot_dict
            else:
                return None
