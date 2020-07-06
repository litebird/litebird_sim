import toml
import distutils
import litebird_sim as lbs
import numpy as np
import healpy as hp
import os
import math
import pysm3
import pysm3.units as u

def print_rnk0(text, rank):
    if rank==0:
        print(text)

def from_sens_to_rms(sens, nside):
    rms = sens/hp.nside2resol(nside, arcmin=True)
    return rms

class Mbs:
    def __init__(self, config_file):
        self.config_file = config_file

    def check_and_fix_config(self):
        rank = 0  ###### NOTE THIS FOR NOW!!!!
        config = toml.load(self.config_file)
        config_mbs = config['map_based_sims']
        try:
            self.parallel = config_mbs['general']['parallel']
        except:
            self.parallel = False
            print_rnk0('WARNING: setting parallel = False', rank)
        assert config_mbs['instrument']
        self.imo_dir = config_mbs['instrument']['imo_directory']
        self.imo_version = config_mbs['instrument']['imo_version']
        assert config_mbs['maps']['nside']
        self.nside = config_mbs['maps']['nside']
        try:
            self.gaussian_smooth = config_mbs['maps']['gaussian_smooth']
        except:
            self.gaussian_smooth = False
            print_rnk0('WARNING: setting gaussian_smooth = False', rank)
        try:
            self.save_coadd = config_mbs['maps']['save_coadd']
        except:
            self.save_coadd = False
            print_rnk0('WARNING: setting save_coadd = False', rank)
        try:
            assert config_mbs['noise']
            self.make_noise = True
        except:
            self.make_noise = False
            print_rnk0('WARNING: setting make_noise=False', rank)
        if self.make_noise:
            assert config_mbs['noise']['nmc_noise']
            self.nmc_noise = config_mbs['noise']['nmc_noise']
            try:
                self.seed_noise = config_mbs['noise']['seed_noise']
            except: self.seed_noise = None
            try:
                self.N_split = config_mbs['noise']['N_split']
                if self.N_split==1:
                    self.N_split = False
            except: self.N_split = False
        try:
            assert config_mbs['cmb']
            self.make_cmb = True
        except:
            self.make_cmb = False
            print_rnk0('WARNING: setting make_cmb = False', rank)
        if self.make_cmb:
            try:
                self.cmb_ps_file = config_mbs['cmb']['cmb_ps_file']
            except:
                self.cmb_ps_file = False
            if self.cmb_ps_file == False:
                print_rnk0('WARNING: No CMB power spectrum defined, using the default one', rank)
                try:
                    self.cmb_r = config_mbs['cmb']['cmb_r']
                except:
                    self.cmb_r = 0
                    print_rnk0('WARNING: setting cmb_r=0', rank)
            assert config_mbs['cmb']['nmc_cmb']
            self.nmc_cmb = config_mbs['cmb']['nmc_cmb']
            try:
                self.seed_cmb = config_mbs['cmb']['seed_cmb']
            except: self.seed_cmb = None
        try:
            assert config_mbs['fg']
            self.make_fg = True
        except:
            self.make_fg = False
            print_rnk0('WARNING: setting make_fg = False', rank)
        if self.make_fg:
            self.fg_models = config_mbs['fg']
        try:
            self.out_dir = config_mbs['output']['out_directory']
        except:
            self.out_dir = './litebird_mbs'
            print_rnk0('WARNING: no output directory defined, results will be save in ./litebird_mbs', rank)
        try:
            self.file_string = config_mbs['output']['file_string']
        except:
            self.file_string = f'date_{date.today().strftime("%y%m%d")}'
            print_rnk0('WARNING: no file_string defined, today date is set as string', rank)


    def read_imo(self, directory=None, release=None):
        if directory:
            self.imo_dir = directory
        if release:
            self.imo_version = release
        imoflatfile = self.imo_dir
        imo = lbs.Imo(imoflatfile)
        instruments = ['LFT', 'MFT', 'HFT']
        self.LB_inst = {}
        for instr in instruments:
            channels = imo.query("/releases/"+self.imo_version+"/Satellite/"+instr+"/info").metadata['channel_names']
            for ch in channels:
                data_file = imo.query("/releases/"+self.imo_version+"/Satellite/"+instr+"/"+ch+"/info")
                freq = data_file.metadata['bandcenter']
                freq_band = data_file.metadata['bandwidth']
                fwhm_arcmin = data_file.metadata['fwhm_arcmin']
                P_sens = data_file.metadata['pol_sensitivity_channel_uKarcmin']
                self.LB_inst[ch] = {'freq':freq, 'freq_band': freq_band, 'beam': fwhm_arcmin, 'P_sens': P_sens }


    def make_noise_sims(self):
        instr = self.LB_inst
        nmc_noise = self.nmc_noise
        nside = self.nside
        npix = hp.nside2npix(nside)
        root_dir = self.out_dir
        out_dir = f'{root_dir}/noise/'
        seed_noise = self.seed_noise
        N_split = self.N_split
        file_str = self.file_string
        channels = instr.keys()
        parallel = self.parallel
        rank = 0
        size = 1
        if parallel:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        if not os.path.exists(out_dir) and rank==0:
                os.makedirs(out_dir)
        nmc_noise = math.ceil(nmc_noise/size)*size
        if nmc_noise!=self.nmc_noise:
            print_rnk0(f'WARNING: setting nmc_noise = {nmc_noise}', rank)
        perrank = nmc_noise//size
        chnl_seed = 12
        for chnl in channels:
                chnl_seed += 67
                P_sens = instr[chnl]['P_sens']
                P_rms = from_sens_to_rms(P_sens, nside)
                T_rms = P_rms/np.sqrt(2.)
                tot_rms = np.array([T_rms, P_rms, P_rms]).reshape(3,1)
                for nmc in range(rank*perrank, (rank+1)*perrank):
                    if seed_noise:
                        np.random.seed(seed_noise+nmc+chnl_seed)
                    nmc_str = str(nmc).zfill(4)
                    if not os.path.exists(out_dir+nmc_str):
                            os.makedirs(out_dir+nmc_str)
                    if N_split:
                        split_rms = tot_rms*np.sqrt(N_split)
                        noise_map = np.zeros((3, npix))
                        for hm in range(N_split):
                            noise_map_split = np.random.randn(3, npix)*split_rms
                            noise_map += noise_map_split
                            file_name = f'{chnl}_noise_SPLIT_{hm+1}of{N_split}_{nmc_str}_{file_str}.fits'
                            file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
                            hp.write_map(file_tot_path, noise_map_split, overwrite=True)
                        noise_map = noise_map/N_split
                    else:
                        noise_map = np.random.randn(3, npix)*tot_rms
                    file_name = f'{chnl}_noise_FULL_{nmc_str}_{file_str}.fits'
                    file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
                    hp.write_map(file_tot_path, noise_map, overwrite=True, dtype=np.float32)

    def make_cmb_sims(self):
        instr = self.LB_inst
        nmc_cmb = self.nmc_cmb
        nside = self.nside
        smooth = self.gaussian_smooth
        parallel = self.parallel
        root_dir = self.out_dir
        out_dir = f'{root_dir}/cmb/'
        file_str = self.file_string
        channels = instr.keys()
        seed_cmb = self.seed_cmb
        cmb_ps_file = self.cmb_ps_file
        rank = 0
        size = 1
        if parallel:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        if not os.path.exists(out_dir) and rank==0:
            os.makedirs(out_dir)
        if cmb_ps_file:
            cl_cmb = hp.read_cl(cmb_ps_file)
        else:
            cmb_ps_scalar_file = os.path.join(
                os.path.dirname(__file__),
                '../datautils/Cls_Planck2018_lensed_scalar.fits')
            cl_cmb_scalar = hp.read_cl(cmb_ps_scalar_file)
            cmb_ps_tensor_r1_file = os.path.join(
                os.path.dirname(__file__),
                '../datautils/Cls_only_tensor_r1.fits')
            cmb_r = self.cmb_r
            cl_cmb_tensor = hp.read_cl(cmb_ps_tensor_r1_file)*cmb_r
            cl_cmb = cl_cmb_scalar+cl_cmb_tensor
        nmc_cmb = math.ceil(nmc_cmb/size)*size
        if nmc_cmb!=self.nmc_cmb:
            print_rnk0(f'WARNING: setting nmc_cmb = {nmc_cmb}', rank)
        perrank = nmc_cmb//size
        for nmc in range(rank*perrank, (rank+1)*perrank):
            if seed_cmb:
                np.random.seed(seed_cmb+nmc)
            nmc_str = str(nmc).zfill(4)
            if not os.path.exists(out_dir+nmc_str):
                os.makedirs(out_dir+nmc_str)
            cmb_map = hp.synfast(cl_cmb, nside, new=True, verbose=False)
            for chnl in channels:
                fwhm = instr[chnl]['beam']
                if smooth:
                    cmb_map_smt = hp.smoothing(cmb_map, fwhm = np.radians(fwhm/60.), verbose=False)
                else:
                    cmb_map_smt = cmb_map
                file_name = f'{chnl}_cmb_{nmc_str}_{file_str}.fits'
                file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
                hp.write_map(file_tot_path, cmb_map_smt, overwrite=True, dtype=np.float32)

    def make_fg_sims(self):
        parallel = self.parallel
        instr = self.LB_inst
        nside = self.nside
        smooth = self.gaussian_smooth
        root_dir = self.out_dir
        out_dir = f'{root_dir}/foregrounds/'
        file_str = self.file_string
        channels = instr.keys()
        fg_models = self.fg_models
        components = list(fg_models.keys())
        ncomp = len(components)
        rank = 0
        if parallel:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            rank_to_use = list(range(ncomp))
        if not os.path.exists(out_dir) and rank==0:
            os.makedirs(out_dir)
        if rank==0:
            for cmp in components:
                if not os.path.exists(out_dir+cmp) and rank==0:
                    os.makedirs(out_dir+cmp)
                fg_config_file_name = fg_models[cmp]
                if ('lb' in fg_config_file_name) or ('pysm' in fg_config_file_name):
                    fg_config_file_path = os.path.join(
                        os.path.dirname(__file__), 'fg_models/')
                    fg_config_file = f'{fg_config_file_path}/{fg_config_file_name}'
                else:
                    fg_config_file = f'{fg_config_file_name}'
                sky = pysm3.Sky(nside=nside, component_config=fg_config_file)
                for chnl in channels:
                    freq = instr[chnl]['freq']
                    fwhm = instr[chnl]['beam']
                    sky_extrap = sky.get_emission(freq*u.GHz)
                    sky_extrap = sky_extrap.to(u.uK_CMB,
                        equivalencies=u.cmb_equivalencies(freq*u.GHz))
                    if smooth:
                        sky_extrap_smt = hp.smoothing(sky_extrap, fwhm = np.radians(fwhm/60.), verbose=False)
                    else:
                        sky_extrap_smt = sky_extrap
                    if rank==0:
                        file_name = f'{chnl}_{cmp}_{file_str}.fits'
                        file_tot_path = f'{out_dir}{cmp}/{file_name}'
                        hp.write_map(file_tot_path, sky_extrap_smt, overwrite=True, dtype=np.float32)

    def coadd_signal_maps(self):
        root_dir = self.out_dir
        fg_dir = f'{root_dir}/foregrounds/'
        cmb_dir = f'{root_dir}/cmb/'
        nside = self.nside
        file_str = self.file_string
        instr = self.LB_inst
        channels = instr.keys()
        coadd_dir = f'{root_dir}/coadd_signal_maps/'
        if not os.path.exists(coadd_dir):
            os.makedirs(coadd_dir)
        if os.path.exists(fg_dir):
            fg_models = self.fg_models
            components = list(fg_models.keys())
            for chnl in channels:
                fg_tot = np.zeros((3, hp.nside2npix(nside)))
                for cmp in components:
                    fg_dir_cmp = f'{fg_dir}{cmp}/'
                    fg_file_name = f'{chnl}_{cmp}_{file_str}.fits'
                    try:
                        fg_cmp = hp.read_map(f'{fg_dir_cmp}{fg_file_name}', (0,1,2), verbose=False)
                    except:
                        fg_cmp = hp.read_map(f'{fg_for_cmp}{file_name}', verbose=False)
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
                        hp.write_map(f'{coadd_dir}{nmc_str}/{tot_file_name}', map_tot, overwrite=True)
                else:
                    tot_file_name = f'{chnl}_coadd_signal_map_{file_str}.fits'
                    hp.write_map(f'{coadd_dir}/{tot_file_name}', map_tot, overwrite=True)

    def run_all(self):
        self.check_and_fix_config()
        self.read_imo()
        rank = 0
        if self.parallel:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        if self.make_noise:
            print_rnk0('generating noise simulations', rank)
            self.make_noise_sims()
        if self.make_cmb:
            print_rnk0('generating cmb simulations', rank)
            self.make_cmb_sims()
        if self.make_fg:
            print_rnk0('generating fg simulations', rank)
            self.make_fg_sims()
        if rank==0:
            #write_summary(params, par_file)
            if self.save_coadd:
                print_rnk0('saving coadded signal maps', rank)
                self.coadd_signal_maps()
