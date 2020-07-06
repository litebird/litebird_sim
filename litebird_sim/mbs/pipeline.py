import healpy as hp
import numpy as np
import argparse
import importlib.util
import lb_mbs.noise
import lb_mbs.cmb
import lb_mbs.foregrounds
import lb_mbs.instrument
from datetime import date, datetime
import os

def print_rnk0(text, rank):
    if rank==0:
        print(text)

def import_config_file():
    """ Get the configuration file and return the parameter module

    Returns
    -------
    module
        module with all the simulation parameters
    """
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--par_file',
                        dest='par_file',
                        default='',
                        help='path to parameter file')
    args = parser.parse_args()
    params_spec = importlib.util.spec_from_file_location('params', args.par_file)
    params = importlib.util.module_from_spec(params_spec)
    params_spec.loader.exec_module(params)
    return params, args.par_file

def check_and_fix_config_file(params):
    """ Check simulation parameters and set default ones

    Parameters
    ----------
    params: module contating all the simulation parameters

    Returns
    -------
    module
        module with all the simulation parameters

    """
    try: params.parallel
    except: params.parallel = False
    rank = 0
    if params.parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    try: params.inst
    except:
        params.inst = 'latest'
        print_rnk0('WARNING: no instrument defined, using the LB latest one', rank)
    assert params.nside
    try: params.gaussian_smooth
    except:
        params.gaussian_smooth = True
        print_rnk0('WARNING: setting gaussian_smooth=True', rank)
    try: params.save_coadd
    except:
        params.save_coadd = False
        print_rnk0('WARNING: setting save_coadd=False', rank)
    try: params.make_noise
    except:
        params.make_noise = False
        print_rnk0('WARNING: setting make_noise=False', rank)
    if params.make_noise:
        assert params.nmc_noise
        try: params.seed_noise
        except: params.seed_noise = None
        try:
            params.N_split
            if params.N_split==1:
                params.N_split = None
        except: params.N_split = False
    try: params.make_cmb
    except:
        params.make_cmb = False
        print_rnk0('WARNING: setting make_cmb=False', rank)
    if params.make_cmb:
        try: params.cmb_ps_file
        except:
            params.cmb_ps_file = False
        if params.cmb_ps_file == False:
            print_rnk0('WARNING: No CMB power spectrum defined, using the default one', rank)
            try: params.cmb_r
            except:
                params.cmb_r = 0
                print_rnk0('WARNING: setting cmb_r=0', rank)
            assert params.nmc_cmb
            try: params.seed_cmb
            except: params.seed_cmb = None
    try: params.make_fg
    except:
        params.make_fg = True
        print_rnk0('WARNING: setting make_fg=True', rank)
    assert params.fg_models
    assert type(params.fg_models)==dict, 'fg_models must be a python dictonary'
    try: params.out_dir
    except:
        params.out_dir = './litebird_mbs'
        print_rnk0('WARNING: no output directory defined, results will be save in ./litebird_mbs', rank)
    try: params.file_string
    except:
        params.file_string = f'date_{date.today().strftime("%y%m%d")}'
        print_rnk0('WARNING: no file_string defined, today date is set as string', rank)
    return params

def coadd_signal_maps(params):
    """ coadd togheter all the signal maps and save them on disk

    Parameters
    ----------
    params: module contating all the simulation parameters

    """
    root_dir = params.out_dir
    fg_dir = f'{root_dir}/foregrounds/'
    cmb_dir = f'{root_dir}/cmb/'
    nside = params.nside
    file_str = params.file_string
    instr = getattr(lb_mbs.instrument, params.inst)
    channels = instr.keys()
    coadd_dir = f'{root_dir}/coadd_signal_maps/'
    if not os.path.exists(coadd_dir):
        os.makedirs(coadd_dir)
    if os.path.exists(fg_dir):
        fg_models = params.fg_models
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
                nmc_cmb = params.nmc_cmb
                for nmc in range(nmc_cmb):
                    nmc_str = str(nmc).zfill(4)
                    cmb_file_name = f'{chnl}_cmb_{nmc_str}_{file_str}.fits'
                    cmb = hp.read_map(f'{cmb_dir}{nmc_str}/{cmb_file_name}', (0,1,2), verbose=False)
                    map_tot = fg_tot+cmb
                    if not os.path.exists(f'{coadd_dir}{nmc_str}'):
                        os.makedirs(f'{coadd_dir}{nmc_str}')
                    tot_file_name = f'{chnl}_coadd_signal_map_{nmc_str}_{file_str}.fits'
                    hp.write_map(f'{coadd_dir}{nmc_str}/{tot_file_name}', map_tot)
            else:
                tot_file_name = f'{chnl}_coadd_signal_map_{file_str}.fits'
                hp.write_map(f'{coadd_dir}/{tot_file_name}', map_tot)


def write_summary(params, par_file):
    """ Write the simulation summary file

    Parameters
    ----------
    params: module contating all the simulation parameters

    par_file: configuration file used to run the simulation

    """

    out_dir = params.out_dir
    file_string = params.file_string
    summary_string = par_file.replace('config', 'summary')
    summary_string = summary_string.replace('.py', '.txt')
    summary_file = f'{out_dir}/{summary_string}'
    f = open(summary_file, 'w+')
    f.write('This file has been generated automatically by the litebird_mbs script\n')
    f.write(f'Date: {datetime.utcnow()} UTC\n')
    f.write(f'----------------------------------\n')
    f.write('\n')
    f.write('The following parameter file has been used:\n')
    f.write('\n')
    fpar = open(par_file, 'r')
    f.write(fpar.read())
    fpar.close()
    f.write(f'----------------------------------\n')
    f.write('\n')
    if params.make_fg:
        f.write('Foregrounds map have been generated with the pysm3 library, ')
        f.write('considering the following models:\n')
        f.write('\n')
        fg_models = params.fg_models
        components = list(fg_models.keys())
        for cmp in components:
            fg_config_file_name = fg_models[cmp]
            if ('lb' in fg_config_file_name) or ('pysm' in fg_config_file_name):
                fg_config_file_path = os.path.join(
                    os.path.dirname(__file__), 'fg_models/')
                fg_config_file = f'{fg_config_file_path}/{fg_config_file_name}'
            else:
                fg_config_file = f'{fg_config_file_name}'
            f.write(f'{cmp.upper()}\n')
            fcmp= open(fg_config_file, 'r')
            f.write(fcmp.read())
            fpar.close()
            f.write(f'-------------\n')
            f.write('\n')
    f.close()

def __main__():
    """ Run the lb_mbs pipeline

    """

    params, par_file = import_config_file()
    params = check_and_fix_config_file(params)
    rank = 0
    if params.parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    if params.make_noise:
        print_rnk0('generating noise simulations', rank)
        lb_mbs.noise.make_noise_sims(params)
    if params.make_cmb:
        print_rnk0('generating cmb simulations', rank)
        lb_mbs.cmb.make_cmb_sims(params)
    if params.make_fg:
        print_rnk0('generating fg simulations', rank)
        lb_mbs.foregrounds.make_fg_sims(params)
    if rank==0:
        write_summary(params, par_file)
        if params.save_coadd:
            print_rnk0('saving coadded signal maps', rank)
            coadd_signal_maps(params)
