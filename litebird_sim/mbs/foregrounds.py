import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u
import argparse
import importlib.util
import os
import lb_mbs.instrument

def make_fg_sims(params):
    """ Write foreground maps on disk

    Parameters
    ----------
    params: module contating all the simulation parameters

    """
    parallel = params.parallel
    instr = getattr(lb_mbs.instrument, params.inst)
    nside = params.nside
    smooth = params.gaussian_smooth
    root_dir = params.out_dir
    out_dir = f'{root_dir}/foregrounds/'
    file_str = params.file_string
    channels = instr.keys()
    fg_models = params.fg_models
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
