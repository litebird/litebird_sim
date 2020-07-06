import healpy as hp
import numpy as np
import argparse
import importlib.util
import os
import math
import lb_mbs.instrument

def print_rnk0(text, rank):
    if rank==0:
        print(text)

def from_sens_to_rms(sens, nside):
    rms = sens/hp.nside2resol(nside, arcmin=True)
    return rms

def make_noise_sims(params):
    """ Write noise maps on disk

    Parameters
    ----------
    params: module contating all the simulation parameters

    """
    instr = getattr(lb_mbs.instrument, params.inst)
    nmc_noise = params.nmc_noise
    nside = params.nside
    npix = hp.nside2npix(nside)
    root_dir = params.out_dir
    out_dir = f'{root_dir}/noise/'
    seed_noise = params.seed_noise
    N_split = params.N_split
    file_str = params.file_string
    channels = instr.keys()
    parallel = params.parallel
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
    if nmc_noise!=params.nmc_noise:
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
