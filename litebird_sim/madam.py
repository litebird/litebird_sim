# -*- encoding: utf-8 -*-

from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

from astropy.io import fits
import jinja2

import litebird_sim
from . import DetectorInfo
from .mapping import DestriperParameters
from .mpi import MPI_COMM_WORLD
from .observations import Observation
from .simulations import Simulation


def _read_templates():
    basepath = (Path(__file__).parent.parent / "templates").absolute()
    template_loader = jinja2.FileSystemLoader(searchpath=basepath)
    template_env = jinja2.Environment(
        loader=template_loader, trim_blocks=True, lstrip_blocks=True
    )
    sim_template = template_env.get_template("madam_simulation_file.txt")
    par_template = template_env.get_template("madam_parameter_file.txt")

    return (sim_template, par_template)


def ensure_parent_dir_exists(file_name: Union[str, Path]):
    parent = Path(file_name).parent
    parent.mkdir(parents=True, exist_ok=True)


def _save_pointings_to_fits(
    obs: Observation,
    det_idx: int,
    file_name: Union[str, Path],
):
    ensure_parent_dir_exists(file_name)

    theta_col = fits.Column(
        name="THETA", array=obs.pointings[det_idx, :, 0], format="E"
    )
    phi_col = fits.Column(name="PHI", array=obs.pointings[det_idx, :, 1], format="E")
    psi_col = fits.Column(name="PSI", array=obs.psi[det_idx, :], format="E")

    table = fits.BinTableHDU.from_columns([theta_col, phi_col, psi_col])

    table.header["DET_NAME"] = obs.name[det_idx]
    table.header["COORD"] = str(obs.pointing_coords)

    table.writeto(
        str(file_name),
        overwrite=True,
    )


def _save_tod_to_fits(
    obs: Observation,
    det_idx: int,
    file_name: Union[str, Path],
):
    ensure_parent_dir_exists(file_name)

    col = fits.Column(name="TOD", array=obs.tod[det_idx, :], format="E")

    table = fits.BinTableHDU.from_columns([col])

    table.header["DET_NAME"] = obs.name[det_idx]
    table.header["DET_IDX"] = det_idx
    table.header["TIME0"] = (
        obs.start_time if isinstance(obs.start_time, float) else str(obs.start_time)
    )
    table.header["MPIRANK"] = litebird_sim.MPI_COMM_WORLD.rank
    table.header["MPISIZE"] = litebird_sim.MPI_COMM_WORLD.size

    table.writeto(
        str(file_name),
        overwrite=True,
    )


def _combine_file_dictionaries(file_dictionaries):
    return sorted(
        [item for sublist in file_dictionaries for item in sublist],
        key=lambda x: x["file_name"],
    )


def save_simulation_for_madam(
    sim: Simulation,
    params: DestriperParameters,
    detectors: Optional[List[DetectorInfo]] = None,
    use_gzip: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    absolute_paths: bool = True,
    madam_subfolder_name: str = "madam",
) -> Optional[Dict[str, Any]]:
    """
    Save the TODs and pointings of a simulation to files suitable to be read by Madam

    This function takes all the TOD samples and pointing angles from `sim` and saves
    them to the directory specified by `output_path` (the default is to save them
    in a sub-folder of the output path of the simulation). The parameter `detector`
    must be a list of :class:`.DetectorInfo` objects, and it specifies which detectors
    will be saved to disk; if it is ``None``, all the detectors in the simulation will
    be considered. The variable `params` specifies how Madam should produce the maps;
    see the documentation for :class:`.DestriperParameters` for more information.

    If `use_gzip` is true, the TOD and pointing files will be compressed using Gzip
    (the default is false, as this might slow down I/O). If `absolute_paths` is ``True``
    (the default), the parameter and simulation files produced by this routine will
    be *absolute*; set it to `False` if you plan to move the FITS files to some other
    directory or computer before running Madam.

    The parameter `madam_subfolder_name` is the name of the directory within the
    output folder of the simulation that will contain the Madam parameter files.

    If you are using MPI, call this function on *all* the MPI processes, not just on
    the one with rank #0.

    The return value is either a dictionary containing all the parameters used to
    fill Madam files (the parameter file and the simulation file) or ``None``;
    the dictionary is only returned for the MPI process with rank #0.
    """

    # All the code revolves around the result of the first call to
    # Simulation.describe_mpi_distribution(), which returns a
    # description of the way observations are spread among the
    # MPI processes. This is *vital* to build correct FITS files
    # for Madam, as the TODs of each detectors must be saved in
    # files whose names contain an increasing integer index. We
    # use `distribution` to properly compute these indexes.
    #
    # Consider this (willingly convoluted) distribution of detectors
    # among observations, indicated with [], and MPI processes:
    #
    # MPI#1:   obs1:[A]  obs2:[B]  obs3:[A]  obs4:[A]
    # MPI#2:   obs1:[C]  obs2:[D]  obs3:[E]  obs4:[F]
    # MPI#3:   obs1:[A]  obs2:[D]  obs3:[C]  obs4:[F]
    # MPI#4:   obs1:[D]  obs2:[D]  obs3:[F]  obs4:[F]
    #
    # Forget the fact that the number of observations for detector A
    # is greater than for detector B (something that Madam would
    # reject), and concentrate on the task required to process MPI#4:
    # it only contains observations for detectors D and F, and it must
    # save them so that they do not interfere with files saved by other
    # MPI processes for the same detectors. Thus, it must know that obs2
    # in MPI#2 (index #0) and obs2 in MPI#3 (index #1) refer to detector
    # D, and thus the FITS files that MPI#4 will save for D will have
    # their index starting from 2.

    distribution = sim.describe_mpi_distribution()
    assert distribution is not None

    if detectors is not None:
        # Compute the intersection between the list of detectors passed as an argument
        # and the detectors that have been actually used in the simulation
        matching_names = list(
            set((x.name for x in detectors))
            & set((x.name for x in distribution.detectors))
        )

        # Filter out the mismatched detectors
        detectors = [x for x in detectors if x.name in matching_names]
    else:
        detectors = distribution.detectors

    rank = litebird_sim.MPI_COMM_WORLD.rank

    madam_detectors = []
    for idx, det in enumerate(detectors):
        det_id = idx + 1
        madam_detectors.append(
            {
                "net_ukrts": det.net_ukrts,
                "slope": det.alpha,
                "fknee_hz": det.fknee_mhz / 1e3,
                "fmin_hz": det.fmin_hz,
                "name": det.name,
                "det_id": det_id,
            }
        )

    if not output_path:
        madam_base_path = sim.base_path / madam_subfolder_name
    else:
        if Path(output_path).is_absolute():
            madam_base_path = Path(output_path)
        else:
            madam_base_path = sim.base_path / output_path

    if absolute_paths:
        madam_base_path = madam_base_path.absolute()

    if rank == 0:
        sim_template, par_template = _read_templates()

        simulation_file_path = madam_base_path / "madam.sim"
        parameter_file_path = madam_base_path / "madam.par"

        ensure_parent_dir_exists(simulation_file_path)
        ensure_parent_dir_exists(parameter_file_path)

        madam_maps_path = madam_base_path / "maps"
        madam_maps_path.mkdir(parents=True, exist_ok=True)
    else:
        sim_template, par_template = None, None
        simulation_file_path, parameter_file_path = None, None
        madam_maps_path = None

    # We might assume that our MPI process is described by
    # distribution.mpi_processes[rank], but here we relax the
    # requirement that the list be ordered by MPI rank and
    # look for the match with a linear search
    this_process_idx = [
        idx
        for idx, val in enumerate(distribution.mpi_processes)
        if val.mpi_rank == rank
    ]
    assert (
        len(this_process_idx) == 1
    ), "more than one MPI rank matches Simulation.describe_mpi_distribution()"
    this_process_idx = this_process_idx[0]

    pointing_files = []
    tod_files = []

    det_start_idx = {}  # type: Dict[str, idx]
    # Count how many observations in the MPI processes with rank smaller than ours
    # refer to the current detector: this is used to determine the index of each
    # FITS file
    for cur_mpi_proc in distribution.mpi_processes[0:this_process_idx]:
        for cur_obs in cur_mpi_proc.observations:
            for cur_det_name in cur_obs.det_names:
                det_start_idx[cur_det_name] = det_start_idx.get(cur_det_name, 0) + 1

    # Files per detector that have been written *by this MPI process*
    num_of_files_per_detector = {}  # type: Dict[str, int]

    for cur_obs in sim.observations:
        # Note that "detectors" is the *global* list of detectors shared among all the
        # MPI processes
        for cur_global_det_idx, cur_detector in enumerate(detectors):
            cur_det_name = cur_detector.name
            if cur_det_name not in cur_obs.name:
                continue

            # This is the index of the detector *within the current observation*.
            # It's used to pick the right column in the pointing/tod matrices
            cur_local_det_idx = list(cur_obs.name).index(cur_det_name)

            local_det_file_index = num_of_files_per_detector.get(cur_det_name, 0)
            file_idx = det_start_idx.get(cur_det_name, 0) + local_det_file_index
            pointing_file_name = f"pnt_{cur_det_name}_{file_idx:05d}.fits"
            if use_gzip:
                pointing_file_name = pointing_file_name + ".gz"
            pointing_file_name = madam_base_path / pointing_file_name

            _save_pointings_to_fits(
                obs=cur_obs, det_idx=cur_local_det_idx, file_name=pointing_file_name
            )

            pointing_files.append(
                {
                    "file_name": pointing_file_name,
                    "det_name": cur_det_name,
                    "det_id": cur_global_det_idx,
                }
            )

            tod_file_name = f"tod_{cur_det_name}_{file_idx:05d}.fits"
            if use_gzip:
                tod_file_name = tod_file_name + ".gz"
            tod_file_name = madam_base_path / tod_file_name

            _save_tod_to_fits(
                obs=cur_obs, det_idx=cur_local_det_idx, file_name=tod_file_name
            )

            tod_files.append(
                {
                    "file_name": tod_file_name,
                    "det_name": cur_det_name,
                    "det_id": cur_global_det_idx,
                }
            )

            num_of_files_per_detector[cur_det_name] = local_det_file_index + 1

    # To check how many files per detector have been created by any MPI process, just
    # count the ones that refer to the *first* detector
    first_det_name = detectors[0].name
    number_of_files = sum(
        [
            1
            for x in distribution.mpi_processes[this_process_idx].observations
            if first_det_name in x.det_names
        ]
    )
    if sim.mpi_comm and litebird_sim.MPI_ENABLED:
        number_of_files = litebird_sim.MPI_COMM_WORLD.allreduce(number_of_files)
        pointing_files = _combine_file_dictionaries(
            litebird_sim.MPI_COMM_WORLD.allgather(pointing_files),
        )
        tod_files = _combine_file_dictionaries(
            litebird_sim.MPI_COMM_WORLD.allgather(tod_files)
        )

    if rank == 0:
        sampling_rate_hz = detectors[0].sampling_rate_hz

        parameters = {
            "current_date": datetime.now(),
            "detectors": madam_detectors,
            "pointing_files": pointing_files,
            "tod_files": tod_files,
            "sampling_rate_hz": sampling_rate_hz,
            "number_of_files": number_of_files,
            "pointings_path": "",
            "tod_path": "",
            "nside": params.nside,
            "simulation_file_name": str(simulation_file_path),
            "parameter_file_name": str(parameter_file_path),
            "samples_per_baseline": int(params.baseline_length_s * sampling_rate_hz),
            "madam_output_path": madam_maps_path,
            "madam_destriped_file_name": "destriped.fits"
            if params.return_destriped_map
            else "",
            "madam_baseline_file_name": "baselines.fits",
            "madam_binned_file_name": "binned.fits" if params.return_binned_map else "",
            "madam_cov_file_name": "cov.fits" if params.return_npp else "",
            "madam_hit_file_name": "hits.fits" if params.return_hit_map else "",
            "iter_max": params.iter_max,
        }

        with simulation_file_path.open("wt") as outf:
            outf.write(sim_template.render(**parameters))

        with parameter_file_path.open("wt") as outf:
            outf.write(par_template.render(**parameters))

        return parameters
    else:
        return None
