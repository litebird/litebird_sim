# -*- encoding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

import jinja2
import numpy as np
from astropy.io import fits
from astropy.time import Time as AstroTime

import litebird_sim
from . import DetectorInfo
from .coordinates import CoordinateSystem
from .hwp import HWP
from .mapmaking import ExternalDestriperParameters
from .observations import Observation
from .pointings_in_obs import (
    _get_pointings_and_pol_angles_det,
)
from .simulations import Simulation, MpiDistributionDescr


def _read_templates():
    template_loader = jinja2.PackageLoader("litebird_sim", "templates")
    template_env = jinja2.Environment(
        loader=template_loader, trim_blocks=True, lstrip_blocks=True
    )
    sim_template = template_env.get_template("madam_simulation_file.txt")
    par_template = template_env.get_template("madam_parameter_file.txt")

    return (sim_template, par_template)


def ensure_parent_dir_exists(file_name: Union[str, Path]):
    parent = Path(file_name).parent
    parent.mkdir(parents=True, exist_ok=True)


def _format_time_for_fits(time: Union[float, AstroTime]) -> Union[float, str]:
    return time if isinstance(time, float) else str(time)


def _save_pointings_to_fits(
    observation: Observation,
    det_idx: int,
    hwp: Optional[HWP],
    pointings: Union[np.ndarray, List[np.ndarray], None],
    output_coordinate_system,
    file_name: Union[str, Path],
    pointing_dtype=np.float64,
):
    ensure_parent_dir_exists(file_name)

    pointings_det, pol_angle = _get_pointings_and_pol_angles_det(
        obs=observation,
        det_idx=det_idx,
        hwp=hwp,
        pointings=pointings,
        output_coordinate_system=output_coordinate_system,
        pointing_dtype=pointing_dtype,
    )

    theta_col = fits.Column(name="THETA", array=pointings_det[:, 0], format="E")
    phi_col = fits.Column(name="PHI", array=pointings_det[:, 1], format="E")
    psi_col = fits.Column(name="PSI", array=pol_angle, format="E")

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["DET_NAME"] = observation.name[det_idx]
    primary_hdu.header["DET_IDX"] = det_idx
    primary_hdu.header["COORD"] = (
        "ECLIPTIC"
        if output_coordinate_system == CoordinateSystem.Ecliptic
        else "GALACTIC"
    )
    primary_hdu.header["TIME0"] = _format_time_for_fits(observation.start_time)
    primary_hdu.header["MPI_RANK"] = litebird_sim.MPI_COMM_WORLD.rank
    primary_hdu.header["MPI_SIZE"] = litebird_sim.MPI_COMM_WORLD.size

    table = fits.BinTableHDU.from_columns([theta_col, phi_col, psi_col])

    fits.HDUList([primary_hdu, table]).writeto(
        str(file_name),
        overwrite=True,
    )


def _save_tod_to_fits(
    observations: Observation,
    det_idx: int,
    file_name: Union[str, Path],
    components: List[str],
):
    ensure_parent_dir_exists(file_name)

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["DET_NAME"] = observations.name[det_idx]
    primary_hdu.header["DET_IDX"] = det_idx
    primary_hdu.header["TIME0"] = _format_time_for_fits(observations.start_time)
    primary_hdu.header["MPI_RANK"] = litebird_sim.MPI_COMM_WORLD.rank
    primary_hdu.header["MPI_SIZE"] = litebird_sim.MPI_COMM_WORLD.size

    hdu_list = [primary_hdu]

    for cur_component in components:
        col = fits.Column(
            name="TOD",
            array=getattr(observations, cur_component)[det_idx, :],
            format="E",
        )
        cur_hdu = fits.BinTableHDU.from_columns([col])

        # We write "cur_component" twice
        cur_hdu.name = cur_component  # This is saved in EXTNAME, all in uppercase
        cur_hdu.header["COMP"] = cur_component  # Here the case is preserved
        hdu_list.append(cur_hdu)

    fits.HDUList(hdu_list).writeto(
        str(file_name),
        overwrite=True,
    )


@dataclass
class _ObsInMpiProcess:
    start_time: Union[float, AstroTime]
    mpi_rank: int
    obs_local_idx: int  # Index of the observation within the MPI process
    obs_global_idx: int = 0  # Index of the FITS file containing this observation


def _sort_obs_per_det(
    distribution: MpiDistributionDescr,
    detector: str,
    mpi_rank: int,
) -> List[_ObsInMpiProcess]:
    sorted_list = sorted(
        [
            _ObsInMpiProcess(
                start_time=cur_obs.start_time,
                mpi_rank=cur_mpi_proc.mpi_rank,
                obs_local_idx=obs_local_idx,
                obs_global_idx=0,  # We'll set this later, once the list is sorted
            )
            for cur_mpi_proc in distribution.mpi_processes
            for (obs_local_idx, cur_obs) in enumerate(cur_mpi_proc.observations)
            if detector in cur_obs.det_names
        ],
        key=lambda x: x.start_time,
    )

    # Now fill obs_global_idx
    for global_idx in range(len(sorted_list)):
        sorted_list[global_idx].obs_global_idx = global_idx

    # Finally, filter out all the observations that don't belong to this MPI process
    return [x for x in sorted_list if x.mpi_rank == mpi_rank]


def _combine_file_dictionaries(file_dictionaries):
    return sorted(
        [item for sublist in file_dictionaries for item in sublist],
        key=lambda x: x["file_name"],
    )


def save_simulation_for_madam(
    sim: Simulation,
    params: ExternalDestriperParameters,
    detectors: Optional[List[DetectorInfo]] = None,
    hwp: Optional[HWP] = None,
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    use_gzip: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    absolute_paths: bool = True,
    madam_subfolder_name: str = "madam",
    components: List[str] = ["tod"],
    components_to_bin: Optional[List[str]] = None,
    pointing_dtype=np.float64,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic,
    save_pointings: bool = True,
    save_tods: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Save the TODs and pointings of a simulation to files suitable to be read by Madam

    This function takes all the TOD samples and pointing angles from `sim` and saves
    them to the directory specified by `output_path` (the default is to save them
    in a sub-folder of the output path of the simulation). The parameter `detector`
    must be a list of :class:`.DetectorInfo` objects, and it specifies which detectors
    will be saved to disk; if it is ``None``, all the detectors in the simulation will
    be considered. The variable `params` specifies how Madam should produce the maps;
    see the documentation for :class:`.ExternalDestriperParameters` for more
    information.

    If `use_gzip` is true, the TOD and pointing files will be compressed using Gzip
    (the default is false, as this might slow down I/O). If `absolute_paths` is ``True``
    (the default), the parameter and simulation files produced by this routine will
    be *absolute*; set it to `False` if you plan to move the FITS files to some other
    directory or computer before running Madam.

    The parameter `madam_subfolder_name` is the name of the directory within the
    output folder of the simulation that will contain the Madam parameter files.

    You can use multiple TODs in the map-making process. By default, the code will
    only dump ``Observation.tod`` in the FITS files, but you can specify additional
    components via the `components` parameter, which is a list of the fields that
    must be saved in the FITS files and included in the parameter and simulation
    files. All these components will be summed in the map-making process.

    If you want to create a map using just a subset of the components listed in the
    `components` parameter, list them in `components_to_bin`. (This is usually
    employed when you pass ``save_pointings=False`` and ``save_tods=False``.) If
    `components_to_bin` is ``None`` (the default), all the elements in `components`
    will be used.

    If you are using MPI, call this function on *all* the MPI processes, not just on
    the one with rank #0.

    The flags `save_pointings` and `save_tods` are used to tell if you want pointings
    and TODs to be saved in FITS files or not. If either flag is set to ``false``, the
    corresponding FITS files will not be produced, but the ``.sim`` file for Madam
    will nevertheless list them as if they were created. This is useful if you plan to
    reuse files from some other call to ``save_simulation_for_madam``; in this case,
    a common trick is to create soft links to them in the output directory where the
    ``.par`` and ``.sim`` files are saved.

    `pointings` and `hwp` are the optional parameters. External pointing
    information, if not included in the observations, must be passed through
    the `pointings` parameter. It is assumed that the pointing information is available in ecliptic coordinates. The pointings are therefore as such. To
    save pointings in other coordinates, the parameter `output_coordinate_system`
    can be used. The HWP object should be passed to `hwp` parameter in order
    to compute the HWP angles.

    When pointings are computed on the fly, they are computed in double
    precision. It can be modified with the argument `pointing_dtype`.

    The return value is either a dictionary containing all the parameters used to
    fill Madam files (the parameter file and the simulation file) or ``None``;
    the dictionary is only returned for the MPI process with rank #0.
    """

    # All the code revolves around the result of the first call to
    # Simulation.describe_mpi_distribution(), which returns a
    # description of the way observations are spread among the
    # MPI processes. This is *vital* to build correct FITS files
    # for Madam, as the TODs of each detector must be saved in
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
    # their index starting from 2. All of this is complicated by the
    # fact that we must respect chronological order, i.e., we must ensure
    # that time increases as observations are saved. Suppose that we are
    # simulating just *one* detector, using 4 observations split between
    # two MPI processes:
    #
    # MPI#0: obs0 obs2
    # MPI#1: obs1 obs3
    #
    # where obs0, obs1, obs2, obs3 are in chronological order. Thus, each
    # MPI process cannot use a monotonically-increasing index!

    if not components_to_bin:
        components_to_bin = components

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

    # Build a dictionary containing the characteristics of each detector
    # to be written in the simulation file for Madam
    madam_detectors = []  # type:List[Dict[str, Any]]
    sorted_obs_per_det = []  # type: List[List[_ObsInMpiProcess]]
    for det_idx, det in enumerate(detectors):
        det_id = det_idx + 1
        madam_detectors.append(
            {
                "net_ukrts": det.net_ukrts,
                "slope": det.alpha,
                "fknee_hz": det.fknee_mhz / 1e3,
                "fmin_hz": det.fmin_hz,
                "name": det.name,
                "det_id": det_id + 1,
            }
        )
        sorted_obs_per_det.append(
            _sort_obs_per_det(
                distribution=distribution,
                detector=det.name,
                mpi_rank=litebird_sim.MPI_COMM_WORLD.rank,
            )
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
        # Rank #0 is special, because it must save the .sim and .par
        # files. Other ranks must just dump pointings and TOD into
        # FITS files.
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
    # look for the exact match with a linear search
    this_process_idx = [
        idx
        for idx, val in enumerate(distribution.mpi_processes)
        if val.mpi_rank == rank
    ]
    assert len(this_process_idx) == 1, (
        "more than one MPI rank matches Simulation.describe_mpi_distribution()"
    )
    this_process_idx = this_process_idx[0]

    pointing_files = []
    tod_files = []

    for cur_obs_idx, cur_obs in enumerate(sim.observations):
        # Note that "detectors" is the *global* list of detectors shared among all the
        # MPI processes
        for cur_global_det_idx, cur_detector in enumerate(detectors):
            cur_det_name = cur_detector.name
            if cur_det_name not in cur_obs.name:
                continue

            # This is the index of the detector *within the current observation*.
            # It's used to pick the right column in the pointing/tod matrices
            cur_local_det_idx = list(cur_obs.name).index(cur_det_name)

            # Retrieve the progressive number of the observation in the global
            # list of observations (i.e., among the MPI processes)
            matching_obs = [
                x
                for x in sorted_obs_per_det[cur_global_det_idx]
                if x.obs_local_idx == cur_obs_idx
            ]
            assert len(matching_obs) == 1, (
                "There is a bug in _sort_obs_per_det(), {} ≠ 1 observations".format(
                    len(matching_obs)
                )
            )
            file_idx = matching_obs[0].obs_global_idx

            pointing_file_name = f"pnt_{cur_det_name}_{file_idx:05d}.fits"
            if use_gzip:
                pointing_file_name = pointing_file_name + ".gz"
            pointing_file_name = madam_base_path / pointing_file_name

            if save_pointings:
                _save_pointings_to_fits(
                    observation=cur_obs,
                    det_idx=cur_local_det_idx,
                    hwp=hwp,
                    pointings=pointings,
                    output_coordinate_system=output_coordinate_system,
                    file_name=pointing_file_name,
                    pointing_dtype=pointing_dtype,
                )

            pointing_files.append(
                {
                    "file_name": pointing_file_name,
                    "det_name": cur_det_name,
                    "det_id": cur_global_det_idx + 1,
                }
            )

            tod_file_name = f"tod_{cur_det_name}_{file_idx:05d}.fits"
            if use_gzip:
                tod_file_name = tod_file_name + ".gz"
            tod_file_name = madam_base_path / tod_file_name

            if save_tods:
                _save_tod_to_fits(
                    observations=cur_obs,
                    det_idx=cur_local_det_idx,
                    file_name=tod_file_name,
                    components=components,
                )

            tod_files.append(
                {
                    "file_name": tod_file_name,
                    "det_name": cur_det_name,
                    "det_id": cur_global_det_idx + 1,
                    "components": components,
                }
            )

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
            "components_to_save": components,
            "components_to_bin": components_to_bin,
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
            "out_map_in_galactic": "T"
            if params.coordinate_system == CoordinateSystem.Galactic
            else "F",
        }

        with simulation_file_path.open("wt") as outf:
            outf.write(sim_template.render(**parameters))

        with parameter_file_path.open("wt") as outf:
            outf.write(par_template.render(**parameters))

        return parameters
    else:
        return None
