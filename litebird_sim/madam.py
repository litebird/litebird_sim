# -*- encoding: utf-8 -*-

from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

from astropy.io import fits
import jinja2

from . import DetectorInfo
from .mapping import DestriperParameters
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

    table.writeto(
        str(file_name),
        overwrite=True,
    )


def save_simulation_for_madam(
    sim: Simulation,
    detectors: List[DetectorInfo],
    params: DestriperParameters,
    use_gzip: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    absolute_paths: bool = True,
):
    """
    Save the TODs and pointings of a simulation to files suitable to be read by Madam

    This function takes all the TOD samples and pointing angles from `sim` and saves
    them to the directory specified by `output_path` (the default is to save them
    in a sub-folder of the output path of the simulation). The parameter `detector`
    must be a list of :class:`.DetectorInfo` objects, and it specifies which detectors
    will be saved to disk. The variable `params` specifies how Madam should produce
    the maps; see the documentation for :class:`.DestriperParameters` for more
    information.

    If `use_gzip` is true, the TOD and pointing files will be compressed using Gzip
    (the default is false, as this might slow down I/O). If `absolute_paths` is ``True``
    (the default), the parameter and simulation files produced by this routine will
    be *absolute*; set it to `False` if you plan to move the FITS files to some other
    directory or computer before running Madam.
    """
    sim_template, par_template = _read_templates()

    if not output_path:
        madam_base_path = sim.base_path / "madam"
    else:
        if Path(output_path).is_absolute():
            madam_base_path = Path(output_path)
        else:
            madam_base_path = sim.base_path / output_path

    if absolute_paths:
        madam_base_path = madam_base_path.absolute()

    madam_detectors = []
    det_to_index = {}
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

        det_to_index[det.name] = det_id

    pointing_files = []
    tod_files = []
    num_of_files_per_detector = {}

    for obs in sim.observations:
        for det_idx, det_name in enumerate(obs.name):
            file_idx = num_of_files_per_detector.get("det_name", 0)
            pointing_file_name = f"pnt_{det_name}_{file_idx:05d}.fits"
            if use_gzip:
                pointing_file_name = pointing_file_name + ".gz"
            pointing_file_name = madam_base_path / pointing_file_name

            _save_pointings_to_fits(
                obs=obs, det_idx=det_idx, file_name=pointing_file_name
            )

            pointing_files.append(
                {
                    "file_name": pointing_file_name,
                    "det_name": det_name,
                    "det_id": det_to_index[det_name],
                }
            )

            tod_file_name = f"tod_{det_name}_{file_idx:05d}.fits"
            if use_gzip:
                tod_file_name = tod_file_name + ".gz"
            tod_file_name = madam_base_path / tod_file_name

            _save_tod_to_fits(obs=obs, det_idx=det_idx, file_name=tod_file_name)

            tod_files.append(
                {
                    "file_name": tod_file_name,
                    "det_name": det_name,
                    "det_id": det_to_index[det_name],
                }
            )

            num_of_files_per_detector[det_name] = file_idx + 1

    # Check that all the detectors have the same sampling rate
    sampling_rate_hz = detectors[0].sampling_rate_hz
    for i in range(1, len(detectors)):
        if detectors[i].sampling_rate_hz != sampling_rate_hz:
            raise ValueError(
                (
                    "All the detectors must have the same sampling frequency "
                    "({val1} Hz ≠ {val2} Hz for '{name1}' and '{name2}')"
                ).format(
                    val1=detectors[i].sampling_rate_hz,
                    val2=sampling_rate_hz,
                    name1=detectors[i].name,
                    name2=detectors[0].name,
                )
            )

    # Check that the number of files per detector is always the same
    number_of_files = num_of_files_per_detector[detectors[0].name]
    for det in detectors:
        if num_of_files_per_detector[det.name] != number_of_files:
            raise ValueError(
                (
                    "All the detectors must be split in the same number of "
                    "observations ({num1} ≠ {num2} for '{name1}' and '{name2}')"
                ).format(
                    num1=num_of_files_per_detector[det.name],
                    num2=number_of_files,
                    name1=det.name,
                    name2=detectors[0].name,
                )
            )

    simulation_file_path = madam_base_path / "madam.sim"
    parameter_file_path = madam_base_path / "madam.par"

    ensure_parent_dir_exists(simulation_file_path)
    ensure_parent_dir_exists(parameter_file_path)

    madam_maps_path = madam_base_path / "maps"
    madam_maps_path.mkdir(parents=True, exist_ok=True)

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
