# -*- encoding: utf-8 -*-

import codecs
from collections import namedtuple
from dataclasses import asdict, dataclass
from datetime import datetime
from deprecation import deprecated
import logging as log
import os
import subprocess
from typing import List, Tuple, Union, Dict, Any, Optional
from pathlib import Path
from shutil import copyfile, copytree, SameFileError

import litebird_sim
from litebird_sim import constants as c
from . import HWP
from .detectors import DetectorInfo, InstrumentInfo
from .distribute import distribute_evenly, distribute_optimally
from .healpix import write_healpix_map_to_file, npix_to_nside
from .imo.imo import Imo
from .mpi import MPI_COMM_WORLD
from .observations import Observation, TodDescription
from .pointings import get_pointings
from .version import (
    __version__ as litebird_sim_version,
    __author__ as litebird_sim_author,
)

from .dipole import DipoleType, add_dipole_to_observations
from .scan_map import scan_map_in_observations
from .spacecraft import SpacecraftOrbit, spacecraft_pos_and_vel
from .noise import add_noise_to_observations
from .mapping import make_bin_map
from .gaindrifts import GainDriftType, GainDriftParams, apply_gaindrift_to_observations

import astropy.time
import astropy.units
import markdown
import numpy as np
import jinja2
import tomlkit

from markdown_katex import KatexExtension

from .scanning import ScanningStrategy, SpinningScanningStrategy


DEFAULT_BASE_IMO_URL = "https://litebirdimo.ssdc.asi.it"

OutputFileRecord = namedtuple("OutputFileRecord", ["path", "description"])


def _tomlkit_to_popo(d):
    from datetime import date, time, datetime

    # This is a fix to issue
    # https://github.com/sdispater/tomlkit/issues/43. It converts an
    # object returned by tomlkit into a list of Plain Old Python
    # Objects (POPOs).
    try:
        # Tomlkit's dictionaries, booleans, dates have a "value" field
        # that returns a POPO
        result = getattr(d, "value")
    except AttributeError:
        result = d

    if isinstance(result, list):
        result = [_tomlkit_to_popo(x) for x in result]
    elif isinstance(result, dict):
        result = {
            _tomlkit_to_popo(key): _tomlkit_to_popo(val) for key, val in result.items()
        }
    elif isinstance(result, tomlkit.items.DateTime):
        result = datetime(
            result.year,
            result.month,
            result.day,
            result.hour,
            result.minute,
            result.second,
            tzinfo=result.tzinfo,
        )
    elif isinstance(result, tomlkit.items.Date):
        result = date(result.year, result.month, result.day)
    elif isinstance(result, tomlkit.items.Time):
        result = time(result.hour, result.minute, result.second)
    elif isinstance(result, tomlkit.items.Integer):
        result = int(result)
    elif isinstance(result, tomlkit.items.Float):
        result = float(result)
    elif isinstance(result, tomlkit.items.String):
        result = str(result)
    elif isinstance(result, tomlkit.items.Bool):
        result = bool(result)

    return result


def get_template_file_path(filename: Union[str, Path]) -> Path:
    """Return a Path object pointing to the full path of a template file.

    Template files are used by the framework to produce automatic
    reports. They are produced using template files, which usually
    reside in the ``templates`` subfolder of the main repository.

    Given a filename (e.g., ``report_header.md``), this function
    returns a full, absolute path to the file within the ``templates``
    folder of the ``litebird_sim`` source code.
    """
    return Path(__file__).parent / ".." / "templates" / filename


@dataclass
class MpiObservationDescr:
    """
    This class is used within :class:`.MpiProcessDescr`. It describes the
    kind and size of the data held by a :class:`.Observation` object.

    Its fields are:

    - `det_names` (list of ``str``): names of the detectors handled by
      this observation
    - `tod_names` (list of ``str``): names of the fields containing the TODs
      (e.g., ``tod``, ``cmb_tod``, ``dipole_tod``, …)
    - `tod_shape` (tuples of ``int``): shape of each TOD held by the observation.
      This is *not* a list, because all the TODs are assumed to have the same shape
    - `tod_dtype` (list of ``str``): string representing the NumPy data type of each
      TODs, in the same order as in the field `tod_name`
    - `tod_description` (list of ``str``): list of human-readable descriptions
      for each TOD, in the same order as in the field `tod_name`
    - `start_time` (either a ``float`` or a ``astropy.time.Time``): start date
      of the observation
    - `duration_s` (``float``): duration of the TOD in seconds
    - `num_of_samples` (``int``): number of samples held by this TOD
    - `num_of_detectors` (``int``): number of detectors held by this TOD. It's
      the length of the field `det_names` (see above)
    """

    det_names: List[str]
    tod_names: List[str]
    tod_shape: Optional[Tuple[int, int]]
    tod_dtype: List[str]
    tod_description: List[str]
    start_time: Union[float, astropy.time.Time]
    duration_s: float
    num_of_samples: int
    num_of_detectors: int


@dataclass
class MpiProcessDescr:
    """
    Description of the kind of data held by a MPI process

    This class is used within :class:`MpiDistributionDescr`. Its fields are:

    - `mpi_rank`: rank of the MPI process described by this instance
    - `observations`: list of :class:`.MpiObservationDescr` objects, each
      describing one observation managed by the MPI process with rank
      `mpi_rank`.

    """

    mpi_rank: int
    observations: List[MpiObservationDescr]


@dataclass
class MpiDistributionDescr:
    """A class that describes how observations are distributed among MPI processes

    The fields defined in this dataclass are the following:

    - `num_of_observations` (int): overall number of observations in *all* the
      MPI processes
    - `detectors` (list of :class:`.DetectorInfo` objects): list of *all* the
      detectors used in the observations
    - `mpi_processes`: list of :class:`.MpiProcessDescr` instances, describing
      the kind of data that each MPI process is currently holding

    Use :meth:`.Simulation.describe_mpi_distribution` to get an instance of this
    object."""

    num_of_observations: int
    detectors: List[DetectorInfo]
    mpi_processes: List[MpiProcessDescr]

    def __repr__(self):
        result = ""
        for cur_mpi_proc in self.mpi_processes:
            result += f"# MPI rank #{cur_mpi_proc.mpi_rank + 1}\n\n"
            for cur_obs_idx, cur_obs in enumerate(cur_mpi_proc.observations):
                result += """## Observation #{obs_idx}
- Start time: {start_time}
- Duration: {duration_s} s
- {num_of_detectors} detector(s) ({det_names})
- TOD(s): {tod_names}
- TOD shape: {tod_shape}
- Type of the TODs: {tod_dtype}

""".format(
                    obs_idx=cur_obs_idx,
                    start_time=cur_obs.start_time,
                    duration_s=cur_obs.duration_s,
                    num_of_detectors=len(cur_obs.det_names),
                    det_names=",".join(cur_obs.det_names),
                    tod_names=", ".join(cur_obs.tod_names),
                    tod_shape="×".join([str(x) for x in cur_obs.tod_shape]),
                    tod_dtype=", ".join(cur_obs.tod_dtype),
                )

        return result


class Simulation:
    """A container object for running simulations

    This is the most important class in the Litebird_sim framework. It
    initializes an output directory that will contain all the products
    of a simulation and will handle the generation of reports and
    writing of output files.

    Be sure to call :py:meth:`Simulation.flush` when the simulation is
    completed. This ensures that all the information are saved to disk
    before the completion of your script.

    You can access the fields `base_path`, `name`, `mpi_comm`, and
    `description` in the `Simulation` object::

        sim = litebird_sim.Simulation(name="My simulation")
        print(f"Running {sim.name}, saving results in {sim.base_path}")

    The member variable `observations` is a list of
    :class:`.Observation` objects, which is initialized by the method
    :meth:`.create_observations`.

    This class keeps track of any output file saved in `base_path`
    through the member variable `self.list_of_outputs`. This is a list
    of objects of type :py:meth:`OutputFileRecord`, which are 2-tuples
    of the form ``(path, description)``, where ``path`` is a
    ``pathlib.Path`` object and ``description`` is a `str` object::

        for curpath, curdescr in sim.list_of_outputs:
            print(f"{curpath}: {curdescr}")

    When pointing information is needed, you can call the method
    :meth:`.Simulation.set_scanning_strategy`, which
    initializes the members `pointing_freq_hz` and
    `spin2ecliptic_quats`; these members are used by functions like
    :func:`.get_pointings`.

    Args:

        base_path (str or `pathlib.Path`): the folder that will
            contain the output. If this folder does not exist and the
            user has sufficient rights, it will be created.

        name (str): a string identifying the simulation. This will
            be used in the reports.

        mpi_comm: either `None` (do not use MPI) or a MPI communicator
            object, like `mpi4py.MPI.COMM_WORLD`.

        description (str): a (possibly long) description of the
            simulation, to be put in the report saved in `base_path`).

        start_time (float or ``astropy.time.Time``): the start time of
            the simulation. It can be either an arbitrary
            floating-point number (e.g., 0) or an
            ``astropy.time.Time`` instance; in the latter case, this
            triggers a more precise (and slower) computation of
            pointing information.

        duration_s (float): Number of seconds the simulation should
            last.

        imo (:class:`.Imo`): an instance of the :class:`.Imo` class

        parameter_file (str or `pathlib.Path`): path to a TOML file
            that contains the parameters for the simulation. This file
            will be copied into `base_path`, and its contents will be
            read into the field `parameters` (a Python dictionary).
    """

    def __init__(
        self,
        base_path=None,
        name=None,
        mpi_comm=MPI_COMM_WORLD,
        description="",
        start_time=None,
        duration_s=None,
        imo=None,
        parameter_file=None,
        parameters=None,
    ):
        self.mpi_comm = mpi_comm

        self._initialize_logging()

        self.base_path = base_path
        self.name = name

        self.observations = []

        self.start_time = start_time
        self.duration_s = duration_s

        self.detectors = []  # type: List[DetectorInfo]
        self.instrument = None  # type: Optional[InstrumentInfo]
        self.hwp = None  # type: Optional[HWP]

        self.spin2ecliptic_quats = None

        self.description = description

        self.random = None

        self.tod_list = []  # type: List[TodDescription]

        if imo:
            self.imo = imo
        else:
            # TODO: read where to read the IMO from some parameter file
            self.imo = Imo()

        assert not (parameter_file and parameters), (
            "you cannot use parameter_file and parameters together "
            + "when constructing a litebird_sim.Simulation object"
        )

        if parameter_file:
            self.parameter_file = Path(parameter_file)

            with self.parameter_file.open("rt") as inpf:
                param_file_contents = "".join(inpf.readlines())

            self.parameters = _tomlkit_to_popo(tomlkit.parse(param_file_contents))
        else:
            self.parameter_file = None
            self.parameters = parameters

        self._init_missing_params()

        if not self.base_path:
            self.base_path = Path()

        self.base_path = Path(self.base_path)
        # Create any parent folder, and don't complain if the folder
        # already exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        if parameter_file:
            # Copy the parameter file to the output directory only if
            # it is not already there (this might happen if you did
            # not specify `base_path`, as the default for `base_path`
            # is the current working directory)
            dest_param_file = (self.base_path / self.parameter_file.name).resolve()
            try:
                copyfile(src=self.parameter_file, dst=dest_param_file)
            except SameFileError:
                pass

        self.list_of_outputs = []  # type: List[OutputFileRecord]

        self.report = ""

        # Add a header to the report
        template_file_path = get_template_file_path("report_header.md")
        with template_file_path.open("rt") as inpf:
            markdown_template = "".join(inpf.readlines())
        self.append_to_report(
            markdown_template,
            name=self.name if (self.name and (self.name != "")) else "<Untitled>",
            description=self.description,
            start_time=self.start_time.to_datetime()
            if isinstance(self.start_time, astropy.time.Time)
            else self.start_time,
            duration_s=self.duration_s,
        )

        # Make sure that self.random is initialized to something meaningful.
        # The user is free to call self.init_random() again later
        self.init_random()

    def init_random(self, seed=12345):
        """
        Initialize a random number generator in the `random` field

        This function creates a random number generator and saves it in the
        field `random`. It should be used whenever a random number generator
        is needed in the simulation. It ensures that different MPI processes
        have their own different seed, which stems from the parameter `seed`.
        The generator is PCG64, and it is ensured that the sequences in
        each MPI process are independent.

        This method is automatically called in the constructor, but it can be
        called again as many times as required. The typical case is when
        one wants to use a seed that has been read from a parameter file.
        """
        from numpy.random import Generator, PCG64, SeedSequence

        # We need to assign a different random number generator to each MPI
        # process, otherwise noise will be correlated. The following code
        # works even if MPI is not used

        # Create a list of N seeds, one per each MPI process
        seed_seq = SeedSequence(seed).spawn(self.mpi_comm.size)

        # Pick the seed for this process
        self.random = Generator(PCG64(seed_seq[self.mpi_comm.rank]))

    def _init_missing_params(self):
        """Initialize empty parameters using self.parameters

        This function should only be called in the ``__init__``
        constructor. It initializes a few member variables with the
        values in self.parameters (usually read from a TOML file), if
        these parameters do not already have a sensible value.

        """
        if not self.parameters:
            return

        try:
            sim_params = self.parameters["simulation"]
        except KeyError:
            return

        if not self.base_path:
            self.base_path = Path(sim_params.get("base_path", Path()))

        if not self.start_time:
            from datetime import date, datetime

            self.start_time = sim_params.get("start_time", None)
            if (
                isinstance(self.start_time, str)
                or isinstance(self.start_time, date)
                or isinstance(self.start_time, datetime)
            ):
                self.start_time = astropy.time.Time(self.start_time)

        if not self.duration_s:
            self.duration_s = sim_params.get("duration_s", None)

            # Let's check if the user specified the measurement unit
            # for the duration
            if isinstance(self.duration_s, str):
                conversions = [
                    ("years", astropy.units.year),
                    ("year", astropy.units.year),
                    ("days", astropy.units.day),
                    ("day", astropy.units.day),
                    ("hours", astropy.units.hour),
                    ("hour", astropy.units.hour),
                    ("minutes", astropy.units.minute),
                    ("min", astropy.units.minute),
                    ("sec", astropy.units.second),
                    ("s", astropy.units.second),
                ]

                for conv_str, conv_unit in conversions:
                    if self.duration_s.endswith(" " + conv_str):
                        value = float(self.duration_s.replace(conv_str, ""))
                        self.duration_s = (value * conv_unit).to("s").value
                        break

                if isinstance(self.duration_s, str):
                    # It's still a string, so no valid unit was found
                    # in the for loop above: convert it back to a
                    # number
                    self.duration_s = float(self.duration_s)

        if not self.name:
            self.name = sim_params.get("name", None)

        if self.description == "":
            self.description = sim_params.get("description", "")

    def _initialize_logging(self):
        if self.mpi_comm:
            mpi_rank = self.mpi_comm.rank
            log_format = "[%(asctime)s %(levelname)s MPI#{0:04d}] %(message)s".format(
                mpi_rank
            )
        else:
            mpi_rank = 0
            log_format = "[%(asctime)s %(levelname)s] %(message)s"

        if "LOG_DEBUG" in os.environ:
            log_level = log.DEBUG
        else:
            log_level = log.INFO

        if "LOG_ALL_MPI" in os.environ:
            log.basicConfig(level=log_level, format=log_format)
        else:
            if mpi_rank == 0:
                log.basicConfig(level=log_level, format=log_format)
            else:
                log.basicConfig(level=log.CRITICAL, format=log_format)

    def write_healpix_map(self, filename: str, pixels, **kwargs) -> str:
        """Save a Healpix map in the output folder

        Args:

            filename (``str`` or ``pathlib.Path``): Name of the
                file. It must be a relative path, but it can include
                subdirectories.

            pixels: array containing the pixels, or list of arrays if
                you want to save several maps into the same FITS table
                (e.g., I, Q, U components)

        Return:

            A `pathlib.Path` object containing the full path of the
            FITS file that has been saved.

        Example::

          import numpy as np

          sim = Simulation(base_path="/storage/litebird/mysim")
          pixels = np.zeros(12)
          sim.write_healpix_map("zero_map.fits.gz", pixels)

        This method saves an Healpix map into a FITS files that is
        written into the output folder for the simulation.

        """
        filename = self.base_path / Path(filename)
        write_healpix_map_to_file(filename=filename, pixels=pixels, **kwargs)

        self.list_of_outputs.append(
            OutputFileRecord(path=filename, description="Healpix map")
        )

        return filename

    def append_to_report(
        self,
        markdown_text: str,
        append_newline=True,
        figures: List[Tuple[Any, str]] = [],
        **kwargs,
    ):
        """Append text and figures to the simulation report

        Args:

            markdown_text (str): text to be appended to the report.

            append_newline (bool): append newlines to the end of the
                text. This ensures that calling again this method will
                produce a separate paragraph.

            figures (list of 2-tuples): list of Matplotlib figures to
                be saved in the report. Each tuple must contain one
                figure and one filename. The figures will be saved
                using the specified file name in the output
                directory. The file name must match the one used as
                reference in the Markdown text.

            kwargs: any other keyword argument will be used to expand
                the text `markdown_text` using the `Jinja2 library
                <https://palletsprojects.com/p/jinja/>`_ library.

        A Simulation class can generate reports in Markdown format.
        Use this function to add some text to the report, possibly
        including figures. The function has no effect if called from
        an MPI rank different from #0.

        It is possible to use objects other than Matplotlib
        figures. The only method this function calls is `savefig`,
        with no arguments.

        Images are saved immediately during the call, but the text
        will be written to disk only when
        :py:meth:`~litebird_sim.simulation.Simulation.flush` is called.

        You can put LaTeX formulae in the text, using ``$`...`$``
        for inline equations and the `math` tag in fenced text for
        displayed equations.

        """

        # Generate the report only if running on MPI rank #0
        if self.mpi_comm.rank != 0:
            return

        template = jinja2.Template(markdown_text)
        expanded_text = template.render(**kwargs)
        self.report += expanded_text

        if append_newline:
            self.report += "\n\n"

        for curfig, curfilename in figures:
            curpath = self.base_path / curfilename
            curfig.savefig(curpath)
            self.list_of_outputs.append(
                OutputFileRecord(path=curpath, description="Figure")
            )

    def _fill_dictionary_with_imo_information(
        self, dictionary: Dict[str, Any], base_imo_url: str
    ):
        # Fill the variable "dictionary" with information about the
        # objects retrieved from the IMO. This is used when producing
        # the final report for a simulation
        if not self.imo:
            return

        entities = [
            self.imo.query_entity(x, track=False)
            for x in self.imo.get_queried_entities()
        ]
        quantities = [
            self.imo.query_quantity(x, track=False)
            for x in self.imo.get_queried_quantities()
        ]
        data_files = [
            self.imo.query_data_file(x, track=False)
            for x in self.imo.get_queried_data_files()
        ]
        warnings = []

        # Check if there are newer versions of the data files used in the simulation
        for cur_data_file in data_files:
            other_data_files = self.imo.get_list_of_data_files(
                cur_data_file.quantity, track=False
            )
            if not other_data_files:
                continue

            if other_data_files[-1].uuid != cur_data_file.uuid:
                warnings.append((cur_data_file, other_data_files[-1]))

        if (not entities) and (not quantities) and (not data_files):
            return

        dictionary["entities"] = entities
        dictionary["quantities"] = quantities
        dictionary["data_files"] = data_files
        dictionary["warnings"] = warnings
        dictionary["base_imo_url"] = base_imo_url

    def _fill_dictionary_with_code_status(self, dictionary, include_git_diff):
        # Fill the variable "dictionary" with information about the
        # status of the "litebird_sim" code (which version is it? was
        # it patched? etc.) It is used when producing the final report
        # for a simulation
        dictionary["litebird_sim_version"] = litebird_sim_version
        dictionary["litebird_sim_author"] = litebird_sim_author

        # Retrieve information about the last git commit
        try:
            proc = subprocess.run(
                ["git", "log", "-1", "--format=format:%h%n%H%n%s%n%an"],
                capture_output=True,
                encoding="utf-8",
            )

            (
                short_commit_hash,
                commit_hash,
                commit_message,
                author,
            ) = proc.stdout.strip().split("\n")

            dictionary["short_commit_hash"] = short_commit_hash
            dictionary["commit_hash"] = commit_hash
            dictionary["author"] = author
            dictionary["commit_message"] = commit_message

            # Retrieve information about changes in the code since the last commit
            if include_git_diff:
                proc = subprocess.run(
                    ["git", "diff", "--no-color", "--exit-code"],
                    capture_output=True,
                    encoding="utf-8",
                )

                if proc.returncode != 0:
                    dictionary["code_diff"] = proc.stdout.strip()

            else:
                dictionary["skip_code_diff"] = True

        except FileNotFoundError:
            # Git is not installed, so ignore the error and continue
            pass
        except Exception as e:
            log.warning(
                f"unable to save information about latest git commit in the report: {e}"
            )

    def flush(self, include_git_diff=True, base_imo_url: str = DEFAULT_BASE_IMO_URL):
        """Terminate a simulation.

        This function must be called when a simulation is complete. It
        will save pending data to the output directory.

        It returns a `Path` object pointing to the HTML file that has
        been saved in the directory pointed by ``self.base_path``.

        """

        dictionary = {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self._fill_dictionary_with_imo_information(
            dictionary, base_imo_url=base_imo_url
        )
        self._fill_dictionary_with_code_status(dictionary, include_git_diff)

        template_file_path = get_template_file_path("report_appendix.md")
        with template_file_path.open("rt") as inpf:
            markdown_template = "".join(inpf.readlines())
        self.append_to_report(markdown_template, **dictionary)

        # Expand the markdown text using Jinja2
        with codecs.open(self.base_path / "report.md", "w", encoding="utf-8") as outf:
            outf.write(self.report)

        # Now generate an HTML file from Markdown.

        # Please keep these in alphabetic order, so we can detect duplicates!
        md_extensions = [
            KatexExtension(),
            "fenced_code",
            "sane_lists",
            "smarty",
            "tables",
            "toc",
        ]
        html = markdown.markdown(self.report, extensions=md_extensions)

        static_path = Path(__file__).parent / ".." / "static"
        with codecs.open(static_path / "report_template.html") as inpf:
            html_full_report = jinja2.Template(inpf.read()).render(
                name=self.name, html=html
            )

        # Copy all the files in static/
        static_files_to_copy = ["sakura.css"]
        for curitem in static_files_to_copy:
            source = static_path / curitem
            if source.is_dir():
                copytree(src=source, dst=self.base_path)
            else:
                copyfile(src=source, dst=self.base_path / curitem)

        # Finally, write down the full HTML report
        html_report_path = self.base_path / "report.html"
        with codecs.open(html_report_path, "w", encoding="utf-8") as outf:
            outf.write(html_full_report)

        return html_report_path

    def create_observations(
        self,
        detectors: List[DetectorInfo],
        num_of_obs_per_detector: int = 1,
        split_list_over_processes=True,
        n_blocks_det=1,
        n_blocks_time=1,
        root=0,
        dtype_tod: Optional[Any] = None,
        tods: List[TodDescription] = [
            TodDescription(name="tod", dtype=np.float32, description="Signal")
        ],
    ):
        """Create a set of Observation objects.

        This method initializes the `Simulation.observations` field of
        the class with a list of observations referring to the
        detectors listed in `detectors`. By default there is *one*
        observation per detector, but you can tune the number using
        the parameter `num_of_obs_per_detector`: this is useful if you
        are simulating a long experiment in a MPI job.

        If `split_list_over_processes` is set to ``True`` (the
        default), the set of observations will be distributed evenly
        among the MPI processes associated with ``self.mpi_comm``
        (initialized in :meth:`.Simulation.__init__`). If
        `split_list_over_processes` is ``False``, then no distribution
        will happen: this can be useful if you are running a MPI job
        but you want to take care of the actual distribution of the
        observations among the MPI workers instead of relying on the
        default distribution algorithm.

        Each observation can hold information about more than one
        detector; the parameters `n_blocks_det` specify how many
        groups of detectors will be created. For instance, if you are
        simulating 10 detectors and you specify ``n_blocks_det=5``,
        this means that each observation will handle ``10 / 5 = 2``
        detectors. The default is that *all* the detectors be kept
        together (``n_blocks_det=1``).

        The parameter `n_blocks_time` specifies the number of time
        splits of the observations. In the case of a 3-month-long
        observation, `n_blocks_time=3` means that each observation
        will cover one month.

        The parameter `tods` specifies how many TOD arrays should be
        created. Each element should be an instance of
        :class:`.TodDescription` and contain the fields ``name`` (the name
        of the member variable that will be created), ``dtype`` (the
        NumPy type to use, like ``numpy.float32``), and ``description``
        (a free-form description). The default is ``numpy.float32``,
        which should be adequate for LiteBIRD's purposes; if you
        want greater accuracy at the expense of doubling memory
        occupation, choose ``numpy.float64``.

        If you specify `dtype_tod`, this will be used as the parameter
        for each TOD specified in `tods`, overriding the value of `dtype`.
        This keyword is kept for legacy reasons but should be avoided
        in newer code.

        Here is an example that creates three TODs::

            sim.create_observations(
                [det1, det2],
                tods=[
                    TodDescription(
                        name="fg_tod",
                        dtype=np.float32,
                        description="Foregrounds (computed by PySM)",
                    ),
                    TodDescription(
                        name="cmb_tod",
                        dtype=np.float32,
                        description="CMB realization following Planck (2018)",
                    ),
                    TodDescription(
                        name="noise_tod",
                        dtype=np.float32,
                        description="Noise TOD (only white noise, no 1/f)",
                    ),
                ],
            )

            # Now you can access these fields:
            # - sim.fg_tod
            # - sim.cmb_tod
            # - sim.noise_tod
        """

        assert (
            self.start_time is not None
        ), "you must set start_time when creating the Simulation object"

        assert isinstance(
            self.duration_s, (float, int)
        ), "you must set duration_s when creating the Simulation object"

        if not detectors:
            detectors = self.detectors

        # if a single detector is passed, make it a list
        if isinstance(detectors, DetectorInfo):
            detectors = [detectors]

        observations = []

        duration_s = self.duration_s  # Cache the value to a local variable
        sampfreq_hz = detectors[0].sampling_rate_hz
        self.detectors = detectors
        num_of_samples = int(sampfreq_hz * duration_s)
        samples_per_obs = distribute_evenly(num_of_samples, num_of_obs_per_detector)

        cur_time = self.start_time

        if not dtype_tod:
            self.tod_list = tods
        else:
            self.tod_list = [
                TodDescription(name=x.name, dtype=dtype_tod, description=x.description)
                for x in tods
            ]

        for cur_obs_idx in range(num_of_obs_per_detector):
            nsamples = samples_per_obs[cur_obs_idx].num_of_elements
            cur_obs = Observation(
                detectors=[asdict(d) for d in detectors],
                start_time_global=cur_time,
                sampling_rate_hz=sampfreq_hz,
                n_samples_global=nsamples,
                n_blocks_det=n_blocks_det,
                n_blocks_time=n_blocks_time,
                comm=(None if split_list_over_processes else self.mpi_comm),
                root=0,
                tods=self.tod_list,
            )
            observations.append(cur_obs)

            time_span = nsamples / sampfreq_hz
            if isinstance(self.start_time, astropy.time.Time):
                time_span = astropy.time.TimeDelta(time_span, format="sec")

            cur_time += time_span

        if split_list_over_processes:
            self.distribute_workload(observations)
        else:
            self.observations = observations

        return observations

    def get_tod_names(self) -> List[str]:
        return [x.name for x in self.tod_list]

    def get_tod_dtypes(self) -> List[Any]:
        return [x.dtype for x in self.tod_list]

    def get_tod_descriptions(self) -> List[str]:
        return [x.description for x in self.tod_list]

    def get_list_of_tods(self) -> List[TodDescription]:
        return self.tod_list

    def distribute_workload(self, observations: List[Observation]):
        if self.mpi_comm.size == 1:
            self.observations = observations
            return

        cur_rank = self.mpi_comm.rank
        span = distribute_optimally(
            elements=observations,
            num_of_groups=self.mpi_comm.size,
            weight_fn=lambda obs: obs.n_samples_global,
        )[cur_rank]

        self.observations = observations[
            span.start_idx : (span.start_idx + span.num_of_elements)
        ]

    def describe_mpi_distribution(self) -> Optional[MpiDistributionDescr]:
        """Return a :class:`.MpiDistributionDescr` object describing observations

        This method returns a :class:`.MpiDistributionDescr` that describes the data
        stored in each MPI process running concurrently. It is a great debugging tool
        when you are using MPI, and it can be used for tasks where you have to carefully
        orchestrate they way different MPI processes run together.

        If this method is called before :meth:`.Simulation.create_observations`, it will
        return ``None``.

        This method should be called by *all* the MPI processes. It can be executed in a
        serial environment (i.e., without MPI) and will still return meaningful values.

        The typical usage for this method is to call it once you have called
        :meth:`.Simulation.create_observations` to check that the TODs have been
        laid in memory in the way you expect::

            sim.create_observations(…)
            distr = sim.describe_mpi_distribution()
            if litebird_sim.MPI_COMM_WORLD.rank == 0:
                print(distr)

        """

        if not self.observations:
            return None

        observation_descr = []  # type: List[MpiObservationDescr]
        for obs in self.observations:
            cur_det_names = list(obs.name)

            shapes = [
                tuple(getattr(obs, cur_tod.name).shape) for cur_tod in self.tod_list
            ]
            # Check that all the TODs have the same shape
            if shapes:
                for i in range(1, len(shapes)):
                    assert shapes[0] == shapes[i], (
                        f"TOD {self.tod_list[0].name} and {self.tod_list[i].name} "
                        + f"have different shapes: {shapes[0]} vs {shapes[i]}"
                    )

            observation_descr.append(
                MpiObservationDescr(
                    det_names=cur_det_names,
                    tod_names=self.get_tod_names(),
                    tod_shape=shapes[0] if shapes else None,
                    tod_dtype=self.get_tod_dtypes(),
                    tod_description=self.get_tod_descriptions(),
                    start_time=obs.start_time,
                    duration_s=obs.n_samples / obs.sampling_rate_hz,
                    num_of_samples=obs.n_samples,
                    num_of_detectors=obs.n_detectors,
                )
            )

        num_of_observations = len(self.observations)

        if self.mpi_comm and litebird_sim.MPI_ENABLED:
            observation_descr_all = MPI_COMM_WORLD.allgather(observation_descr)
            num_of_observations_all = MPI_COMM_WORLD.allgather(num_of_observations)
        else:
            observation_descr_all = [observation_descr]
            num_of_observations_all = [num_of_observations]

        mpi_processes = []  # type: List[MpiProcessDescr]
        for i in range(MPI_COMM_WORLD.size):
            mpi_processes.append(
                MpiProcessDescr(
                    mpi_rank=i,
                    observations=observation_descr_all[i],
                )
            )

        return MpiDistributionDescr(
            num_of_observations=sum(num_of_observations_all),
            detectors=self.detectors,
            mpi_processes=mpi_processes,
        )

    def set_scanning_strategy(
        self,
        scanning_strategy: Union[None, ScanningStrategy] = None,
        imo_url: Union[None, str] = None,
        delta_time_s: float = 60.0,
        append_to_report: bool = True,
    ):
        """Simulate the motion of the spacecraft in free space

        This method computes the quaternions that encode the evolution
        of the spacecraft's orientation in time, assuming the scanning
        strategy described in the parameter `scanning_strategy` (an
        object of a class derived by :class:`.ScanningStrategy`; most
        likely, you want to use :class:`SpinningScanningStrategy`).
        The result is saved in the member variable
        ``spin2ecliptic_quats``, which is an instance of the class
        :class:`.Spin2EclipticQuaternions`.

        You can choose to use the `imo_url` parameter instead of
        `scanning_strategy`: in this case, it will be assumed that you
        want to simulate a nominal, spinning scanning strategy, and
        the object in the IMO with address `imo_url` (e.g.,
        ``/releases/v1.0/satellite/scanning_parameters/``) describing
        the parameters of the scanning strategy will be loaded. In
        this case, a :class:`SpinningScanningStrategy` object will be
        created automatically.

        The parameter `delta_time_s` specifies how often should
        quaternions be computed; see
        :meth:`.ScanningStrategy.set_scanning_strategy` for
        more information.

        If the parameter `append_to_report` is set to ``True`` (the
        default), some information about the pointings will be included
        in the report saved by the :class:`.Simulation` object. This will
        be done only if the process has rank #0.

        """
        assert not (scanning_strategy and imo_url), (
            "you must either specify scanning_strategy or imo_url (but not"
            "the two together) when calling Simulation.set_scanning_strategy"
        )

        if not scanning_strategy:
            if not imo_url:
                imo_url = "/releases/v1.0/satellite/scanning_parameters/"

            scanning_strategy = SpinningScanningStrategy.from_imo(
                imo=self.imo, url=imo_url
            )

        # TODO: if MPI is enabled, we should probably parallelize this call
        self.spin2ecliptic_quats = scanning_strategy.generate_spin2ecl_quaternions(
            start_time=self.start_time,
            time_span_s=self.duration_s,
            delta_time_s=delta_time_s,
        )
        quat_memory_size_bytes = self.spin2ecliptic_quats.nbytes()

        num_of_obs = len(self.observations)
        if append_to_report and litebird_sim.MPI_ENABLED:
            num_of_obs = MPI_COMM_WORLD.allreduce(num_of_obs)

        if append_to_report and MPI_COMM_WORLD.rank == 0:
            template_file_path = get_template_file_path("report_quaternions.md")
            with template_file_path.open("rt") as inpf:
                markdown_template = "".join(inpf.readlines())
            self.append_to_report(
                markdown_template,
                num_of_obs=num_of_obs,
                num_of_mpi_processes=MPI_COMM_WORLD.size,
                delta_time_s=delta_time_s,
                quat_memory_size_bytes=quat_memory_size_bytes,
            )

    @deprecated(
        deprecated_in="0.9",
        current_version=litebird_sim_version,
        details="Use set_scanning_strategy",
    )
    def generate_spin2ecl_quaternions(
        self,
        scanning_strategy: Union[None, ScanningStrategy] = None,
        imo_url: Union[None, str] = None,
        delta_time_s: float = 60.0,
        append_to_report=True,
    ):
        self.set_scanning_strategy(
            scanning_strategy=scanning_strategy,
            imo_url=imo_url,
            delta_time_s=delta_time_s,
            append_to_report=append_to_report,
        )

    def set_instrument(self, instrument: InstrumentInfo):
        """Set the instrument to be used in the simulation.

        This function sets the ``self.instrument`` field to the instance
        of the class :class:`.InstrumentInfo` that has been passed as
        argument. The purpose of the instrument is to provide the reference
        frame for the direction of each detector.

        Note that you should not simulate more than one instrument in the same
        simulation. This is enforced by the fact that if you call `set_instrument`
        twice, the second call will overwrite the instrument that was formerly
        set.
        """
        self.instrument = instrument

    def set_hwp(self, hwp: HWP):
        """Set the HWP to be used in the simulation

        The argument must be a class derived from :class:`.HWP`, for instance
        :class:`.IdealHWP`.
        """
        self.hwp = hwp

    def compute_pointings(
        self,
        append_to_report: bool = True,
        dtype_pointing=np.float32,
    ):
        """Trigger the computation of pointings.

        This method must be called after having set the scanning strategy, the
        instrument, and the list of detectors to simulate through calls to
        :meth:`.set_instrument` and :meth:`.add_detector`. It combines the
        quaternions of the spacecraft, of the instrument, and of the detectors
        and sets the fields ``pointings``, ``psi``, and ``pointing_coords`` in
        each observation owned by the simulation.
        """
        assert self.detectors, (
            "You must call Simulation.create_observations() "
            "before calling Simulation.compute_pointings"
        )
        assert self.instrument
        assert self.spin2ecliptic_quats

        memory_occupation = 0
        num_of_obs = 0
        for cur_obs in self.observations:
            quaternion_buffer = np.zeros(
                (cur_obs.n_samples, 1, 4),
                dtype=np.float64,
            )
            get_pointings(
                cur_obs,
                self.spin2ecliptic_quats,
                detector_quats=cur_obs.quat,
                bore2spin_quat=self.instrument.bore2spin_quat,
                hwp=self.hwp,
                quaternion_buffer=quaternion_buffer,
                dtype_pointing=dtype_pointing,
                store_pointings_in_obs=True,
            )
            del quaternion_buffer

            memory_occupation += cur_obs.pointings.nbytes + cur_obs.psi.nbytes
            num_of_obs += 1

        if append_to_report and litebird_sim.MPI_ENABLED:
            memory_occupation = MPI_COMM_WORLD.allreduce(memory_occupation)
            num_of_obs = MPI_COMM_WORLD.allreduce(num_of_obs)

        if append_to_report and MPI_COMM_WORLD.rank == 0:
            template_file_path = get_template_file_path("report_pointings.md")
            with template_file_path.open("rt") as inpf:
                markdown_template = "".join(inpf.readlines())
            self.append_to_report(
                markdown_template,
                num_of_obs=num_of_obs,
                hwp_description=str(self.hwp) if self.hwp else "No HWP",
                num_of_mpi_processes=MPI_COMM_WORLD.size,
                memory_occupation=memory_occupation,
            )

    def compute_pos_and_vel(
        self,
        delta_time_s=86400.0,
        solar_velocity_km_s: float = c.SOLAR_VELOCITY_KM_S,
        solar_velocity_gal_lat_rad: float = c.SOLAR_VELOCITY_GAL_LAT_RAD,
        solar_velocity_gal_lon_rad: float = c.SOLAR_VELOCITY_GAL_LON_RAD,
    ):
        """Computes the position and the velocity of the spacescraft for computing
        the dipole.
        It wraps the :class:`.SpacecraftOrbit` and calls :meth:`.SpacecraftOrbit`.
        The parameters that can be modified are the sampling of position and velocity
        and the direction and amplitude of the solar dipole.
        Default values for solar dipole from Planck 2018 Solar dipole (see arxiv:
        1807.06207)
        """

        orbit = SpacecraftOrbit(
            self.start_time,
            solar_velocity_km_s=solar_velocity_km_s,
            solar_velocity_gal_lat_rad=solar_velocity_gal_lat_rad,
            solar_velocity_gal_lon_rad=solar_velocity_gal_lon_rad,
        )

        self.pos_and_vel = spacecraft_pos_and_vel(
            orbit=orbit, obs=self.observations, delta_time_s=delta_time_s
        )

    def fill_tods(
        self,
        maps: Dict[str, np.ndarray],
        input_map_in_galactic: bool = True,
        interpolation: Union[str, None] = "",
        append_to_report: bool = True,
    ):
        """Fills the TODs, scanning a map.

        This method must be called after having set the scanning strategy, the
        instrument, the list of detectors to simulate through calls to
        :meth:`.set_instrument` and :meth:`.add_detector`, and the methond
        compute_pointings. maps is assumed to be produced by :class:`.Mbs`
        """

        scan_map_in_observations(
            self.observations,
            maps=maps,
            input_map_in_galactic=input_map_in_galactic,
            interpolation=interpolation,
        )

        if append_to_report and MPI_COMM_WORLD.rank == 0:
            template_file_path = get_template_file_path("report_scan_map.md")
            with template_file_path.open("rt") as inpf:
                markdown_template = "".join(inpf.readlines())
            if type(maps) is dict:
                if "Mbs_parameters" in maps.keys():
                    if maps["Mbs_parameters"].make_fg:
                        fg_model = maps["Mbs_parameters"].fg_models
                    else:
                        fg_model = "N/A"

                    self.append_to_report(
                        markdown_template,
                        nside=maps["Mbs_parameters"].nside,
                        has_cmb=maps["Mbs_parameters"].make_cmb,
                        has_fg=maps["Mbs_parameters"].make_fg,
                        fg_model=fg_model,
                    )
            else:
                nside = npix_to_nside(len(maps[0]))
                self.append_to_report(
                    markdown_template,
                    nside=nside,
                    has_cmb="N/A",
                    has_fg="N/A",
                    fg_model="N/A",
                )

    def add_dipole(
        self,
        t_cmb_k: float = c.T_CMB_K,
        dipole_type: DipoleType = DipoleType.TOTAL_FROM_LIN_T,
        append_to_report: bool = True,
    ):
        """Fills the tod with dipole.

        This method must be called after having set the scanning strategy, the
        instrument, the list of detectors to simulate through calls to
        :meth:`.set_instrument` and :meth:`.add_detector`, and the pointing
        through :meth:`.compute_pointings`.
        """

        if not hasattr(self, "pos_and_vel"):
            self.compute_pos_and_vel()

        add_dipole_to_observations(
            obs=self.observations,
            pos_and_vel=self.pos_and_vel,
            t_cmb_k=t_cmb_k,
            dipole_type=dipole_type,
        )

        if append_to_report and MPI_COMM_WORLD.rank == 0:
            template_file_path = get_template_file_path("report_dipole.md")

            dip_lat_deg = np.rad2deg(self.pos_and_vel.orbit.solar_velocity_gal_lat_rad)
            dip_lon_deg = np.rad2deg(self.pos_and_vel.orbit.solar_velocity_gal_lon_rad)
            dip_velocity = self.pos_and_vel.orbit.solar_velocity_km_s

            with template_file_path.open("rt") as inpf:
                markdown_template = "".join(inpf.readlines())
            self.append_to_report(
                markdown_template,
                t_cmb_k=t_cmb_k,
                dipole_type=dipole_type,
                dip_lat_deg=dip_lat_deg,
                dip_lon_deg=dip_lon_deg,
                dip_velocity=dip_velocity,
            )

    def add_noise(
        self,
        noise_type: str = "one_over_f",
        random: Union[np.random.Generator, None] = None,
        append_to_report: bool = True,
    ):
        """Adds noise to tods.

        This method must be called after having set the instrument,
        the list of detectors to simulate through calls to
        :meth:`.set_instrument` and :meth:`.add_detector`.
        """

        if random is None:
            random = self.random

        add_noise_to_observations(
            obs=self.observations,
            noise_type=noise_type,
            random=random,
        )

        if append_to_report and MPI_COMM_WORLD.rank == 0:
            template_file_path = get_template_file_path("report_noise.md")
            with template_file_path.open("rt") as inpf:
                markdown_template = "".join(inpf.readlines())
            self.append_to_report(
                markdown_template,
                noise_type="white + 1/f " if noise_type == "one_over_f" else "white",
            )

    def binned_map(
        self,
        nside: int,
        do_covariance: bool = False,
        output_map_in_galactic: bool = True,
        append_to_report: bool = True,
    ):

        """
        Bins the tods of `sim.observations` into maps.
        The syntax mimics the one of :meth:`litebird_sim.make_bin_map`
        """

        if append_to_report and MPI_COMM_WORLD.rank == 0:
            template_file_path = get_template_file_path("report_binned_map.md")
            with template_file_path.open("rt") as inpf:
                markdown_template = "".join(inpf.readlines())
            self.append_to_report(
                markdown_template,
                nside=nside,
                coord="Galactic" if output_map_in_galactic else "Ecliptic",
            )

        return make_bin_map(
            obs=self.observations,
            nside=nside,
            do_covariance=do_covariance,
            output_map_in_galactic=output_map_in_galactic,
        )

    def apply_gaindrift(
        self,
        drift_params: GainDriftParams = None,
        user_seed: int = 12345,
        component: str = "tod",
        append_to_report: bool = True,
    ):
        """A method to apply the gain drift to the observation.

        This is a wrapper around :func:`.apply_gaindrift_to_observations()` that
        injects gain drift to a list of :class:`.Observation` instance.

        Args:

            drift_params (:class:`.GainDriftParams`, optional): The gain
                drift injection parameters object. Defaults to None.

            user_seed (int, optional): A seed provided by the user.
                Defaults to 12345.

            component (str, optional): The name of the TOD on which the
                gain drift has to be injected. Defaults to "tod".

            append_to_report (bool, optional): Defaults to True.
        """

        if drift_params is None:
            drift_params = GainDriftParams()

        apply_gaindrift_to_observations(
            obs=self.observations,
            drift_params=drift_params,
            user_seed=user_seed,
            component=component,
        )

        if append_to_report and MPI_COMM_WORLD.rank == 0:

            dictionary = {
                "sampling_dist": "Gaussian" if drift_params.sampling_dist else "Uniform"
            }

            if drift_params.drift_type == GainDriftType.LINEAR_GAIN:
                dictionary["drift_type"] = "Linear"
                dictionary["linear_drift"] = True
                dictionary[
                    "calibration_period_sec"
                ] = drift_params.calibration_period_sec

            elif drift_params.drift_type == GainDriftType.THERMAL_GAIN:
                dictionary["drift_type"] = "Thermal"
                dictionary["thermal_drift"] = True

                keys_to_get = [
                    "sigma_drift",
                    "focalplane_group",
                    "oversample",
                    "fknee_drift_mHz",
                    "alpha_drift",
                    "detector_mismatch",
                    "thermal_fluctuation_amplitude_K",
                    "focalplane_Tbath_K",
                    "sampling_uniform_low",
                    "sampling_uniform_high",
                    "sampling_gaussian_loc",
                    "sampling_gaussian_scale",
                ]

                for key in keys_to_get:
                    dictionary[key] = getattr(drift_params, key)

            template_file_path = get_template_file_path("report_gaindrift.md")
            with template_file_path.open("rt") as inpf:
                markdown_template = "".join(inpf.readlines())
            self.append_to_report(
                markdown_text=markdown_template,
                component=component,
                user_seed=user_seed,
                **dictionary,
            )
