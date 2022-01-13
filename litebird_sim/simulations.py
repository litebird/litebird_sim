# -*- encoding: utf-8 -*-

import codecs
from collections import namedtuple
from dataclasses import asdict
from datetime import datetime
import logging as log
import os
import subprocess
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
from shutil import copyfile, copytree, SameFileError

from .detectors import DetectorInfo
from .distribute import distribute_evenly, distribute_optimally
from .healpix import write_healpix_map_to_file
from .imo.imo import Imo
from .mpi import MPI_COMM_WORLD
from .observations import Observation
from .version import (
    __version__ as litebird_sim_version,
    __author__ as litebird_sim_author,
)

import astropy.time
import astropy.units
import markdown
import numpy as np
import jinja2
import tomlkit

from markdown_katex import KatexExtension

from .scanning import ScanningStrategy, SpinningScanningStrategy


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
    :meth:`.Simulation.generate_spin2ecl_quaternions`, which
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

        self.spin2ecliptic_quats = None

        self.description = description

        self.random = None

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

    def _fill_dictionary_with_imo_information(self, dictionary: Dict[str, Any]):
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

    def flush(self, include_git_diff=True):
        """Terminate a simulation.

        This function must be called when a simulation is complete. It
        will save pending data to the output directory.

        It returns a `Path` object pointing to the HTML file that has
        been saved in the directory pointed by ``self.base_path``.

        """

        dictionary = {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self._fill_dictionary_with_imo_information(dictionary)
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
        dtype_tod=np.float32,
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
        simulating 10 detectors and you specify ``nblocks_det=5``,
        this means that each observation will handle ``10 / 5 = 2``
        detectors. The default is that *all* the detectors be kept
        together (``nblocks_det=1``).

        The parameter `n_blocks_time` specifies the number of time
        splits of the observations. In the case of a 3-month-long
        observation, `n_blocks_time=3` means that each observation
        will cover one month.

        The parameter `dtype_tod` specifies the data type to be used
        for the samples in the timestream. The default is
        ``numpy.float32``, which should be adequate for LiteBIRD's
        purposes; if you want greater accuracy at the expense of
        doubling memory occupation, choose ``numpy.float64``.

        """

        assert (
            self.start_time is not None
        ), "you must set start_time when creating the Simulation object"

        assert isinstance(
            self.duration_s, (float, int)
        ), "you must set duration_s when creating the Simulation object"

        observations = []

        duration_s = self.duration_s  # Cache the value to a local variable
        sampfreq_hz = detectors[0].sampling_rate_hz
        detectors = [asdict(d) for d in detectors]
        num_of_samples = int(sampfreq_hz * duration_s)
        samples_per_obs = distribute_evenly(num_of_samples, num_of_obs_per_detector)

        cur_time = self.start_time

        for cur_obs_idx in range(num_of_obs_per_detector):
            nsamples = samples_per_obs[cur_obs_idx].num_of_elements
            cur_obs = Observation(
                detectors=detectors,
                start_time_global=cur_time,
                sampling_rate_hz=sampfreq_hz,
                n_samples_global=nsamples,
                n_blocks_det=n_blocks_det,
                n_blocks_time=n_blocks_time,
                comm=(None if split_list_over_processes else self.mpi_comm),
                root=0,
                dtype_tod=dtype_tod,
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

    def generate_spin2ecl_quaternions(
        self,
        scanning_strategy: Union[None, ScanningStrategy] = None,
        imo_url: Union[None, str] = None,
        delta_time_s: float = 60.0,
    ):
        """Simulate the motion of the spacecraft in free space

        This method computes the quaternions that encode the evolution
        of the spacecraft's orientation in time, assuming the scanning
        strategy described in the parameter `scanning_strategy` (an
        object of a class derived by :class:`.ScanningStrategy`; most
        likely, you want to use :class:`SpinningScanningStrategy`).

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
        :meth:`.ScanningStrategy.generate_spin2ecl_quaternions` for
        more information.

        """
        assert not (scanning_strategy and imo_url), (
            "you must either specify scanning_strategy or imo_url (but not"
            "the two together) when calling Simulation.generate_spin2ecl_quaternions"
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

        template_file_path = get_template_file_path("report_generate_pointings.md")
        with template_file_path.open("rt") as inpf:
            markdown_template = "".join(inpf.readlines())
        self.append_to_report(
            markdown_template,
            num_of_obs=len(self.observations),
            delta_time_s=delta_time_s,
            quat_memory_size_bytes=quat_memory_size_bytes,
        )
