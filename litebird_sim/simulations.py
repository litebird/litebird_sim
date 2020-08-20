# -*- encoding: utf-8 -*-

import codecs
from collections import namedtuple
from datetime import datetime
import logging as log
import subprocess
from typing import List, Tuple, Union, Dict, Any
from pathlib import Path
from shutil import copyfile, copytree

from .distribute import distribute_evenly, distribute_optimally
from .healpix import write_healpix_map_to_file
from .imo.imo import Imo
from .observations import Observation
from .version import (
    __version__ as litebird_sim_version,
    __author__ as litebird_sim_author,
)

from astropy.time import Time, TimeDelta
import markdown
import jinja2

from markdown_katex import KatexExtension

OutputFileRecord = namedtuple("OutputFileRecord", ["path", "description"])


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
    :class:`.Observation` objects, which is initialized by the methods
    :meth:`.create_observations` (when ``distribute=True``) and
    :meth:`.distribute_workload`.

    This class keeps track of any output file saved in `base_path`
    through the member variable `self.list_of_outputs`. This is a list
    of objects of type :py:meth:`OutputFileRecord`, which are 2-tuples
    of the form ``(path, description)``, where ``path`` is a
    ``pathlib.Path`` object and ``description`` is a `str` object::

        for curpath, curdescr in sim.list_of_outputs:
            print(f"{curpath}: {curdescr}")

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

        imo (:class:`.Imo`): an instance of the :class:`.Imo` class

    """

    def __init__(
        self, base_path=Path(), name=None, mpi_comm=None, description="", imo=None,
    ):
        self.base_path = Path(base_path)
        self.name = name

        self.mpi_comm = mpi_comm

        self.observations = []

        self.description = description

        if imo:
            self.imo = imo
        else:
            # TODO: read where to read the IMO from some parameter file
            self.imo = Imo()

        # Create any parent folder, and don't complain if the folder
        # already exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.list_of_outputs = []  # type: List[OutputFileRecord]

        self.report = ""

        # Add an header to the report
        template_file_path = get_template_file_path("report_header.md")
        with template_file_path.open("rt") as inpf:
            markdown_template = "".join(inpf.readlines())
        self.append_to_report(
            markdown_template,
            name=name if (name and (name != "")) else "<Untitled>",
            description=description,
        )

    def write_healpix_map(self, filename: str, pixels, **kwargs,) -> str:
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
        write_healpix_map_to_file(
            filename=filename, pixels=pixels, **kwargs,
        )

        self.list_of_outputs.append(
            OutputFileRecord(path=filename, description="Healpix map")
        )

        return filename

    def append_to_report(
        self, markdown_text: str, figures: List[Tuple[Any, str]] = [], **kwargs
    ):
        """Append text and figures to the simulation report

        Args:

            markdown_text (str): text to be appended to the report.

            figures (list of 2-tuples): list of Matplotlib figures to
                be saved in the report. Each tuple must contain one
                figure and one filename. The figures will be saved
                using the specified file name in the output
                directory. The file name must match the one used as
                reference in the Markdown text.

            kwargs: any other keyword argument will be used to expand
                the text `markdown_text` using the `Jinja2 library
                <https://palletsprojects.com/p/jinja/>`_ library.

        A Simulation class can generate reports in Markdown
        format. Use this function to add some text to the report,
        possibly including figures.

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

        template = jinja2.Template(markdown_text)
        expanded_text = template.render(**kwargs)
        self.report += expanded_text

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
            data_files = self.imo.get_list_of_data_files(
                cur_data_file.quantity, track=False
            )
            if not data_files:
                continue

            if data_files[-1].uuid != cur_data_file.uuid:
                warnings.append((cur_data_file, data_files[-1]))

        if (not entities) and (not quantities) and (not data_files):
            return

        dictionary["entities"] = entities
        dictionary["quantities"] = quantities
        dictionary["data_files"] = data_files
        dictionary["warnings"] = warnings

    def _fill_dictionary_with_code_status(self, dictionary):
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
            proc = subprocess.run(
                ["git", "diff", "--no-color", "--exit-code"],
                capture_output=True,
                encoding="utf-8",
            )

            if proc.returncode != 0:
                dictionary["code_diff"] = proc.stdout.strip()

        except FileNotFoundError:
            # Git is not installed, so ignore the error and continue
            pass
        except Exception as e:
            log.warning(
                f"unable to save information about latest git commit in the report: {e}"
            )

    def flush(self):
        """Terminate a simulation.

        This function must be called when a simulation is complete. It
        will save pending data to the output directory.

        It returns a `Path` object pointing to the HTML file that has
        been saved in the directory pointed by ``self.base_path``.

        """

        dictionary = {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self._fill_dictionary_with_imo_information(dictionary)
        self._fill_dictionary_with_code_status(dictionary)

        template_file_path = get_template_file_path("report_appendix.md")
        with template_file_path.open("rt") as inpf:
            markdown_template = "".join(inpf.readlines())
        self.append_to_report(
            markdown_template, **dictionary,
        )

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
        with codecs.open(html_report_path, "w", encoding="utf-8",) as outf:
            outf.write(html_full_report)

        return html_report_path

    def create_observations(
        self,
        detectors: List[str],
        num_of_obs_per_detector: int,
        start_time,
        duration_s: float,
        sampling_rate_hz: float,
        distribute=True,
        use_mjd=False,
    ):
        "Create a set of Observation objects"

        self.observations = []
        num_of_samples = sampling_rate_hz * duration_s
        obs_spans = distribute_evenly(num_of_samples, num_of_obs_per_detector)

        # Keep only one every comm.size starting at my rank
        obs_spans = obs_spans[self.mpi_comm.rank::self.mpi_comm.size]
        for obs_span in obs_spans:
            obs_start_time = obs_span.start_idx / sampling_rate_hz
            if use_mjd:
                obs_start_time = (Time(mjd=start_time.mjd)
                                  + TimeDelta(seconds=obs_start_time))

            cur_obs = Observation(
                detectors=detectors,
                start_time=obs_start_time,
                sampling_rate_hz=sampling_rate_hz,
                nsamples=obs_span.num_of_elements,
                use_mjd=use_mjd,
            )
            self.observations.append(cur_obs)

        return self.observations
