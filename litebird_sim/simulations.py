# -*- encoding: utf-8 -*-

import codecs
from collections import namedtuple
from typing import List, Tuple, Any
from pathlib import Path
from shutil import copyfile, copytree

from .detectors import Detector
from .distribute import distribute_evenly, distribute_optimally
from .healpix import write_healpix_map_to_file
from .observations import Observation

from astropy.time import Time, TimeDelta
import markdown
import jinja2

from markdown_katex import KatexExtension

OutputFileRecord = namedtuple("OutputFileRecord", ["path", "description"])


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

    """

    def __init__(
        self, base_path=Path(), name=None, mpi_comm=None, description="",
    ):
        self.base_path = Path(base_path)
        self.name = name

        self.mpi_comm = mpi_comm

        self.observations = []

        self.description = description

        # Create any parent folder, and don't complain if the folder
        # already exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.list_of_outputs = []  # type: List[OutputFileRecord]

        self.report = f"""# {name}

{description}

"""

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

    def flush(self):
        """Terminate a simulation.

        This function must be called when a simulation is complete. It
        will save pending data to the output directory.

        """

        # Expand the markdown text using Jinja2
        with codecs.open(self.base_path / "report.md", "w", encoding="utf-8") as outf:
            outf.write(self.report)

        # Now generate an HTML file from Markdown.

        # Please keep these in alphabetic order, so we can detect duplicates!
        md_extensions = [KatexExtension(), "sane_lists", "smarty", "tables"]
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
        with codecs.open(
            self.base_path / "report.html", "w", encoding="utf-8",
        ) as outf:
            outf.write(html_full_report)

    def create_observations(
        self,
        detectors: List[Detector],
        num_of_obs_per_detector: int,
        start_time,
        duration_s: float,
        distribute=True,
        use_mjd=False,
    ):
        "Create a set of Observation objects"

        observations = []
        for detidx, cur_det in enumerate(detectors):
            cur_sampfreq_hz = cur_det.sampfreq_hz
            num_of_samples = cur_sampfreq_hz * duration_s
            samples_per_obs = distribute_evenly(num_of_samples, num_of_obs_per_detector)

            if isinstance(start_time, float):
                cur_time = start_time
            else:
                if use_mjd:
                    cur_time = start_time
                else:
                    cur_time = start_time.mjd

            for cur_obs_idx in range(num_of_obs_per_detector):
                nsamples = samples_per_obs[cur_obs_idx].num_of_elements
                cur_obs = Observation(
                    detector=cur_det,
                    start_time=cur_time,
                    sampfreq_hz=cur_sampfreq_hz,
                    nsamples=nsamples,
                    use_mjd=use_mjd,
                )
                observations.append(cur_obs)
                span_s = cur_sampfreq_hz * nsamples

                if use_mjd:
                    cur_time = Time(mjd=cur_time) + TimeDelta(seconds=span_s)
                else:
                    cur_time += span_s

        if distribute:
            self.distribute_workload(observations)

        return observations

    def distribute_workload(self, observations: List[Observation]):
        if self.mpi_comm.size == 1:
            self.observations = observations
            return

        cur_rank = self.mpi_comm.rank
        span = distribute_optimally(
            elements=observations,
            num_of_groups=self.mpi_comm.size,
            weight_fn=lambda obs: obs.nsamples,
        )[cur_rank]

        self.observations = observations[
            span.start_idx : (span.start_idx + span.num_of_elements)
        ]
