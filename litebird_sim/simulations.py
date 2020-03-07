# -*- encoding: utf-8 -*-

import codecs
from collections import namedtuple
from typing import List, Tuple, Any
from pathlib import Path
from shutil import copyfile, copytree

from . import write_healpix_map_to_file

import markdown
import jinja2

from markdown_katex import KatexExtension

OutputFileRecord = namedtuple("OutputFileRecord", ["path", "description"])


class Simulation:
    """A container object for running simulations

    Args:

        base_path (str or `pathlib.Path`): the folder that will
            contain the output. If this folder does not exist and the
            user has sufficient rights, it will be created.

        name (str): a string identifying the simulation. This will
            be used in the reports.

        use_mpi (bool): a Boolean flag specifying if the simulation
            should take advantage of MPI or not.

        description (str): a (possibly long) description of the
            simulation, to be put in the report saved in `base_path`).

    This is the most important class in the Litebird_sim framework. It
    initializes an output directory that will contain all the products
    of a simulation and will handle the generation of reports and
    writing of output files.

    Be sure to call :py:meth:`Simulation.flush` when the simulation is
    completed. This ensures that all the information are saved to disk
    before the completion of your script.

    You can access the fields `base_path`, `name`, `use_mpi`, and
    `description` in the `Simulation` object::

        sim = litebird_sim.Simulation(name="My simulation")
        print(f"Running {sim.name}, saving results in {sim.base_path}")


    This class keeps track of any output file saved in `base_path`
    through the member variable `self.list_of_outputs`. This is a list
    of objects of type :py:meth:`OutputFileRecord`, which are 2-tuples
    of the form ``(path, description)``, where ``path`` is a
    ``pathlib.Path`` object and ``description`` is a `str` object::

        for curpath, curdescr in sim.list_of_outputs:
            print(f"{curpath}: {curdescr}")

    """

    def __init__(
        self, base_path=Path(), name=None, use_mpi=False, description="",
    ):
        self.base_path = Path(base_path)
        self.name = name
        self.use_mpi = use_mpi
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
