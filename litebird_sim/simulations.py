# -*- encoding: utf-8 -*-

from pathlib import Path
from . import write_healpix_map_to_file


class Simulation:
    """A container object for running simulations

    This is the most important class in the Litebird_sim framework. It
    initializes an output directory that will contain all the products
    of a simulation and will handle the generation of reports and
    writing of output files.

    To create an object, you can pass one or more of the following
    keywords:

    - `base_path` (either a ``str`` or ``pathlib.Path`` object): the
      folder that will contain the output. If this folder does not
      exist and the user has sufficient rights, it will be created.

    - `name` (``str``): a string identifying the simulation. This will
      be used in the reports.

    - `use_mpi` (``bool``): a Boolean flag specifying if the simulation
      should take advantage of MPI or not.

    - `description` (``str``): a (possibly long) description of the
      simulation, to be put in the report saved in `base_path`).

    You can access the fields `base_path`, `name`, `use_mpi`, and
    `description` in the `Simulation` object::

        sim = litebird_sim.Simulation(name="My simulation")
        print(f"Running simulation {sim.name} and saving results in {sim.base_path}")

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

    def write_healpix_map(
        self, filename, pixels, **kwargs,
    ):
        """Save a Healpix map in the output folder

        This function saves an Healpix map into a FITS files that is
        written into the output folder for the simulation. It accepts
        the following arguments:

        - `filename` (``str`` or ``pathlib.Path``): name of the
          file. It can contain subdirectories.

        - `pixels`: array containing the pixels, or list of arrays if
          you want to save several maps into the same FITS table
          (e.g., I, Q, U components)

        Here is a simple example::

          import numpy as np

          sim = Simulation(base_path="/storage/litebird/mysim")
          pixels = np.zeros(12)
          sim.write_healpix_map("zero_map.fits.gz", pixels)

        """
        write_healpix_map_to_file(
            filename=self.base_path / Path(filename), pixels=pixels, **kwargs,
        )
